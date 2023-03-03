import json
from pathlib import Path

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.scheduler.scheduler import Scheduler
import torch
from torch import nn
from torch.nn import functional as torch_f
from torch.optim import Optimizer

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.umei import Backbone, BackboneOutput, Decoder
from monai.utils import MetricReduction, ensure_tuple

from umei.models import create_model
from umei.omega import ExpConfBase, SegExpConf
from umei.utils import DataKey

class SimpleReprMixin(object):
    """A mixin implementing a simple __repr__."""
    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

class ExpModelBase(LightningModule):
    def __init__(self, conf: ExpConfBase):
        super().__init__()
        self.conf = conf
        self.backbone: Backbone = create_model(conf.backbone)
        self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

    def dummy(self):
        with torch.no_grad():
            self.backbone.eval()
            dummy_input = torch.zeros(1, self.conf.num_input_channels, *self.conf.sample_shape)
            dummy_output = self.backbone.forward(dummy_input)
            print('backbone output shapes:')
            for x in dummy_output.feature_maps:
                print(x.shape)
        return dummy_input, dummy_output

    @property
    def log_exp_dir(self) -> Path:
        from pytorch_lightning.loggers import WandbLogger
        logger: WandbLogger = self.trainer.logger   # type: ignore
        return Path(logger.experiment.dir)

    def on_fit_start(self) -> None:
        with open(self.log_exp_dir / 'fit-summary.txt', 'w') as f:
            print(self, file=f, end='\n\n\n')
            print('optimizers:\n', file=f)
            for optimizer in ensure_tuple(self.optimizers()):
                print(optimizer, file=f)
            print('\n\n', file=f)
            print('schedulers:\n', file=f)
            for scheduler in ensure_tuple(self.lr_schedulers()):
                print(scheduler, file=f)

    def get_grouped_parameters(self) -> list[dict]:
        # modify from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        from torch.nn.modules.conv import _ConvNd
        whitelist_weight_modules = (
            nn.Linear,
            _ConvNd,
        )
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        blacklist_weight_modules = (
            nn.LayerNorm,
            _BatchNorm,
            _InstanceNorm,
            nn.Embedding,
        )
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            m: nn.Module
            if hasattr(m, 'no_weight_decay'):
                for pn in m.no_weight_decay():
                    no_decay.add(f'{mn}.{pn}' if mn else pn)

            for pn, p in m.named_parameters(prefix=mn, recurse=False):
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(pn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(pn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(pn)

        # validate that we considered every parameter
        params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(params_dict.keys() - union_params) == 0, \
            f"parameters {params_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        with open(self.log_exp_dir / 'parameters.json', 'w') as f:
            obj = {
                'decay': list(decay),
                'no decay': list(no_decay),
            }
            json.dump(obj, f, indent=4, ensure_ascii=False)
        # create the pytorch optimizer object
        return [
            {"params": [params_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.conf.optimizer.weight_decay},
            {"params": [params_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0},
        ]

    def configure_optimizers(self):
        from timm.optim import create_optimizer_v2
        optim = self.conf.optimizer
        optimizer = create_optimizer_v2(
            self.get_grouped_parameters(),  # type: ignore
            optim.name,
            optim.lr,
            optim.weight_decay,
            **optim.kwargs,
        )
        from timm.scheduler import create_scheduler
        scheduler = create_scheduler(self.conf.scheduler, optimizer)[0]
        if type(scheduler).__repr__ == object.__repr__:
            type(scheduler).__repr__ = SimpleReprMixin.__repr__
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.conf.monitor,
        }

    def lr_scheduler_step(
        self,
        scheduler: Scheduler,
        optimizer_idx: int,
        metric,
    ) -> None:
        return scheduler.step(self.current_epoch, metric)

    def optimizer_zero_grad(self, _epoch, _batch_idx, optimizer: Optimizer, _optimizer_idx):
        optimizer.zero_grad(set_to_none=self.conf.optimizer_set_to_none)

class SegModel(ExpModelBase):
    conf: SegExpConf

    def __init__(self, conf: SegExpConf):
        super().__init__(conf)
        self.decoder: Decoder = create_model(conf.decoder)
        with torch.no_grad():
            self.decoder.eval()
            dummy_input, dummy_encoder_output = super().dummy()
            dummy_decoder_output = self.decoder.forward(dummy_encoder_output.feature_maps, dummy_input)
            print('decoder output shapes:')
            for x in dummy_decoder_output.feature_maps:
                print(x.shape)
            decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]
        # decoder feature map: from small to large
        # i-th seg head for the last i-th output from decoder, i.e., the 0-th seg head for the largest output
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(decoder_feature_sizes[-i - 1], conf.num_seg_classes, kernel_size=1)
            for i in range(conf.num_seg_heads)
        ])
        seg_head_weights = torch.tensor([0.5 ** i for i in range(conf.num_seg_heads)])
        self.seg_head_weights = nn.Parameter(seg_head_weights / seg_head_weights.sum(), requires_grad=False)
        for seg_head in self.seg_heads:
            nn.init.trunc_normal_(seg_head.weight, std=0.02)
            if seg_head.bias is not None:
                nn.init.zeros_(seg_head.bias)

        self.seg_loss_fn = DiceCELoss(
            include_background=self.conf.dice_include_background,
            to_onehot_y=not conf.multi_label,
            sigmoid=conf.multi_label,
            softmax=not conf.multi_label,
            squared_pred=self.conf.dice_squared,
            smooth_nr=self.conf.dice_nr,
            smooth_dr=self.conf.dice_dr,
        )
        # metric for val
        self.dice_metric = DiceMetric(include_background=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.forward(x)
        feature_maps = self.decoder.forward(output.feature_maps, x).feature_maps
        if self.conf.self_ensemble:
            return torch.stack([
                torch_f.interpolate(seg_head(fm), x.shape[2:], mode='trilinear')
                for fm, seg_head in zip(reversed(feature_maps), self.seg_heads)
            ]).mean(dim=0)
        else:
            ret = self.seg_heads[0](feature_maps[-1])
            if ret.shape[2:] != x.shape[2:]:
                ret = torch_f.interpolate(ret, x.shape[2:], mode='trilinear')
            return ret

    def compute_loss(self, output_logits: list[torch.Tensor] | torch.Tensor, seg_label: torch.Tensor):
        if isinstance(output_logits, list):
            seg_loss = torch.stack([
                self.seg_loss_fn(
                    torch_f.interpolate(output_logit, seg_label.shape[2:], mode='trilinear'),
                    seg_label,
                )
                for output_logit in output_logits
            ])
            return seg_loss[0], torch.dot(seg_loss, self.seg_head_weights)
        else:
            return self.seg_loss_fn(output_logits, seg_label)

    def training_step(self, batch: dict, *args, **kwargs) -> STEP_OUTPUT:
        img = batch[DataKey.IMG]
        seg_label = batch[DataKey.SEG]
        backbone_output: BackboneOutput = self.backbone(img)
        decoder_output = self.decoder.forward(backbone_output.feature_maps, img)
        seg_outputs = [
            seg_head(feature_map)
            for seg_head, feature_map in zip(self.seg_heads, reversed(decoder_output.feature_maps))
        ]
        single_loss, ds_loss = self.compute_loss(seg_outputs, seg_label)
        self.log('train/single_loss', single_loss)
        self.log('train/ds_loss', ds_loss)
        return ds_loss

    def sw_infer(self, img: torch.Tensor, progress: bool = None, softmax: bool = False):
        ret = sliding_window_inference(
            img,
            roi_size=self.conf.sample_shape,
            sw_batch_size=self.conf.sw_batch_size,
            predictor=self.forward,
            overlap=self.conf.sw_overlap,
            mode=self.conf.sw_blend_mode,
            progress=progress,
        )
        if softmax:
            ret = ret.softmax(dim=1)
        return ret

    def tta_infer(self, img: torch.Tensor, progress: bool = None, softmax: bool = False):
        pred = self.sw_infer(img, progress, softmax)
        for flip_idx in self.tta_flips:
            pred += torch.flip(self.sw_infer(torch.flip(img, flip_idx), progress, softmax), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def infer(self, img: torch.Tensor, progress: bool = None, tta_softmax: bool = False):
        if progress is None:
            progress = self.trainer.testing if self._trainer is not None else True

        if self.conf.do_tta:
            return self.tta_infer(img, progress, tta_softmax)
        else:
            return self.sw_infer(img, progress)

    def on_validation_epoch_start(self):
        if self.conf.val_empty_cuda_cache:
            torch.cuda.empty_cache()
        self.dice_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        seg = batch[DataKey.SEG]
        pred_logit = self.sw_infer(batch[DataKey.IMG])
        pred_logit = torch_f.interpolate(
            pred_logit,
            seg.shape[2:],
            mode='trilinear',
        )
        loss = self.compute_loss(pred_logit, seg)
        self.log('val/loss', loss, sync_dist=True)

        if self.conf.multi_label:
            pred = (pred_logit.sigmoid() > 0.5).long()
        else:
            pred = pred_logit.argmax(dim=1, keepdim=True)
            pred = one_hot(pred, self.conf.num_seg_classes)
            seg = one_hot(seg, self.conf.num_seg_classes)
        self.dice_metric(pred, seg)

    def validation_epoch_end(self, *_) -> None:
        if self.conf.val_empty_cuda_cache:
            torch.cuda.empty_cache()
        dice = self.dice_metric.aggregate(reduction=MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i], sync_dist=True)
        if self.conf.multi_label:
            avg = dice.mean()
        else:
            avg = dice[1:].mean()
        self.log('val/dice/avg', avg, sync_dist=True)
