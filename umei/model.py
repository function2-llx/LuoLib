from typing import Optional

import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as torch_f
from torch.optim import AdamW, Optimizer, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers import Conv
from monai.umei import Decoder, Backbone, BackboneOutput
from monai.utils import MetricReduction

from umei.args import SegArgs, UMeIArgs
from umei.utils import DataKey

def filter_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict:
    return {
        k.split('.', 1)[1]: v
        for k, v in state_dict.items()
        if k.startswith(f'{prefix}.')
    }

class UMeI(LightningModule):
    cls_loss_fn: nn.Module
    seg_loss_fn: nn.Module

    def __init__(self, args: UMeIArgs, **kwargs):
        super().__init__()
        self.args = args
        self.encoder = self.build_encoder()
        with torch.no_grad():
            self.encoder.eval()
            dummy_input = torch.zeros(1, args.num_input_channels, *args.sample_shape)
            dummy_encoder_output = self.encoder.forward(dummy_input)
            print('backbone output shapes:')
            for x in dummy_encoder_output.feature_maps:
                print(x.shape)
        if self.args.num_cls_classes is not None:
            self.cls_head = self.build_cls_head(dummy_encoder_output)
        self.decoder = self.build_decoder(dummy_encoder_output)
        if self.decoder is not None:
            with torch.no_grad():
                self.decoder.eval()
                dummy_decoder_output = self.decoder.forward(dummy_encoder_output.feature_maps, dummy_input)
                print('decoder output shapes:')
                for x in dummy_decoder_output.feature_maps:
                    print(x.shape)
                decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]
            from monai.networks.blocks import UnetOutBlock
            # i-th seg head for the last i-th output from decoder
            self.seg_heads = nn.ModuleList([
                Conv[Conv.CONV, self.args.spatial_dims](decoder_feature_sizes[-i - 1], args.num_seg_classes, kernel_size=1)
                # UnetOutBlock(args.spatial_dims, decoder_feature_sizes[-i - 1], args.num_seg_classes)
                for i in range(args.num_seg_heads)
            ])
        if self.args.spatial_dims == 2:
            self.tta_flips = [[2], [3], [2, 3]]
        else:
            self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

    def build_cls_head(self, dummy_encoder_output: BackboneOutput):
        encoder_cls_feature_size = dummy_encoder_output.cls_feature.shape[1]
        cls_head = nn.Linear(encoder_cls_feature_size + self.args.clinical_feature_size, self.args.num_cls_classes)
        nn.init.constant_(torch.as_tensor(cls_head.bias), 0)
        return cls_head

    # TODO: refactor these following mmseg
    def build_encoder(self) -> Backbone:
        match self.args.backbone:
            case 'resnet':
                from monai.networks import nets
                resnet_builder = getattr(nets, f'resnet{self.args.model_depth}')
                model: nn.Module = resnet_builder(
                    n_input_channels=self.args.num_input_channels,
                    feed_forward=False,
                    conv1_t_size=self.args.resnet_conv1_size,
                    conv1_t_stride=self.args.resnet_conv1_stride,
                    layer1_stride=self.args.resnet_layer1_stride,
                    shortcut_type=self.args.resnet_shortcut,
                )

                if self.args.pretrain_path is not None:
                    # assume pre-trained weights are from https://github.com/Tencent/MedicalNet
                    dp_model = nn.DataParallel(model)
                    state_dict = dp_model.state_dict()
                    pretrain_state_dict = torch.load(self.args.pretrain_path, map_location='cpu')['state_dict']
                    state_dict.update({
                        k: v for k, v in pretrain_state_dict.items()
                        if k in state_dict and k != 'module.conv1.weight'
                    })
                    dp_model.load_state_dict(state_dict)
                    model: nets.ResNet = dp_model.module  # type: ignore

                    # handle number of input channels that is possible different from the pre-trained model
                    for attr in ['weight', 'bias']:
                        param: Optional[nn.Parameter] = getattr(model.conv1, attr, None)
                        pretrain_param_data: Optional[torch.Tensor] = getattr(pretrain_state_dict,
                                                                              f'module.conv1.{attr}',
                                                                              None)
                        if param is not None and pretrain_param_data is not None:
                            param.data = pretrain_param_data.repeat(1, self.args.num_input_channels)
                    print(f'load pre-trained med-3d weights from {self.args.pretrain_path}')
                return model
            case 'vit':
                from monai.networks.nets import ViT
                return ViT(
                    in_channels=self.args.num_input_channels,
                    img_size=self.args.sample_shape,
                    patch_size=self.args.vit_patch_shape,
                    hidden_size=self.args.vit_hidden_size,
                    classification=False,
                )
            case 'swt':
                if self.args.umei_impl:
                    from umei.models.swin_monai import SwinTransformer
                    model = SwinTransformer(
                        in_chans=self.args.num_input_channels,
                        embed_dim=self.args.base_feature_size,
                        window_size=self.args.swin_window_size,
                        patch_size=self.args.vit_patch_shape,
                        depths=self.args.vit_depths,
                        num_heads=self.args.num_heads,
                        use_checkpoint=True,
                        conv_stem=self.args.vit_conv_stem,
                    )
                    if self.args.pretrain_path is not None:
                        if not self.args.pretrain_path.exists():
                            print(f'warning: {self.args.pretrain_path} not found')
                        else:
                            state_dict = filter_state_dict(torch.load(self.args.pretrain_path)["state_dict"], 'encoder')
                            miss, unexpected = model.load_state_dict(state_dict, strict=False)
                            assert len(miss) == 0
                            print(f'load pre-trained encoder from {self.args.pretrain_path}')
                            print('unexpected: ', len(unexpected))
                            print(unexpected)
                else:
                    from monai.networks.nets.swin_unetr import SwinTransformer
                    model = SwinTransformer(
                        in_chans=self.args.num_input_channels,
                        embed_dim=self.args.base_feature_size,
                        window_size=self.args.swin_window_size,
                        patch_size=self.args.vit_patch_shape,
                        depths=self.args.vit_depths,
                        num_heads=self.args.num_heads,
                        use_checkpoint=True,
                    )
                    if self.args.pretrain_path is not None:
                        # assume weights from https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/
                        state_dict = {
                            k.split('.', 1)[1].replace('fc', 'linear'): v
                            for k, v in torch.load(self.args.pretrain_path)["state_dict"].items()
                            if k.startswith('swinViT.') or k.startswith('module.')
                        }
                        miss, unexpected = model.load_state_dict(state_dict, strict=False)
                        assert len(miss) == 0
                        print(f'load pre-trained swin-unetr encoder from {self.args.pretrain_path}')
                        print('unexpected: ', len(unexpected))
                return model
            case 'swin':
                from umei.models.backbones.swin import SwinBackbone
                args = self.args
                model = SwinBackbone(
                    args.num_input_channels,
                    args.layer_channels,
                    args.kernel_sizes,
                    args.layer_depths,
                    args.num_heads,
                    drop_path_rate=args.drop_path_rate,
                    use_checkpoint=args.gradient_checkpointing,
                    stem_stride=args.stem_stride,
                )
                return model
            case _:
                raise ValueError(f'not supported encoder: {self.args.backbone}')

    def build_decoder(self, dummy_backbone_output: BackboneOutput) -> Optional[Decoder]:
        match self.args.decoder:
            case 'sunetr':
                if self.args.umei_impl:
                    from umei.models.swin_unetr_decoder import SwinUnetrDecoder
                    input_stride = None
                    if self.args.umei_sunetr_decode_use_in:
                        input_stride = [patch_size // 2 for patch_size in self.args.vit_patch_shape]
                    model = SwinUnetrDecoder(
                        in_channels=self.args.num_input_channels,
                        feature_size=self.args.base_feature_size,
                        num_layers=len(self.args.vit_depths),
                        input_stride=input_stride,
                    )
                    if self.args.decoder_pretrain_path is not None:
                        state_dict = filter_state_dict(torch.load(self.args.decoder_pretrain_path)["state_dict"], 'decoder')
                        miss, unexpected = model.load_state_dict(state_dict, strict=False)
                        assert len(unexpected) == 0
                        print('missing:', len(miss))
                        print(miss)
                        print(f'load pre-trained decoder from {self.args.pretrain_path}')
                else:
                    from monai.networks.nets import SwinUnetrDecoder
                    model = SwinUnetrDecoder(
                        in_channels=self.args.num_input_channels,
                        feature_size=self.args.base_feature_size,
                        use_encoder5=self.args.use_encoder5,
                    )
                    if self.args.decoder_pretrain_path is not None:
                        # assume weights from https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/
                        state_dict = {
                            k: v
                            for k, v in torch.load(self.args.pretrain_path)["state_dict"].items()
                            if not k.startswith('swinViT.') and not k.startswith('out.')
                        }
                        miss, unexpected = model.load_state_dict(state_dict, strict=False)
                        assert len(miss) == 0 and len(unexpected) == 0
                        print(f'load pre-trained swin-unetr decoder from {self.args.pretrain_path}')
                return model
            case 'conv':
                from umei.models.decoders.plain_conv_unet import PlainConvUNetDecoder
                args = self.args
                model = PlainConvUNetDecoder(
                    self.args.num_input_channels,
                    [x.shape[1] for x in dummy_backbone_output.feature_maps],
                    num_post_upsamplings=args.num_post_upsamplings,
                )
                return model
            case _:
                raise ValueError(f'not supported decoder: {self.args.decoder}')

    def training_step(self, batch: dict, *args, **kwargs) -> STEP_OUTPUT:
        img = batch[DataKey.IMG]
        encoder_out: BackboneOutput = self.encoder(img)
        ret = {'loss': torch.tensor(0., device=self.device)}
        if DataKey.CLS in batch:
            cls_feature = encoder_out.cls_feature
            if DataKey.CLINICAL in batch:
                cls_feature = torch.cat((cls_feature, batch[DataKey.CLINICAL]), dim=1)
            cls_out = self.cls_head(cls_feature)
            cls_loss = self.cls_loss_fn(cls_out, batch[DataKey.CLS])
            # self.log('cls_loss', cls_loss, prog_bar=True)
            ret['loss'] += cls_loss * self.args.cls_loss_factor
            ret['cls_loss'] = cls_loss
            ret['cls_logit'] = cls_out
        if self.decoder is not None and DataKey.SEG in batch:
            seg_label = batch[DataKey.SEG]
            feature_maps = self.decoder.forward(encoder_out.feature_maps, img).feature_maps
            seg_loss = torch.stack([
                self.seg_loss_fn(
                    torch_f.interpolate(seg_head(fm), seg_label.shape[2:], mode=self.args.interpolate),
                    seg_label
                ) / 2 ** i
                for i, (fm, seg_head) in enumerate(zip(reversed(feature_maps), self.seg_heads))
            ]).sum() / (2 - 0.5 ** (self.args.num_seg_heads - 1))
            ret['loss'] += seg_loss * self.args.seg_loss_factor
            ret['seg_loss'] = seg_loss
        for k in ['cls_loss', 'seg_loss']:
            if k in ret:
                self.log(f'train/{k}', ret[k])
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder.forward(x)
        if self.decoder is None:
            return output.cls_feature
        feature_maps = self.decoder.forward(output.feature_maps, x).feature_maps
        if self.args.self_ensemble:
            return torch.stack([
                torch_f.interpolate(seg_head(fm), x.shape[2:], mode='trilinear')
                for fm, seg_head in zip(reversed(feature_maps), self.seg_heads)
            ]).mean(dim=0)
        else:
            ret = self.seg_heads[0](feature_maps[-1])
            if ret.shape[2:] != x.shape[2:]:
                ret = torch_f.interpolate(ret, x.shape[2:], mode='trilinear')
            return ret

    def forward_cls(self, img: torch.Tensor, clinical: Optional[torch.Tensor] = None, tta: bool = False):
        if tta:
            logit = self.forward_cls(img, clinical, False)
            for flip_idx in self.tta_flips:
                logit += self.forward_cls(torch.flip(img, flip_idx), clinical, False)
            logit /= len(self.tta_flips) + 1
        else:
            cls_feature = self.encoder.forward(img).cls_feature
            if clinical is not None:
                cls_feature = torch.cat((cls_feature, clinical), dim=1)
            logit = self.cls_head(cls_feature)
        return logit

    def get_grouped_parameters(self) -> list[dict]:
        return [{
            'params': self.parameters(),
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }]

    def get_optimizer(self):
        optimizer_cls = {
            'adamw': AdamW,
            'radam': RAdam,
        }[self.args.optim]
        optimizer = optimizer_cls(
            self.get_grouped_parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

    def get_lr_scheduler(self, optimizer):
        # from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
        from umei.scheduler import LinearWarmupCosineAnnealingLR

        if self.args.warmup_epochs:
            return LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.args.warmup_epochs,
                max_epochs=int(self.args.num_train_epochs),
            )
        else:
            return CosineAnnealingLR(
                optimizer,
                T_max=int(self.args.num_train_epochs),
            )

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return {
            'optimizer': optimizer,
            'lr_scheduler': self.get_lr_scheduler(optimizer),
            'monitor': self.args.monitor,
        }

    def optimizer_zero_grad(self, _epoch, _batch_idx, optimizer: Optimizer, _optimizer_idx):
        optimizer.zero_grad(set_to_none=self.args.optimizer_set_to_none)

class SegModel(UMeI):
    args: SegArgs

    def __init__(self, args: SegArgs):
        super().__init__(args, has_decoder=True)
        self.args = args
        self.seg_loss_fn = DiceFocalLoss(
            include_background=self.args.include_background,
            to_onehot_y=not args.mc_seg,
            sigmoid=args.mc_seg,
            softmax=not args.mc_seg,
            squared_pred=self.args.squared_dice,
            smooth_nr=self.args.dice_nr,
            smooth_dr=self.args.dice_dr,
        )
        # metric for val
        self.dice_metric = DiceMetric(include_background=True)

    def sw_infer(self, img: torch.Tensor, progress: bool = None):
        if progress is None:
            progress = self.trainer.testing if self._trainer is not None else True

        return sliding_window_inference(
            img,
            roi_size=self.args.sample_shape,
            sw_batch_size=self.args.sw_batch_size,
            predictor=self.forward,
            overlap=self.args.sw_overlap,
            mode=self.args.sw_blend_mode,
            progress=progress,
        )

    def tta_infer(self, img: torch.Tensor, progress: bool = None):
        pred_logit = self.sw_infer(img, progress)
        for flip_idx in self.tta_flips:
            pred_logit += torch.flip(self.sw_infer(torch.flip(img, flip_idx)), flip_idx)
        pred_logit /= len(self.tta_flips) + 1
        return pred_logit

    def infer_logit(self, img: torch.Tensor, progress: bool = None):
        if self.args.do_tta:
            return self.tta_infer(img, progress)
        else:
            return self.sw_infer(img, progress)

    def on_validation_epoch_start(self):
        if self.args.val_empty_cuda_cache:
            torch.cuda.empty_cache()
        self.dice_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        # batch = batch[DataSplit.VAL]
        seg = batch[DataKey.SEG]
        pred_logit = self.sw_infer(batch[DataKey.IMG])
        pred_logit = torch_f.interpolate(
            pred_logit,
            seg.shape[2:],
            mode=self.args.interpolate,
        )

        if self.args.mc_seg:
            pred = (pred_logit.sigmoid() > 0.5).long()
        else:
            pred = pred_logit.argmax(dim=1, keepdim=True)
            pred = one_hot(pred, self.args.num_seg_classes)
            seg = one_hot(seg, self.args.num_seg_classes)
        self.dice_metric(pred, seg)

    def validation_epoch_end(self, *args) -> None:
        if self.args.val_empty_cuda_cache:
            torch.cuda.empty_cache()
        dice = self.dice_metric.aggregate(reduction=MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i], sync_dist=True)
        if self.args.mc_seg:
            avg = dice.mean()
        else:
            avg = dice[1:].mean()
        self.log('val/dice/avg', avg, sync_dist=True)
