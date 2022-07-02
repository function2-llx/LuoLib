from argparse import Namespace
from functools import partial

from pytorch_lightning import LightningModule
import torch
from torch.optim import AdamW, Optimizer

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.blocks import UnetOutBlock
from monai.networks.nets import SwinTransformer, SwinUNETR, SwinUnetrDecoder
from monai.transforms import AsDiscrete
from monai.utils import MetricReduction

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.data_utils import get_loader
from utils.utils import AverageMeter

class AmosModel(LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args

        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        if args.split_model:
            self.encoder = SwinTransformer(
                in_chans=args.in_channels,
                embed_dim=args.feature_size,
                window_size=(7, 7, 7),
                patch_size=(2, 2, 2),
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                # mlp_ratio=4.0,
                # qkv_bias=True,
                use_checkpoint=True,
            )
            self.decoder = SwinUnetrDecoder(args.in_channels, feature_size=args.feature_size)
            self.seg_head = UnetOutBlock(
                spatial_dims=3,
                in_channels=args.feature_size,
                out_channels=args.out_channels,
            )
        else:
            self.model = SwinUNETR(
                img_size=inf_size,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size=args.feature_size,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=args.dropout_path_rate,
                use_checkpoint=args.use_checkpoint,
            )

        if args.squared_dice:
            self.loss_func = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=True,
                smooth_nr=args.smooth_nr,
                smooth_dr=args.smooth_dr,
            )
        else:
            self.loss_func = DiceCELoss(to_onehot_y=True, softmax=True)

        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=self.forward,
            overlap=args.infer_overlap,
        )

        self.post_label = AsDiscrete(to_onehot=args.out_channels)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
        self.acc_func = DiceMetric(
            include_background=False,
            reduction=MetricReduction.MEAN,
            get_not_nans=True,
        )

        if args.use_ssl_pretrained:
            try:
                weight = torch.load('./pretrained_models/model_swinvit.pt')
                self.model.load_from(weights=weight)
                print('Using pretrained self-supervied Swin UNETR backbone weights !')
            except ValueError:
                raise ValueError('Self-supervised pre-trained weights not available for' + str(args.model_name))

        loader = get_loader(args)
        self.train_loader = loader[0]
        self.val_loader = loader[1]

        self.run_loss = AverageMeter()
        self.run_acc = AverageMeter()

    def forward(self, x: torch.Tensor):
        if self.args.split_model:
            e_out = self.encoder.forward(x)
            fm = self.decoder.forward(x, e_out.hidden_states).feature_maps[-1]
            return self.seg_head(fm)
        else:
            return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        assert self.args.optim_name == 'adamw'
        optimizer = AdamW(
            self.parameters(),
            lr=self.args.optim_lr,
            weight_decay=self.args.reg_weight
        )
        assert self.args.lrschedule == 'warmup_cosine'
        return {
            'optimizer': optimizer,
            'lr_scheduler': LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.args.warmup_epochs,
                max_epochs=self.args.max_epochs
            )
        }

    def on_train_epoch_start(self):
        self.run_loss.reset()

    def training_step(self, batch_data, idx, *args, **kwargs):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        for param in self.parameters():
            param.grad = None
        logits = self.forward(data)
        loss = self.loss_func(logits, target)
        self.log('train/loss', loss.item())
        self.run_loss.update(loss.item(), n=data.shape[0])
        return loss

    def on_train_epoch_end(self):
        # self.log('train/loss', self.run_loss.avg)
        for param in self.parameters():
            param.grad = None

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        pass

    def on_validation_epoch_start(self):
        self.run_acc.reset()

    def validation_step(self, batch_data, idx, *args, **kwargs):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        logits = self.model_inferer(data)
        val_labels_list = decollate_batch(target)
        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        self.acc_func.reset()
        self.acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc, not_nans = self.acc_func.aggregate()

        self.run_acc.update(acc.item(), n=not_nans.item())

    def validation_epoch_end(self, _outputs):
        self.log('val/acc', self.run_acc.avg * 100)
