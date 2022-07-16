import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
import monai.transforms
from monai.utils import BlendMode, MetricReduction

from umei import UMeI
from umei.datasets.amos import AmosArgs

class AmosModel(UMeI):
    args: AmosArgs

    def __init__(self, args: AmosArgs):
        super().__init__(args, has_decoder=True)
        self.seg_loss_fn = DiceCELoss(
            include_background=self.args.dice_include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=self.args.squared_dice,
            smooth_nr=self.args.dice_nr,
            smooth_dr=self.args.dice_dr,
        )
        self.post_transform = monai.transforms.Compose([
            monai.transforms.KeepLargestConnectedComponent(is_onehot=False, applied_labels=args.post_labels),
        ])
        self.dice_metric = DiceMetric()
        self.resampler = monai.transforms.SpatialResample()

    def sw_infer(self, img: torch.Tensor):
        return sliding_window_inference(
            img,
            roi_size=self.args.sample_shape,
            sw_batch_size=self.args.sw_batch_size,
            predictor=self.forward,
            overlap=self.args.sw_overlap,
            mode=BlendMode.GAUSSIAN,
        )

    def on_validation_epoch_start(self):
        self.dice_metric.reset()

    def validation_step(self, batch: dict[str, dict[str, torch.Tensor]], *args, **kwargs):
        batch = batch['val']
        pred_logit = self.sw_infer(batch[self.args.img_key])
        pred = pred_logit.argmax(dim=1, keepdim=True)
        self.dice_metric(
            one_hot(pred, self.args.num_seg_classes),
            one_hot(batch[self.args.seg_key], self.args.num_seg_classes),
        )

    def validation_epoch_end(self, *args) -> None:
        dice = self.dice_metric.aggregate(reduction=MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i])
        self.log('val/dice/avg', dice[1:].mean())

    def on_test_epoch_start(self) -> None:
        self.dice_metric.reset()

    def test_step(self, batch, *args, **kwargs):
        img_meta_dict = batch[f'{self.args.img_key}_meta_dict']
        pred_logit = self.sw_infer(batch[self.args.img_key])
        pred_logit = self.resampler.__call__(
            pred_logit[0],
            src_affine=img_meta_dict['affine'],
            dst_affine=img_meta_dict['original_affine'],
            spatial_size=img_meta_dict['spatial_shape'],
        )
        pred = pred_logit.argmax(dim=0, keepdim=True)
        pred = self.post_transform(pred)
        self.dice_metric(
            # add dummy batch dim
            one_hot(pred.view(1, *pred.shape), self.args.num_seg_classes),
            one_hot(batch[self.args.seg_key], self.args.num_seg_classes),
        )

    def test_epoch_end(self, *args) -> None:
        dice = self.dice_metric.aggregate(reduction=MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'test/dice/{i}', dice[i])
        self.log('test/dice/avg', dice[1:].mean())

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': CosineAnnealingLR(optimizer, T_max=int(self.args.num_train_epochs)),
        }
