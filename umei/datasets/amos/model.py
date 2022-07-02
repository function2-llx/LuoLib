import torch
from torch.optim import AdamW

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
import monai.transforms
from monai.utils import BlendMode, MetricReduction

from swin_unetr.BTCV.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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
            # monai.transforms.Activations(softmax=True),
            # monai.transforms.AsDiscrete(argmax=True),
            monai.transforms.KeepLargestConnectedComponent(is_onehot=False),
            # monai.transforms.FillHoles(),
        ])
        self.dice_metric = DiceMetric(reduction=MetricReduction.MEAN_BATCH)

    def validation_step(self, batch: dict[str, dict[str, torch.Tensor]], *args, **kwargs):
        batch = batch['val']
        pred_logit = sliding_window_inference(
            batch[self.args.img_key],
            roi_size=self.args.sample_shape,
            sw_batch_size=self.args.sw_batch_size,
            predictor=self.forward,
            overlap=self.args.val_sw_overlap,
            mode=BlendMode.GAUSSIAN,
        )
        pred = pred_logit.argmax(dim=1, keepdim=True)
        if self.args.val_post:
            pred = torch.stack([self.post_transform(p) for p in pred])
        self.dice_metric(
            one_hot(pred, self.args.num_seg_classes),
            one_hot(batch[self.args.seg_key], self.args.num_seg_classes),
        )

    def validation_epoch_end(self, *args) -> None:
        dice = self.dice_metric.aggregate() * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i])
        self.log('val/dice/avg', dice[1:].mean())

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.args.warmup_epochs,
                    max_epochs=int(self.args.num_train_epochs),
                ),
                'monitor': self.args.monitor,
            }
        }
