import torch
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
import monai.transforms
from monai.utils import MetricReduction
from umei import UDecoderBase, UEncoderBase, UMeI
from umei.datasets.amos import AmosArgs

class AmosModel(UMeI):
    args: AmosArgs

    def __init__(self, args: AmosArgs, encoder: UEncoderBase, decoder: UDecoderBase):
        super().__init__(args, encoder, decoder)
        self.seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, include_background=False)
        self.post_transform = monai.transforms.Compose([
            # monai.transforms.Activations(softmax=True),
            # monai.transforms.AsDiscrete(argmax=True),
            monai.transforms.KeepLargestConnectedComponent(is_onehot=False),
            monai.transforms.FillHoles(),
        ])
        self.dice_metric = DiceMetric(reduction=MetricReduction.MEAN_BATCH)

    def sw_infer(self, batch_img: torch.Tensor, overlap: float, post: bool) -> torch.Tensor:
        logit = sliding_window_inference(
            batch_img,
            roi_size=self.args.sample_shape,
            sw_batch_size=self.args.sw_batch_size,
            predictor=self.output_seg,
            overlap=overlap,
        )
        pred = logit.argmax(dim=1, keepdim=True)
        if post:
            pred = torch.stack([self.post_transform(p) for p in pred])
        return pred

    def validation_step(self, batch: dict[str, dict[str, torch.Tensor]], *args, **kwargs):
        batch = batch['val']
        pred = self.sw_infer(batch[self.args.img_key], overlap=0.1, post=False)
        self.dice_metric(
            one_hot(pred, self.args.num_seg_classes),
            one_hot(batch[self.args.seg_key], self.args.num_seg_classes),
        )

    def validation_epoch_end(self, *args) -> None:
        dice = self.dice_metric.aggregate() * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i].item())
        self.log('val/dice/avg', dice[1:].mean().item())

    def predict_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        return self.sw_infer(batch[self.args.img_key], overlap=0.8, post=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=int(self.args.num_train_epochs),
                ),
                'monitor': self.args.monitor,
            }
        }
