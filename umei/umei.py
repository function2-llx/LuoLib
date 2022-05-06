from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.losses import DiceFocalLoss

from .model import UDecoderBase, UDecoderOutput, UEncoderBase, UEncoderOutput
from .utils import UMeIArgs

class UMeI(LightningModule):
    def __init__(
        self,
        args: UMeIArgs,
        encoder: UEncoderBase,
        decoder: Optional[UDecoderBase] = None,
        num_seg_classes: Optional[int] = None,
        num_seg_heads: int = 1,
    ):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_model = args.model_name
        self.cls_head = nn.Linear(encoder.cls_feature_size + args.clinical_feature_size, args.num_cls_classes)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        nn.init.constant_(torch.as_tensor(self.cls_head.bias), 0)

        if decoder is not None:
            assert 1 <= num_seg_heads <= len(decoder.feature_sizes)
            self.seg_heads = nn.ModuleList([
                nn.Conv3d(feature_size, num_seg_classes, kernel_size=1, stride=1)
                for feature_size in decoder.feature_sizes[::-1][:num_seg_heads]
            ])
            self.seg_loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        encoder_out: UEncoderOutput = self.encoder(batch[self.args.img_key])
        ret = {'loss': torch.tensor(0., device=self.device)}
        if self.args.cls_key in batch:
            cls_out = self.cls_head(torch.cat((encoder_out.cls_feature, batch[self.args.clinical_key]), dim=1))
            cls_loss = self.cls_loss_fn(cls_out, batch[self.args.cls_key])
            # self.log('cls_loss', cls_loss, prog_bar=True)
            ret['loss'] += cls_loss * self.args.cls_loss_factor
            ret['cls_loss'] = cls_loss
            ret['cls_logit'] = cls_out
        if self.decoder is not None and self.args.seg_key in batch:
            seg_label: torch.IntTensor = batch[self.args.seg_key]
            decoder_out: UDecoderOutput = self.decoder(encoder_out.feature_maps)
            seg_loss = torch.sum(torch.stack([
                self.seg_loss_fn(interpolate(seg_head(feature_map), seg_label.shape[2:]), seg_label) / 2 ** i
                for i, (feature_map, seg_head) in enumerate(zip(decoder_out.feature_maps[::-1], self.seg_heads))
            ])) / (1 - 1 / 2 ** len(self.seg_heads))
            # self.log('seg_loss', seg_loss, prog_bar=True)
            ret['loss'] += seg_loss * self.args.seg_loss_factor
            ret['seg_loss'] = seg_loss
        return ret

    def training_step(self, batch: dict, *args, **kwargs) -> STEP_OUTPUT:
        output = self.forward(batch)
        for k in ['cls_loss', 'seg_loss']:
            if k in output:
                self.log(f'train/{k}', output[k])
        return output

    def validation_step(self, splits_batch: dict[str, dict[str, torch.Tensor]], *args, **kwargs) -> Optional[STEP_OUTPUT]:
        splits_output = {}
        for split, batch in splits_batch.items():
            batch_size = batch[self.args.img_key].shape[0]
            output = self.forward(batch)
            for k in ['cls_loss', 'seg_loss']:
                if k in output:
                    self.log(f'{split}/{k}', output[k], batch_size=batch_size)
                    self.log(f'combined/{k}', output[k], batch_size=batch_size)
            splits_output[split] = output
        return splits_output

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    mode=self.args.monitor_mode,
                    factor=self.args.lr_reduce_factor,
                    patience=self.args.patience,
                    verbose=True,
                ),
                'monitor': f'combined/{self.args.monitor}',
            }
        }
