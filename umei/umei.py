from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn.functional import interpolate

from .model import UDecoderBase, UDecoderOutput, UEncoderBase, UEncoderOutput
from .args import UMeIArgs

class UMeI(LightningModule):
    cls_loss_fn: nn.Module
    seg_loss_fn: nn.Module

    def __init__(
        self,
        args: UMeIArgs,
        encoder: UEncoderBase,
        decoder: Optional[UDecoderBase] = None,
        num_seg_heads: int = 1,
    ):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        if self.args.num_cls_classes is not None:
            self.cls_head = nn.Linear(encoder.cls_feature_size + args.clinical_feature_size, args.num_cls_classes)
            # self.cls_loss_fn = nn.CrossEntropyLoss()
            nn.init.constant_(torch.as_tensor(self.cls_head.bias), 0)

        if decoder is not None:
            from monai.networks.blocks import UnetOutBlock
            # i-th seg head for the last i-th output from decoder
            self.seg_heads = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=3,
                    in_channels=args.base_feature_size ** 2 ** i,
                    out_channels=args.num_seg_classes,
                )
                for i in range(num_seg_heads)
            ])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        img = batch[self.args.img_key]
        encoder_out: UEncoderOutput = self.encoder(img)
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
            decoder_out: UDecoderOutput = self.decoder.forward(img, encoder_out.hidden_states)
            seg_loss = torch.sum(torch.stack([
                self.seg_loss_fn(
                    interpolate(seg_head(feature_map), seg_label.shape[2:], mode='trilinear'),
                    seg_label
                ) / 2 ** i
                for i, (feature_map, seg_head) in enumerate(zip(decoder_out.feature_maps[::-1], self.seg_heads))
            ])) / (2 * (1 - 1 / 2 ** len(self.seg_heads)))
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

    def output_seg(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.decoder.forward(x, self.encoder.forward(x).hidden_states).feature_maps[-1]
        return self.seg_heads[0](fm)
