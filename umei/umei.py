from __future__ import annotations

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.optim import Optimizer

from .args import UMeIArgs
from .model import UEncoderOutput, build_decoder, build_encoder

class UMeI(LightningModule):
    cls_loss_fn: nn.Module
    seg_loss_fn: nn.Module

    def __init__(self, args: UMeIArgs, *, has_decoder: bool):
        super().__init__()
        self.args = args
        self.encoder = build_encoder(args)
        with torch.no_grad():
            self.encoder.eval()
            dummy_input = torch.zeros(1, args.num_input_channels, *args.sample_shape)
            dummy_encoder_output = self.encoder.forward(dummy_input)
            encoder_feature_sizes = [feature.shape[1] for feature in dummy_encoder_output.hidden_states]
        if self.args.num_cls_classes is not None:
            encoder_cls_feature_size = dummy_encoder_output.cls_feature.shape[1]
            self.cls_head = nn.Linear(encoder_cls_feature_size + args.clinical_feature_size, args.num_cls_classes)
            nn.init.constant_(torch.as_tensor(self.cls_head.bias), 0)

        self.decoder = None
        if has_decoder:
            self.decoder = build_decoder(args, encoder_feature_sizes)
            with torch.no_grad():
                dummy_decoder_output = self.decoder.forward(dummy_input, dummy_encoder_output.hidden_states)
                decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]
            from monai.networks.blocks import UnetOutBlock
            # i-th seg head for the last i-th output from decoder
            self.seg_heads = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=3,
                    in_channels=decoder_feature_sizes[-i - 1],
                    out_channels=args.num_seg_classes,
                )
                for i in range(args.num_seg_heads)
            ])

    def training_step(self, batch: dict, *args, **kwargs) -> STEP_OUTPUT:
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
            feature_maps = self.decoder.forward(img, encoder_out.hidden_states).feature_maps
            seg_loss = torch.stack([
                self.seg_loss_fn(
                    interpolate(seg_head(fm), seg_label.shape[2:], mode='trilinear'),
                    seg_label
                ) / 2 ** i
                for i, (fm, seg_head) in enumerate(zip(reversed(feature_maps), self.seg_heads))
            ]).sum()
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
        feature_maps = self.decoder.forward(x, output.hidden_states).feature_maps
        if self.args.self_ensemble:
            return torch.stack([
                interpolate(seg_head(fm), x.shape[2:], mode='trilinear')
                for fm, seg_head in zip(reversed(feature_maps), self.seg_heads)
            ]).mean(dim=0)
        else:
            return interpolate(self.seg_heads[0](feature_maps[-1]), x.shape[2:], mode='trilinear')

    def optimizer_zero_grad(self, _epoch, _batch_idx, optimizer: Optimizer, _optimizer_idx):
        optimizer.zero_grad(set_to_none=self.args.optimizer_set_to_none)
