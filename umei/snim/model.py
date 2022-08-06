from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Type

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import distributed as dist, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.data import MetaTensor
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers import Act, Norm
from monai.utils import ImageMetaKey
from umei.models.swin import SwinTransformer
from .args import MaskValue, SnimArgs
from .utils import ChannelFirst, ChannelLast, channel_first, channel_last, patchify, unpatchify

class SnimEncoder(SwinTransformer):
    def __init__(self, args: SnimArgs):
        super().__init__(
            in_chans=args.num_input_channels,
            embed_dim=args.base_feature_size,
            window_size=args.swin_window_size,
            patch_size=args.vit_patch_shape,
            depths=args.vit_depths,
            num_heads=args.vit_num_heads,
            use_checkpoint=True,
        )
        self.args = args

        if args.mask_value == MaskValue.PARAM:
            self.mask_token = nn.Parameter(torch.empty(1, 1, self.patch_embed.embed_dim))

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p_x = patchify(x.clone(), self.patch_size)
        if self.args.mask_value == MaskValue.UNIFORM:
            p_x[mask] = torch.rand(mask.sum(), p_x.shape[-1], device=x.device)
        elif self.args.mask_value == MaskValue.DIST:
            # l_x = rearrange(p_x, 'n h w d c -> n (h w d) c')
            for i in range(x.shape[0]):
                samples = rearrange(p_x[i], 'h w d c -> c (h w d)')
                mu = samples.mean(dim=1)
                # force to use higher precision
                with torch.autocast(x.device.type, dtype=torch.float32):
                    cov = samples.cov()
                if torch.is_nonzero(cov.count_nonzero()):
                    dist = torch.distributions.MultivariateNormal(mu, cov)
                    p_x[i][mask[i]] = dist.sample(mask[i].sum().view(-1))
        elif self.args.mask_value == MaskValue.PARAM:
            # only for visualization
            p_x[mask] = 0
        x_mask = unpatchify(p_x, self.patch_size)
        return x_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x_mask = self.apply_mask(x, mask)
        # when using mask_token, original x is input into patch embedding
        if self.args.mask_value != MaskValue.PARAM:
            x = x_mask
        x = self.patch_embed(x)
        if self.args.mask_value == MaskValue.PARAM:
            x = channel_last(x)
            x[mask] = self.mask_token.to(x.dtype)
            x = channel_first(x)
        x = self.pos_drop(x)
        hidden_states = self.forward_layers(x)
        return x_mask, hidden_states

class LayerNorm3d(nn.Sequential):
    def __init__(self, dim: int):
        super().__init__(
            ChannelLast(),
            nn.LayerNorm(dim),
            ChannelFirst(),
        )

@Norm.factory_function("layer3d")
def layer_factory(_dim) -> Type[LayerNorm3d]:
    return LayerNorm3d

class SnimDecoder(nn.Module):
    def __init__(self, args: SnimArgs):
        super().__init__()
        self.args = args
        assert tuple(args.vit_patch_shape) == (2, 2, 2)
        self.lateral_convs = nn.ModuleList([
            ResidualUnit(
                spatial_dims=3,
                in_channels=args.base_feature_size << i,
                out_channels=args.base_feature_size << i,
                kernel_size=1,
                strides=1,
                norm=(Norm.LAYER3D, {'dim': args.base_feature_size << i}),
                act=(Act.PRELU, {'num_parameters': args.base_feature_size << i}),
                subunits=2,
            )
            for i in range(args.vit_stages)
        ])
        self.up_projects = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=args.base_feature_size << i + 1,
                    out_channels=args.base_feature_size << i,
                    kernel_size=2,
                    stride=2,
                ),
                LayerNorm3d(args.base_feature_size << i),
                nn.PReLU(args.base_feature_size << i),
                Convolution(
                    spatial_dims=3,
                    in_channels=args.base_feature_size << i,
                    out_channels=args.base_feature_size << i,
                    kernel_size=1,
                    strides=1,
                    norm=(Norm.LAYER3D, {'dim': args.base_feature_size << i}),
                    act=(Act.PRELU, {'num_parameters': args.base_feature_size << i}),
                ),
            )
            for i in range(args.vit_stages - 1)
        ])
        self.convs = nn.ModuleList([
            ResidualUnit(
                spatial_dims=3,
                in_channels=args.base_feature_size << i,
                out_channels=args.base_feature_size << i,
                kernel_size=1,
                strides=1,
                norm=(Norm.LAYER3D, {'dim': args.base_feature_size << i}),
                act=(Act.PRELU, {'num_parameters': args.base_feature_size << i}),
                subunits=2,
            )
            for i in range(args.vit_stages - 1)
        ])
        self.pred = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=args.base_feature_size,
                out_channels=args.base_feature_size,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm3d(args.base_feature_size),
            nn.PReLU(args.base_feature_size),
            Convolution(
                spatial_dims=3,
                in_channels=args.base_feature_size,
                out_channels=args.num_input_channels,
                kernel_size=1,
                strides=1,
                conv_only=True,
            )
        )

    def forward(self, hidden_states: list[torch.Tensor]):
        x = self.lateral_convs[-1](hidden_states[-1])
        for z, lateral_conv, up_proj, conv in zip(
            hidden_states[-2::-1],
            self.lateral_convs[-2::-1],
            self.up_projects[::-1],
            self.convs[::-1],
        ):
            z = lateral_conv(z)
            x = up_proj(x)
            x = x + z
            x = conv(x)
        pred = self.pred(x)
        return pred

class SnimModel(pl.LightningModule):
    logger: WandbLogger

    def __init__(self, args: SnimArgs):
        super().__init__()
        self.args = args

        self.corner_counter = nn.Conv3d(1, 1, kernel_size=args.p_block_shape, bias=False)
        self.corner_counter.weight.requires_grad = False

        self.encoder = SnimEncoder(args)
        self.decoder = SnimDecoder(args)
        self.loss_fn = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
        }[args.loss]

        self.initialize_weights()

    def initialize_weights(self):
        if self.args.mask_value == MaskValue.PARAM:
            torch.nn.init.normal_(self.encoder.mask_token, std=0.02)
        # initialize nn.Linear, nn.LayerNorm nn.Conv3d with kernel_size=1
        self.apply(self._init_weights)
        nn.init.constant_(self.corner_counter.weight, 1)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)) and all(k <= s for k, s in zip(m.kernel_size, m.stride)):
            w: torch.Tensor = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def gen_patch_mask(self, batch_size: int, img_shape: Sequence[int]) -> torch.Tensor:
        mask_shape = [
            size // patch_size
            for size, patch_size in zip(img_shape, self.args.vit_patch_shape)
        ]
        # corner spatial shape
        corner_ss = [
            size + block_patch_num - 1
            for size, block_patch_num in zip(mask_shape, self.args.p_block_shape)
        ]
        sample_space_size = np.product(corner_ss)
        if self.args.mask_ratio == 1:
            mask_num = np.product(mask_shape)
        else:
            sample_num = np.round(
                np.log(1 - self.args.mask_ratio) /
                np.log(1 - np.product(self.args.p_block_shape) / sample_space_size)
            )
            mask_num = int(sample_space_size * (1 - (1 - 1 / sample_space_size) ** sample_num))
        if mask_num == 0:
            mask = torch.zeros(batch_size, *mask_shape, dtype=torch.bool)
        else:
            noise: torch.Tensor = torch.rand(batch_size, sample_space_size, device=self.device)
            kth = noise.kthvalue(mask_num, dim=-1, keepdim=True).values
            corner_mask = rearrange(
                noise <= kth,
                'n (h w d) -> n 1 h w d',
                h=corner_ss[0], w=corner_ss[1], d=corner_ss[2],
            )
            mask = self.corner_counter(corner_mask.float()).round() >= 1
            mask = rearrange(mask, 'n 1 h w d -> n h w d')

        return mask

    def forward(self, x: torch.Tensor):
        # p_xxx: patchified xxx
        p_x = patchify(x, self.args.vit_patch_shape)
        mask = self.gen_patch_mask(x.shape[0], x.shape[2:])
        x_mask, hidden_states = self.encoder.forward(x, mask)
        pred = self.decoder.forward(hidden_states)
        p_pred = patchify(pred, self.args.vit_patch_shape)
        if self.args.norm_pix_loss:
            # note: the mean and var are calculated across channels
            mean = p_x.mean(dim=-1, keepdim=True)
            var = p_x.var(dim=-1, keepdim=True)
            p_x = (p_x - mean) / torch.sqrt(var + 1e-6)
        mask_loss = self.loss_fn(p_pred[mask], p_x[mask])
        visible_loss = self.loss_fn(p_pred[~mask], p_x[~mask])
        loss = mask_loss + self.args.visible_factor * visible_loss
        return mask_loss, visible_loss, loss, mask, x_mask, pred

    def training_step(self, x: MetaTensor, *args, **kwargs):
        mask_loss, visible_loss, loss, _, _, _ = self.forward(x.as_tensor())
        self.log('train/loss', loss)
        self.log('train/loss(masked)', mask_loss)
        self.log('train/loss(visible)', visible_loss)
        return loss

    def validation_step(self, x: MetaTensor, *args, **kwargs):
        filename = Path(x.meta[ImageMetaKey.FILENAME_OR_OBJ][0]).name
        x = x.as_tensor()
        mask_loss, non_mask_loss, loss, mask, x_mask, pred = self.forward(x)
        self.log('val/loss', loss, sync_dist=True)
        self.log('val/loss(masked)', mask_loss, sync_dist=True)
        self.log('val/loss(visible)', non_mask_loss, sync_dist=True)

        # for better visualization
        x_mask.clamp_(min=0, max=1)
        pred.clamp_(min=0, max=1)
        pred_ol = patchify(pred.clone(), self.args.vit_patch_shape)
        pred_ol[~mask] = patchify(x, self.args.vit_patch_shape)[~mask].to(pred_ol.dtype)
        pred_ol = unpatchify(pred_ol, self.args.vit_patch_shape)

        slice_ids = [int(x.shape[-1] * r) for r in [0.25, 0.5, 0.75]]
        filenames = [None] * self.trainer.world_size
        dist.all_gather_object(filenames, filename)
        all_x, all_x_mask, all_pred, all_pred_ol = self.all_gather([x, x_mask, pred, pred_ol])
        if self.trainer.is_global_zero:
            for filename, x, x_mask, pred, pred_ol in zip(filenames, all_x, all_x_mask, all_pred, all_pred_ol):
                for slice_idx in slice_ids:
                    self.logger.log_image(
                        f'val/{filename}/{slice_idx}',
                        images=[
                            x[0, ..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                            x_mask[0, ..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                            pred[0, ..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                            pred_ol[0, ..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                        ],
                        caption=['original', 'mask', 'pred', 'pred-ol'],
                    )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.args.max_steps),
                'interval': 'step',
            }
        }

    def optimizer_zero_grad(self, _epoch, _batch_idx, optimizer: Optimizer, _optimizer_idx):
        optimizer.zero_grad(set_to_none=self.args.optimizer_set_to_none)

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, **kwargs):
    #     self.should_skip_lr_scheduler_step = False
    #     scaler = getattr(self.trainer.strategy.precision_plugin, "scaler", None)
    #     if scaler:
    #         scale_before_step = scaler.get_scale()
    #     optimizer.step(closure=optimizer_closure)
    #     if scaler:
    #         scale_after_step = scaler.get_scale()
    #         self.should_skip_lr_scheduler_step = scale_before_step > scale_after_step
    #
    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if self.should_skip_lr_scheduler_step:
    #         return
    #     scheduler.step()
