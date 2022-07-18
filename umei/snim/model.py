from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.data import MetaTensor
from monai.utils import ImageMetaKey
from umei.models.swin import SwinTransformer
from .args import MaskValue, SnimArgs
from .utils import channel_first, channel_last, patchify, unpatchify

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
                with torch.autocast(x.device.type, dtype=torch.float32):
                    cov = samples.cov()
                dist = torch.distributions.MultivariateNormal(mu, cov)
                p_x[i][mask[i]] = dist.sample(mask[i].sum().view(-1))
        elif self.args.mask_value == MaskValue.PARAM:
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

class SnimUpBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        out_channels = in_channels >> 1
        self.up_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()
        self.linear = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        x = self.up_conv(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.linear(x)
        x = self.norm2(x)
        return x + z

class SnimDecoder(nn.Module):
    def __init__(self, args: SnimArgs):
        super().__init__()
        self.args = args
        bottleneck_dim = args.base_feature_size << len(args.vit_depths) - 1
        self.bottleneck = nn.Sequential(
            nn.Conv3d(bottleneck_dim, bottleneck_dim, kernel_size=1),
            nn.InstanceNorm3d(bottleneck_dim),
            nn.GELU(),
            nn.Conv3d(bottleneck_dim, bottleneck_dim, kernel_size=1),
            nn.InstanceNorm3d(bottleneck_dim),
        )
        self.ups = nn.ModuleList([
            SnimUpBlock(args.base_feature_size << i)
            for i in range(1, len(args.vit_depths))
        ])
        self.pred = nn.ConvTranspose3d(
            args.base_feature_size,
            args.num_input_channels,
            kernel_size=args.vit_patch_shape,   # type: ignore
            stride=args.vit_patch_shape,    # type: ignore
        )

    def forward(self, hidden_states: list[torch.Tensor]):
        x = self.bottleneck(hidden_states[-1])
        for z, up in zip(hidden_states[-2::-1], self.ups[::-1]):
            x = up(x, z)
        x = self.pred(x)
        return x

class SnimModel(pl.LightningModule):
    logger: WandbLogger

    def __init__(self, args: SnimArgs):
        super().__init__()
        self.args = args

        self.corner_counter = nn.Conv3d(1, 1, kernel_size=args.p_block_shape, bias=False)
        nn.init.constant_(self.corner_counter.weight, 1)
        self.corner_counter.weight.requires_grad = False

        self.encoder = SnimEncoder(args)
        self.decoder = SnimDecoder(args)
        self.loss_fn = nn.MSELoss()
        self.initialize_weights()

    def initialize_weights(self):
        if self.args.mask_value == MaskValue.PARAM:
            torch.nn.init.normal_(self.encoder.mask_token, std=0.02)
        # initialize nn.Linear, nn.LayerNorm nn.Conv3d with kernel_size=1
        self.apply(self._init_weights)

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
        if self.args.mask_ratio == 1:
            mask_num = np.product(mask_shape)
        else:
            mask_num: int = np.round(
                np.log(1 - self.args.mask_ratio) /
                np.log(1 - np.product(self.args.p_block_shape) / np.product(corner_ss))
            ).astype(int)
        if mask_num == 0:
            mask = torch.zeros(batch_size, *mask_shape, dtype=torch.bool)
        else:
            noise: torch.Tensor = torch.rand(batch_size, np.product(corner_ss), device=self.device)
            kth = noise.kthvalue(mask_num, dim=-1, keepdim=True).values
            corner_mask = rearrange(
                noise <= kth,
                'n (h w d) -> n 1 h w d',
                h=corner_ss[0], w=corner_ss[1], d=corner_ss[2],
            )
            mask = self.corner_counter(corner_mask.float()).round() >= 1
            mask = rearrange(mask, 'n 1 h w d -> n h w d')

        return mask

    @staticmethod
    def repeat_hidden_state(z: torch.Tensor, k: int):
        return z.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3).repeat_interleave(k, dim=4)

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
        non_mask_loss = self.loss_fn(p_pred[~mask], p_x[~mask])
        loss = mask_loss + self.args.non_mask_factor * non_mask_loss
        return mask_loss, non_mask_loss, loss, mask, x_mask, pred

    def training_step(self, x: MetaTensor, *args, **kwargs):
        mask_loss, non_mask_loss, loss, _, _, _ = self.forward(x.as_tensor())
        self.log('train/loss', loss)
        self.log('train/mask-loss', mask_loss)
        self.log('train/non-mask-loss', non_mask_loss)
        return loss

    def validation_step(self, x: MetaTensor, *args, **kwargs):
        filename = Path(x.meta[ImageMetaKey.FILENAME_OR_OBJ][0]).name
        x = x.as_tensor()
        mask_loss, non_mask_loss, loss, mask, x_mask, pred = self.forward(x)
        self.log('val/loss', loss)
        self.log('val/loss(mask)', mask_loss)
        self.log('val/loss(non-mask)', non_mask_loss)

        # for better visualization
        x_mask.clamp_(min=0, max=1)
        pred.clamp_(min=0, max=1)
        pred_ol = patchify(pred.clone(), self.args.vit_patch_shape)
        pred_ol[~mask] = patchify(x, self.args.vit_patch_shape)[~mask].to(pred_ol.dtype)
        pred_ol = unpatchify(pred_ol, self.args.vit_patch_shape)

        slice_idx = x.shape[-1] // 2
        self.logger.log_image(
            f'val/{filename}',
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
            'lr_scheduler': CosineAnnealingLR(optimizer, T_max=self.args.max_steps),
        }
