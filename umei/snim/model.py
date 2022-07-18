from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.data import MetaTensor
from monai.utils import ImageMetaKey
from umei.models.swin import SwinTransformer
from umei.utils import MyWandbLogger

from .args import MaskValue, SnimArgs
from .utils import channel_first, channel_last, patchify, unpatchify

class SnimEncoder(SwinTransformer):
    def __init__(
        self,
        mask_value: MaskValue,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_value = mask_value

        if mask_value == MaskValue.PARAM:
            self.mask_token = nn.Parameter(torch.empty(1, 1, self.patch_embed.embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        if self.mask_value == MaskValue.UNIFORM:
            p_x = patchify(x.clone(), self.patch_size)
            p_x[mask] = torch.rand(mask.sum(), p_x.shape[-1])
            x = unpatchify(p_x)
        elif self.mask_value == MaskValue.DIST:
            raise NotImplementedError
        x = self.patch_embed(x)
        if self.mask_value == MaskValue.PARAM:
            x[mask] = self.mask_token.to(x.dtype)
        x = channel_first(x)
        x = self.pos_drop(x)
        hidden_states = self.forward_layers(x)
        return hidden_states

class SnimModel(pl.LightningModule):
    logger: MyWandbLogger

    def __init__(self, args: SnimArgs):
        super().__init__()
        self.args = args

        self.corner_counter = nn.Conv3d(1, 1, kernel_size=args.p_block_shape, bias=False)
        nn.init.constant_(self.corner_counter.weight, 1)
        self.corner_counter.weight.requires_grad = False

        self.encoder = SnimEncoder(
            in_chans=args.num_input_channels,
            embed_dim=args.base_feature_size,
            window_size=args.swin_window_size,
            patch_size=args.vit_patch_shape,
            depths=args.vit_depths,
            num_heads=args.vit_num_heads,
            use_checkpoint=True,
            mask_value=args.mask_value,
        )

        self.pred = nn.Conv3d(
            (args.base_feature_size << len(args.vit_depths)) - args.base_feature_size,
            args.num_input_channels * np.product(args.vit_patch_shape),
            kernel_size=1,
        )
        self.loss_fn = nn.MSELoss()

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed's projection like nn.Linear (instead of nn.Convnd)
        w: torch.Tensor = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # so does the linear reconstruction head
        w: torch.Tensor = self.pred.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.args.mask_value == MaskValue.PARAM:
            torch.nn.init.normal_(self.encoder.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def gen_patch_mask(self, batch_size: int, img_shape: Sequence[int]) -> torch.Tensor:
        mask_shape = [
            size // patch_size
            for size, patch_size in zip(img_shape, self.args.vit_patch_shape)
        ]
        # corner spatial shape
        corner_ss = [
            size + block_patch_num - 1
            for size, block_patch_num in zip(mask_shape, self.args.mask_block_shape)
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
        mask = self.gen_patch_mask(x.shape[0], x.shape[2:])
        hidden_states = self.encoder.forward(x, mask)
        context = torch.cat([self.repeat_hidden_state(z, 1 << i) for i, z in enumerate(hidden_states)], dim=1)
        p_pred = self.pred(context)
        p_pred = channel_last(p_pred)
        pred = unpatchify(p_pred, self.args.vit_patch_shape)
        p_x = patchify(x, self.args.vit_patch_shape)
        if self.args.norm_pix_loss:
            # note: the mean and var are calculated across channels
            mean = p_x.mean(dim=-1, keepdim=True)
            var = p_x.var(dim=-1, keepdim=True)
            p_x = (p_x - mean) / torch.sqrt(var + 1e-6)
        mask_loss = self.loss_fn(p_pred[mask], p_x[mask])
        non_mask_loss = self.loss_fn(p_pred[~mask], p_x[~mask])
        loss = mask_loss + self.args.non_mask_factor * non_mask_loss
        return mask_loss, non_mask_loss, loss, mask, pred

    def training_step(self, x: torch.Tensor, *args, **kwargs):
        mask_loss, non_mask_loss, loss, _, _ = self.forward(x)
        self.log('train/loss', loss)
        self.log('train/mask-loss', mask_loss)
        self.log('train/non-mask-loss', non_mask_loss)
        return loss

    def validation_step(self, x: MetaTensor, *args, **kwargs):
        mask_loss, non_mask_loss, loss, mask, pred = self.forward(x)
        self.log('val/loss', loss)
        self.log('val/loss(mask)', mask_loss)
        self.log('val/loss(non-mask)', non_mask_loss)

        # for better visualization
        pred.clamp_(min=0, max=1)

        x_mask = self.patchify(x.clone())
        x_mask[mask] = 0
        x_mask = self.unpatchify(x_mask)

        pred_ol = self.patchify(pred.clone())
        pred_ol[~mask] = self.patchify(x)[~mask].to(pred_ol.dtype)
        pred_ol = self.unpatchify(pred_ol)

        slice_idx = x.shape[-1] // 2
        filename = Path(x.meta[ImageMetaKey.FILENAME_OR_OBJ][0]).name
        self.logger.log_image(
            f'val/{filename}',
            images=[
                x[0, ..., slice_idx].rot90(dims=(1, 2)).cpu(),
                x_mask[0, ..., slice_idx].rot90(dims=(1, 2)).cpu(),
                pred[0, ..., slice_idx].rot90(dims=(1, 2)).cpu(),
                pred_ol[0, ..., slice_idx].rot90(dims=(1, 2)).cpu(),
            ],
            caption=['original', f'mask', 'pred', 'pred-ol'],
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': CosineAnnealingLR(optimizer, T_max=int(self.args.num_train_epochs)),
        }
