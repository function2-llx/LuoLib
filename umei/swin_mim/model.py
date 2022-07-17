from __future__ import annotations

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
from umei.utils import MyWandbLogger


from .args import SwinMAEArgs
from .mask_swin import MaskSwin
from .utils import channel_last

class SwinMIM(pl.LightningModule):
    logger: MyWandbLogger

    def __init__(self, args: SwinMAEArgs):
        super().__init__()
        self.args = args

        self.encoder = MaskSwin(
            mask_ratio=args.mask_ratio,
            block_shape=args.mask_block_shape,
            in_chans=args.num_input_channels,
            base_feature_size=args.base_feature_size,
            window_size=args.swin_window_size,
            patch_size=args.vit_patch_shape,
            depths=args.vit_depths,
            num_heads=args.vit_num_heads,
            use_checkpoint=True,
        )

        self.pred = nn.Conv3d(
            (args.base_feature_size << len(args.vit_depths)) - args.base_feature_size,
            args.num_input_channels * np.product(args.vit_patch_shape),
            kernel_size=1,
        )
        self.loss_fn = nn.MSELoss()

        # initialize weights
        # initialize patch_embed like nn.Linear (instead of nn.Convnd)
        w: torch.Tensor = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # so does the linear reconstruction head
        w: torch.Tensor = self.pred.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def patchify(self, x):
        return rearrange(
            x,
            'n c (h ph) (w pw) (d pd) -> n h w d (ph pw pd c)',
            ph=self.args.vit_patch_shape[0],
            pw=self.args.vit_patch_shape[1],
            pd=self.args.vit_patch_shape[2],
        )

    def unpatchify(self, x):
        return rearrange(
            x,
            'n h w d (ph pw pd c) -> n c (h ph) (w pw) (d pd)',
            ph=self.args.vit_patch_shape[0],
            pw=self.args.vit_patch_shape[1],
            pd=self.args.vit_patch_shape[2],
        )

    def forward(self, x: torch.Tensor):
        # p_xxx: patchified xxx
        encode = self.encoder.forward(x)
        mask = encode.mask
        context = torch.cat([
            z.repeat_interleave(1 << i, dim=2).repeat_interleave(1 << i, dim=3).repeat_interleave(1 << i, dim=4)
            for i, z in enumerate(encode.hidden_states)
        ], dim=1)
        p_pred = self.pred(context)
        p_pred = channel_last(p_pred)
        pred = self.unpatchify(p_pred)
        p_x = self.patchify(x)
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
        self.log('val/mask-loss', mask_loss)
        self.log('val/non-mask-loss', non_mask_loss)

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
