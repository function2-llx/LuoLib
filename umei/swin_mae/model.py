from __future__ import annotations

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from umei.utils import MyWandbLogger
from .args import SwinMAEArgs
from .mask_swin import MaskSwin

class SwinMAEDecoder:
    def __init__(
        self,
        feature_size: int = 24,
        num_layers: int = 4,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
        use_skip: bool = True,
    ) -> None:
        super().__init__()

        self.bottleneck = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size << num_layers - 1,
            out_channels=feature_size << num_layers - 1,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.ups = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size << i,
                out_channels=feature_size << i - 1,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
                use_skip=use_skip,
            )
            for i in range(1, num_layers)
        ])

    def forward(self, hidden_states: list[torch.Tensor]):
        x = self.bottleneck(hidden_states[-1])
        for z, up in zip(hidden_states[-2:-1], self.ups[::-1]):
            x = up(x, z)
        return x

class SwinMAE(pl.LightningModule):
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
        self.decoder = SwinMAEDecoder(feature_size=args.base_feature_size, use_skip=args.use_skip)

        self.pred = nn.Conv3d(
            args.base_feature_size,
            args.num_input_channels * np.product(args.vit_patch_shape),
            kernel_size=1,
        )
        self.loss_fn = nn.MSELoss()

        # initialize weights
        # initialize patch_embed like nn.Linear (instead of nn.Convnd)
        w: torch.Tensor = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def forward(self, x: torch.Tensor):
        encoder_out = self.encoder.forward(x)
        mask = encoder_out.mask

        decoder_out = self.decoder.forward(encoder_out.hidden_states)
        p_pred = self.pred(decoder_out.feature_maps[-1])
        p_x = self.patchify(x)
        if self.args.norm_pix_loss:
            # note: the mean and var are calculated across channels
            mean = p_x.mean(dim=-1, keepdim=True)
            var = p_x.var(dim=-1, keepdim=True)
            p_x = (p_x - mean) / torch.sqrt(var + 1e-6)
        loss = self.loss_fn(p_pred[mask], p_x[mask])
        return loss

    def training_step(self, x: torch.Tensor, *args, **kwargs):
        # p_x = self.patchify(x)
        # encoder_out = self.encoder.forward(x)
        # mask = encoder_out.mask
        #
        # decoder_out = self.decoder.forward(encoder_out.hidden_states)
        # p_pred = self.pred(decoder_out.feature_maps[-1])
        # if self.args.norm_pix_loss:
        #     mean = p_x.mean(dim=-1, keepdim=True)
        #     var = p_x.var(dim=-1, keepdim=True)
        #     p_x = (p_x - mean) / torch.sqrt(var + 1e-6)
        # loss = self.loss_fn(p_pred[mask], p_x[mask])
        loss = self.forward(x)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, x: torch.Tensor, batch_idx: int, *args, **kwargs):
        p_x = self.patchify(x)
        encoder_out = self.encoder.forward(x)
        mask = encoder_out.mask
        x_mask = p_x.clone()
        x_mask[mask] = 0
        x_mask = rearrange(
            x_mask,
            'n h w d (ph pw pd) c -> n c (h ph) (w pw) (d pd)',
            ph=self.args.vit_patch_shape[0],
            pw=self.args.vit_patch_shape[1],
            pd=self.args.vit_patch_shape[2],
        )


        decoder_out = self.decoder.forward(encoder_out.hidden_states)
        p_pred = self.pred(decoder_out.feature_maps[-1])
        if self.args.norm_pix_loss:
            mean = p_x.mean(dim=-1, keepdim=True)
            var = p_x.var(dim=-1, keepdim=True)
            p_x = (p_x - mean) / torch.sqrt(var + 1e-6)
        loss = self.loss_fn(p_pred[mask], p_x[mask])
        self.log('val/loss', loss)

        slice_idx = x.shape[-1] // 2
        self.logger.log_image(
            f'val/reconstruction-{batch_idx}',
            images=[x[0, ..., slice_idx].cpu(), x_mask[0, ..., slice_idx].cpu(), pred[0, ..., slice_idx].cpu()],
            caption=['gt', 'mask', 'pred'],
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': CosineAnnealingLR(optimizer, T_max=int(self.args.num_train_epochs)),
        }
