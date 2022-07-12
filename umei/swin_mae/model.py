from __future__ import annotations

from collections.abc import Sequence

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from monai.networks.nets import SwinTransformer, SwinUnetrDecoder
from umei.model import UEncoderOutput
from umei.swin_mae.args import SwinMAEArgs

class MaskSwin(SwinTransformer):
    def __init__(
        self,
        mask_ratio: float,
        block_shape: Sequence[int],
        *,
        base_feature_size: int,
        patch_shape: Sequence[int] | int,
        **kwargs,
    ):
        super().__init__(patch_size=patch_shape, embed_dim=base_feature_size, **kwargs)
        self.mask_ratio = mask_ratio
        # self.mask_num = mask_num
        self.block_patch_shape = tuple(
            block_size // patch_size
            for block_size, patch_size in zip(block_shape, patch_shape)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, base_feature_size))
        self.corner_counter = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=self.block_patch_shape, bias=False)

        nn.init.constant_(self.corner_counter.weight, 1)
        self.corner_counter.weight.requires_grad = False

    def random_masking(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # corner spatial shape
        corner_ss = [
            size + block_patch_num - 1
            for size, block_patch_num in zip(x.shape[2:], self.block_patch_shape)
        ]
        mask_num: int = np.round(
            np.log(1 - self.mask_ratio) /
            np.log(1 - np.product(self.block_patch_shape) / np.product(corner_ss))
        ).astype(int)
        noise: torch.Tensor = torch.rand(x.shape[0], np.product(corner_ss), device=x.device)
        kth = noise.kthvalue(mask_num, dim=-1, keepdim=True).values
        corner_mask = rearrange(noise <= kth, 'n (h w d) -> n 1 h w d', h=corner_ss[0], w=corner_ss[1], d=corner_ss[2])
        mask = self.corner_counter(corner_mask.float()).round() >= 1
        mask = rearrange(mask, 'n 1 h w d -> n h w d')
        x_mask = x.clone()
        rearrange(x_mask, 'n c h w d -> n h w d c')[mask] = self.mask_token
        print('sample mask ratio:', mask.sum() / mask.numel())
        return x_mask, mask

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0_mask, mask = self.random_masking(x0)
        mask = mask.view(x0.shape[0], -1)
        return mask.sum(dim=-1) / mask.shape[1]
        # x0_mask = self.pos_drop(x0_mask)
        # x0_out = self.proj_out(x0_mask, normalize)
        # x1 = self.layers1[0](x0_mask.contiguous())
        # x1_out = self.proj_out(x1, normalize)
        # x2 = self.layers2[0](x1.contiguous())
        # x2_out = self.proj_out(x2, normalize)
        # x3 = self.layers3[0](x2.contiguous())
        # x3_out = self.proj_out(x3, normalize)
        # x4 = self.layers4[0](x3.contiguous())
        # x4_out = self.proj_out(x4, normalize)
        # return UEncoderOutput(self.avg_pool(x4_out).view(x4_out.shape[:2]), [x0_out, x1_out, x2_out, x3_out, x4_out])

class SwinMAE(pl.LightningModule):
    def __init__(self, args: SwinMAEArgs):
        super().__init__()
        self.args = args

        self.encoder = MaskSwin(
            mask_ratio=args.mask_num,
            block_shape=args.mask_block_shape,
            in_chans=args.num_input_channels,
            base_feature_size=args.base_feature_size,
            window_size=args.swin_window_size,
            patch_shape=args.vit_patch_shape,
            depths=args.vit_depths,
            num_heads=args.vit_num_heads,
            # mlp_ratio=4.0,
            # qkv_bias=True,
            use_checkpoint=True,
        )

        self.decoder = SwinUnetrDecoder(
            in_channels=args.num_input_channels,
            feature_size=args.base_feature_size,
            use_encoder5=self.args.use_encoder5,
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, args.num_input_channels, *args.sample_shape)
            dummy_encoder_output = self.encoder.forward(dummy_input)
            dummy_decoder_output = self.decoder.forward(dummy_input, dummy_encoder_output.hidden_states)
            decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Convnd)
        w: torch.Tensor = self.encoder.patch_embed.weight.data
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

    def training_step(self, x: torch.Tensor, *args, **kwargs):
        x_mask, mask = self.random_masking(x)
