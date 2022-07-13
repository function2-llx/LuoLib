from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from einops import rearrange
import numpy as np
import torch
from torch import nn

from .swin import SwinTransformer

__all__ = ['MaskSwin']

@dataclass
class MaskSwinOutput:
    mask: torch.Tensor = None
    hidden_states: list[torch.Tensor] = None

class MaskSwin(SwinTransformer):
    def __init__(
        self,
        mask_ratio: float,
        block_shape: Sequence[int],
        *,
        base_feature_size: int,
        patch_size: Sequence[int] | int,
        **kwargs,
    ):
        super().__init__(**kwargs, embed_dim=base_feature_size, patch_size=patch_size)
        self.mask_ratio = mask_ratio

        # how many patches does a block have (along each axis)
        self.block_patch_shape = tuple(
            block_size // patch_size
            for block_size, patch_size in zip(block_shape, patch_size)
        )

        self.mask_token = nn.Parameter(torch.empty(1, 1, base_feature_size))
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
        rearrange(x_mask, 'n c h w d -> n h w d c')[mask] = self.mask_token.to(x_mask.dtype)
        return x_mask, mask

    def test_mask_ratio(self, x):
        x = self.patch_embed(x)
        x_mask, mask = self.random_masking(x)
        mask = mask.view(x.shape[0], -1)
        return mask.sum(dim=-1) / mask.shape[1]

    def forward(self, x: torch.Tensor) -> MaskSwinOutput:
        x = self.patch_embed(x)
        x, mask = self.random_masking(x)
        x = self.pos_drop(x)
        hidden_states = self.forward_layers(x)
        return MaskSwinOutput(
            mask=mask,
            hidden_states=hidden_states,
        )
