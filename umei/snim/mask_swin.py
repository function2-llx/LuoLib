from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from einops import rearrange
import numpy as np
import torch
from torch import nn

from umei.models.swin import SwinTransformer

__all__ = ['SnimEncoder']

from .utils import channel_first, channel_last

@dataclass
class MaskSwinOutput:
    mask: torch.Tensor = None
    hidden_states: list[torch.Tensor] = None

class SnimEncoder(SwinTransformer):
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
