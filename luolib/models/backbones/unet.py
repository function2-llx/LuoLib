from collections.abc import Sequence

import torch
from torch import nn

from luolib.types import spatial_param_t
from ..blocks import BasicConvLayer
from ..layers import Act, Norm

__all__ = [
    'UNetBackbone',
]

class UNetBackbone(nn.Module):
    """if res_block, this is a ResNet backbone with conv stem and basic block, right?"""
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        layer_channels: list[int],
        kernel_sizes: Sequence[spatial_param_t[int]],
        strides: Sequence[spatial_param_t[int]],
        num_blocks: list[int] | None,
        norm: str | tuple = (Norm.INSTANCE, {'affine': True}),
        act: str | tuple = Act.LEAKYRELU,
        res_block: bool = False,
    ):
        super().__init__()
        num_layers = len(layer_channels)
        if num_blocks is None:
            num_blocks = [1] * num_layers
        self.layers = nn.ModuleList([
            BasicConvLayer(
                spatial_dims,
                num_blocks[i],
                layer_channels[i - 1] if i else in_channels,
                layer_channels[i],
                kernel_sizes[i], strides[i],
                norm, act,
                res_block and i > 0,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps
