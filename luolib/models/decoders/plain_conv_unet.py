from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from luolib.types import spatial_param_seq_t
from luolib.utils import fall_back_none
from ..layers.factories import default_instance
from ..blocks import UNetUpLayer, get_conv_layer
from ..init import init_common
from ..layers import Act
from .base import NestedBackbone

__all__ = [
    'PlainConvUNetDecoder',
]

class PlainConvUNetDecoder(NestedBackbone):
    def __init__(
        self,
        *,
        spatial_dims: int,
        layer_channels: list[int],
        kernel_sizes: spatial_param_seq_t[int],
        strides: spatial_param_seq_t[int],
        norm: tuple | str | None = None,
        act: tuple | str | None = Act.LEAKYRELU,
        lateral_channels: Sequence[int] | None = None,
        lateral_kernel_sizes: spatial_param_seq_t[int] | None = None,
        **kwargs,
    ):
        """
        Args:
            strides: strides[0] is not used (usually = 1)
            layer_channels: resolution: high → low
        """
        super().__init__(**kwargs)
        norm = fall_back_none(norm, default_instance())
        num_layers = len(layer_channels) - 1
        self.layers = nn.ModuleList([
            UNetUpLayer(spatial_dims, layer_channels[i + 1], layer_channels[i], kernel_sizes[i], strides[i + 1])
            for i in range(num_layers)
        ])
        if lateral_kernel_sizes is None:
            self.laterals = nn.ModuleList([
                nn.Identity()
                for _ in range(num_layers + 1)
            ])
        else:
            if lateral_channels is None:
                lateral_channels = layer_channels
            self.laterals = nn.ModuleList([
                get_conv_layer(spatial_dims, lc, channels, kernel_size, norm=norm, act=act)
                for lc, channels, kernel_size in zip(lateral_channels, layer_channels, lateral_kernel_sizes)
            ])

        self.apply(init_common)

    def process(self, feature_maps: Sequence[torch.Tensor], *args, **kwargs):
        feature_maps = [
            lateral(x)
            for x, lateral in zip(feature_maps, self.laterals)
        ]
        x = feature_maps[-1]
        ret = [x]
        for layer, skip in zip(self.layers[::-1], feature_maps[-2::-1]):
            x = layer(x, skip)
            ret.append(x)
        return ret[::-1]
