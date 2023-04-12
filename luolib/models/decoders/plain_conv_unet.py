from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.umei import Decoder, DecoderOutput
from umei.models import decoder_registry
from umei.models.blocks import UNetUpLayer, get_conv_layer
from umei.models.init import init_linear_conv
from umei.models.layers import Act, Norm
from umei.types import spatial_param_seq_t

__all__ = []

@decoder_registry.register_module('unet')
class PlainConvUNetDecoder(Decoder):
    def __init__(
        self,
        spatial_dims: int,
        layer_channels: list[int],
        kernel_sizes: spatial_param_seq_t[int],
        norm: tuple | str | None = Norm.INSTANCE,
        act: tuple | str | None = Act.LEAKYRELU,
        lateral_kernel_sizes: spatial_param_seq_t[int] | None = None,
    ) -> None:
        super().__init__()
        num_layers = len(layer_channels) - 1
        self.layers = nn.ModuleList([
            UNetUpLayer(spatial_dims, layer_channels[i + 1], layer_channels[i], kernel_sizes[i])
            for i in range(num_layers)
        ])
        if lateral_kernel_sizes is None:
            self.laterals = nn.ModuleList([
                nn.Identity()
                for _ in range(num_layers + 1)
            ])
        else:
            self.laterals = nn.ModuleList([
                get_conv_layer(spatial_dims, channels, channels, kernel_size, norm=norm, act=act)
                for channels, kernel_size in zip(layer_channels, lateral_kernel_sizes)
            ])

        self.apply(init_linear_conv)

    def forward(self, backbone_feature_maps: Sequence[torch.Tensor], x_in: torch.Tensor) -> DecoderOutput:
        feature_maps = []
        backbone_feature_maps = [
            lateral(x)
            for x, lateral in zip(backbone_feature_maps, self.laterals)
        ]
        x = backbone_feature_maps[-1]
        for layer, skip in zip(self.layers[::-1], backbone_feature_maps[-2::-1]):
            x = layer(x, skip)
            feature_maps.append(x)
        return DecoderOutput(feature_maps)
