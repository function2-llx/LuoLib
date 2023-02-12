from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.umei import Decoder, DecoderOutput

from umei.models.adaptive_resampling import AdaptiveUpsampling
from umei.models.blocks import ResLayer, get_conv_layer
from umei.models.layers import Act, Norm

class PlainConvUNetDecoder(Decoder):
    upsamplings: Sequence[AdaptiveUpsampling] | nn.ModuleList

    def __init__(
        self,
        layer_channels: list[int],
        dropout: tuple | str | float | None = None,
        norm: tuple | str | None = Norm.INSTANCE,
        act: tuple | str | None = Act.LEAKYRELU,
        num_post_upsampling_layers: int  = 0,
        post_upsampling_channels: int | None = None,
    ) -> None:
        super().__init__()
        num_layers = len(layer_channels) - 1
        self.layers = nn.ModuleList([
            ResLayer(
                _num_blocks := 1,
                _in_channels := layer_channels[i] * 2,
                _out_channels := layer_channels[i],
                _kernel_size := 3,
                dropout,
                norm,
                act,
            )
            for i in range(num_layers)
        ])
        self.lateral_convs = nn.ModuleList([
            ResLayer(
                _num_blocks := 1,
                _in_channels := layer_channels[i],
                _out_channels := layer_channels[i],
                _kernel_size := 3,
                dropout,
                norm,
                act,
            )
            for i in range(num_layers)
        ])
        self.bottleneck = ResLayer(
            _num_blocks := 1,
            _in_channels := layer_channels[-1],
            _out_channels := layer_channels[-1],
            _kernel_size := 3,
            dropout,
            norm,
            act,
        )
        self.upsamplings = nn.ModuleList([
            AdaptiveUpsampling(layer_channels[i + 1], layer_channels[i])
            for i in range(num_layers)
        ])
        if post_upsampling_channels is None:
            post_upsampling_channels = layer_channels[0] >> 1

        self.post_upsamplings = nn.ModuleList([
            AdaptiveUpsampling(
                post_upsampling_channels if i else layer_channels[0],
                post_upsampling_channels,
                kernel_size=2,
            )
            for i in range(num_post_upsampling_layers)
        ])
        self.post_convs = nn.ModuleList([
            get_conv_layer(post_upsampling_channels, post_upsampling_channels)
            for _ in range(num_post_upsampling_layers)
        ])

    def forward(self, backbone_feature_maps: Sequence[torch.Tensor], x_in: torch.Tensor) -> DecoderOutput:
        feature_maps = []
        x = backbone_feature_maps[-1]
        x = self.bottleneck(x)
        for lateral_conv, upsampling, layer, skip in zip(self.lateral_convs[::-1], self.upsamplings[::-1], self.layers[::-1], backbone_feature_maps[-2::-1]):
            x = upsampling.forward(x, x.shape[-1] < skip.shape[-1])
            skip = lateral_conv(skip)
            x = layer(torch.cat([x, skip], dim=1))
            feature_maps.append(x)

        for upsampling, conv in zip(self.post_upsamplings, self.post_convs):
            x = upsampling(x, x.shape[-1] != x_in.shape[-1])
            x = conv(x)
            feature_maps.append(x)

        return DecoderOutput(feature_maps)
