from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.umei import Decoder, DecoderOutput

from umei.models.adaptive_resampling import AdaptiveUpsampling
from umei.models.blocks import ResLayer, get_conv_layer
from umei.models.init import init_linear_conv3d
from umei.models.layers import Act, Norm

class PlainConvUNetDecoder(Decoder):
    upsamplings: Sequence[AdaptiveUpsampling] | nn.ModuleList

    def __init__(
        self,
        layer_channels: list[int],
        dropout: tuple | str | float | None = None,
        norm: tuple | str | None = Norm.INSTANCE,
        act: tuple | str | None = Act.LEAKYRELU,
        num_post_upsamplings: int = 0,
        post_upsampling_channels: int | None = None,
        bottleneck: bool = False,
        lateral: bool = True,
    ) -> None:
        super().__init__()
        num_layers = len(layer_channels) - 1
        if bottleneck:
            self.bottleneck = ResLayer(
                _num_blocks := 1,
                layer_channels[-1],
                layer_channels[-1],
                _kernel_size := 3,
                dropout,
                norm,
                act,
            )
        else:
            self.bottleneck = nn.Identity()
        self.layers = nn.ModuleList([
            nn.Sequential(
                get_conv_layer(layer_channels[i] * 2, layer_channels[i], 3, dropout=dropout, norm=norm, act=act),
                get_conv_layer(layer_channels[i], layer_channels[i], 3, dropout=dropout, norm=norm, act=act),
            )
            for i in range(num_layers)
        ])
        self.laterals = nn.ModuleList([
            get_conv_layer(layer_channels[i], layer_channels[i], 1, dropout=dropout, norm=norm, act=act) if lateral
            else nn.Identity()
            for i in range(num_layers)
        ])
        self.upsamplings = nn.ModuleList([
            AdaptiveUpsampling(layer_channels[i + 1], layer_channels[i])
            for i in range(num_layers)
        ])

        if post_upsampling_channels is None:
            post_upsampling_channels = layer_channels[0]
        self.post_upsamplings = nn.ModuleList([
            nn.Sequential(
                AdaptiveUpsampling(
                    layer_channels[0] if i == 0 else post_upsampling_channels,
                    post_upsampling_channels,
                    kernel_size=3,
                ),
                get_conv_layer(post_upsampling_channels, post_upsampling_channels, 3, dropout=dropout, norm=norm, act=act),
                get_conv_layer(post_upsampling_channels, post_upsampling_channels, 3, dropout=dropout, norm=norm, act=act),
            )
            for i in range(num_post_upsamplings)
        ])

        self.apply(init_linear_conv3d)

    def forward(self, backbone_feature_maps: Sequence[torch.Tensor], x_in: torch.Tensor) -> DecoderOutput:
        feature_maps = []
        x = backbone_feature_maps[-1]
        x = self.bottleneck(x)
        for lateral, upsampling, layer, skip in zip(self.laterals[::-1], self.upsamplings[::-1], self.layers[::-1], backbone_feature_maps[-2::-1]):
            x = upsampling.forward((x, x.shape[-1] < skip.shape[-1]))
            skip = lateral(skip)
            x = layer(torch.cat([x, skip], dim=1))
            feature_maps.append(x)

        for post_upsampling in self.post_upsamplings:
            x = post_upsampling((x, x.shape[-1] != x_in.shape[-1]))
            feature_maps.append(x)

        return DecoderOutput(feature_maps)
