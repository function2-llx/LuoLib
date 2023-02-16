from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.layers import get_act_layer, get_norm_layer
from monai.umei import Decoder, DecoderOutput

from umei.models.adaptive_resampling import AdaptiveUpsampling
from umei.models.blocks import ResLayer, get_conv_layer
from umei.models.init import init_linear_conv
from umei.models.layers import Act, Norm

class PlainConvUNetDecoder(Decoder):
    upsamplings: Sequence[AdaptiveUpsampling] | nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        layer_channels: list[int],
        dropout: tuple | str | float | None = None,
        norm: tuple | str | None = Norm.INSTANCE,
        act: tuple | str | None = Act.LEAKYRELU,
        num_post_upsamplings: int = 0,
        post_upsampling_channels: int | None = None,
    ) -> None:
        super().__init__()
        num_layers = len(layer_channels) - 1
        self.bottleneck = nn.Identity()
        self.layers = nn.ModuleList([
            nn.Sequential(
                get_conv_layer(layer_channels[i] * 2, layer_channels[i], 3, dropout=dropout, norm=norm, act=act),
                get_conv_layer(layer_channels[i], layer_channels[i], 3, dropout=dropout, norm=norm, act=act),
            )
            for i in range(num_layers)
        ])
        self.laterals = nn.ModuleList([
            # nn.Identity()
            get_conv_layer(layer_channels[i], layer_channels[i], 1, dropout=dropout, norm=norm, act=act)
            for i in range(num_layers)
        ])
        self.upsamplings = nn.ModuleList([
            AdaptiveUpsampling(layer_channels[i + 1], layer_channels[i])
            # nn.Sequential(
            #     AdaptiveUpsampling(layer_channels[i + 1], layer_channels[i]),
            #     get_norm_layer(norm, 3, layer_channels[i]),
            # )
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
                # this normalization is crucial if a ResLayer with the same in/out channels is followed (no 1-conv for
                # residual so no normalization as well), or the residual term will become too large during training
                # get_norm_layer(norm, 3, post_upsampling_channels),
                # ResLayer(
                #     _num_blocks := 1,
                #     _in_channels := post_upsampling_channels,
                #     _out_channels := post_upsampling_channels,
                #     _kernel_size := 3,
                #     dropout,
                #     norm,
                #     act,
                # )
            )
            for i in range(num_post_upsamplings)
        ])

        self.apply(init_linear_conv)

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
