from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.umei import Decoder, DecoderOutput

from umei.models.adaptive_resampling import AdaptiveUpsampling
from umei.models.blocks import ResLayer
from umei.models.layers import Act, Norm

class PlainConvUNetDecoder(Decoder):
    upsamplings: Sequence[AdaptiveUpsampling] | nn.ModuleList

    def __init__(
        self,
        layer_channels: list[int],
        dropout: tuple | str | float | None = None,
        norm: tuple | str | None = Norm.INSTANCE,
        act: tuple | str | None = Act.LEAKYRELU,
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
        self.upsamplings = nn.ModuleList([
            AdaptiveUpsampling(layer_channels[i + 1], layer_channels[i])
            for i in range(num_layers)
        ])

    def forward(self, backbone_feature_maps: Sequence[torch.Tensor], *args) -> DecoderOutput:
        feature_maps = []
        x = backbone_feature_maps[-1]
        from monai.networks.nets import Unet, BasicUnet, DynUnet
        for upsampling, layer, skip in zip(self.upsamplings[::-1], self.layers[::-1], backbone_feature_maps[-2::-1]):
            x = upsampling.forward(x, x.shape[-1] < skip.shape[-1])
            x = layer(torch.cat([x, skip], dim=1))
            feature_maps.append(x)

        return DecoderOutput(feature_maps)
