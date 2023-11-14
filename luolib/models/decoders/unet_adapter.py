from collections.abc import Sequence

import torch
from torch import nn

from luolib.types import spatial_param_seq_t
from luolib.utils import fall_back_none
from ..blocks import BasicConvLayer, UNetUpLayer
from ..layers import Act, Norm

__all__ = ['UNetAdapter']

class UNetAdapter(nn.Module):
    """
    Upsample the "bottleneck" feature map (the feature map with highest resolution output by the previous backbone in
    this context) with give strides
    """

    def __init__(
        self,
        spatial_dims: int,
        num_input_channels: int,
        layer_channels: Sequence[int],
        bottleneck_channels: int,
        kernel_sizes: spatial_param_seq_t[int],
        strides: spatial_param_seq_t[int],
        num_up: int | None = None,
        norm: tuple | str = (Norm.INSTANCE, {'affine': True}),
        act: tuple | str = Act.LEAKYRELU,
        upsample_norm: tuple | str | None = None,
        upsample_act: tuple | str | None = None,
        res_block: bool = False,
        **kwargs,
    ):
        """
        Args:
            layer_channels: the feature maps channels produced by this module, resolution: high → low
            bottleneck_channels: number of channels of first feature map produced by previous backbone
            layer_blocks: how many basic conv blocks (each with 2 convs) per layer
            strides: bottleneck shape * strides = input shape
        """
        super().__init__(**kwargs)
        layer_channels = list(layer_channels)
        layer_channels.append(bottleneck_channels)
        num_layers = len(layer_channels) - 1
        num_up = fall_back_none(num_up, num_layers)
        assert 1 <= num_up <= num_layers
        self.encode_layers = nn.ModuleList([
            BasicConvLayer(
                spatial_dims,
                1,
                num_input_channels if i == 0 else layer_channels[i - 1],
                layer_channels[i],
                kernel_sizes[i], strides[i],
                norm, act,
                res_block and i > 0,
            )
            for i in range(num_layers)
        ])
        self.decode_layers = nn.ModuleList([
            UNetUpLayer(
                spatial_dims,
                layer_channels[i + 1], layer_channels[i],
                kernel_sizes[i], strides[i],
                norm, act,
                upsample_norm, upsample_act,
            )
            for i in range(num_layers - num_up, num_layers)
        ])

    def forward(self, feature_maps: list[torch.Tensor], x: torch.Tensor):
        ret = feature_maps[::-1]
        encodes = []
        for encode_layer in self.encode_layers:
            x = encode_layer(x)
            encodes.append(x)
        for decode_layer, skip in zip(self.decode_layers[::-1], encodes[::-1]):
            ret.append(decode_layer(ret[-1], skip))
        return ret[::-1]
