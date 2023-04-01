from collections.abc import Mapping, Sequence

import torch
from torch import nn

from monai.umei import Decoder, DecoderOutput

from umei.conf import ModelConf
from umei.types import spatial_param_seq_t
from ..registry import decoder_registry
from ..blocks import BasicConvLayer, UNetUpLayer
from ..layers import Act, Norm
from ..utils import create_model

__all__ = []

@decoder_registry.register_module('full-res-adapter')
class FullResAdapter(Decoder):
    def __init__(
        self,
        inner_decoder_conf: Mapping,
        spatial_dims: int,
        num_input_channels: int,
        layer_channels: Sequence[int],
        kernel_sizes: spatial_param_seq_t[int],
        strides: spatial_param_seq_t[int],
        norm: tuple | str = Norm.INSTANCE,
        act: tuple | str = Act.LEAKYRELU,
    ):
        super().__init__()
        num_layers = len(layer_channels) - 1
        # complement default values
        self.inner_decoder = create_model(ModelConf(**inner_decoder_conf), decoder_registry)
        self.encode_layers = nn.ModuleList([
            BasicConvLayer(
                spatial_dims=spatial_dims,
                num_blocks=2,
                in_channels=num_input_channels if i == 0 else layer_channels[i - 1],
                out_channels=layer_channels[i],
                kernel_size=kernel_sizes[i],
                stride=1 if i == 0 else strides[i - 1],
                norm=norm,
                act=act,
                res_block=False,
            )
            for i in range(num_layers)
        ])
        self.decode_layers = nn.ModuleList([
            UNetUpLayer(spatial_dims, layer_channels[i + 1], layer_channels[i], kernel_sizes[i])
            for i in range(num_layers)
        ])

    def forward(self, backbone_feature_maps: list[torch.Tensor], x_in: torch.Tensor) -> DecoderOutput:
        output = self.inner_decoder.forward(backbone_feature_maps, x_in)
        encodes = []
        for encode_layer in self.encode_layers:
            x_in = encode_layer(x_in)
            encodes.append(x_in)
        for decode_layer, skip in zip(self.decode_layers[::-1], encodes[::-1]):
            output.feature_maps.append(decode_layer(output.feature_maps[-1], skip))

        return output
