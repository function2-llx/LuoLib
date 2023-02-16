from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.layers import get_norm_layer
from umei.models.adaptive_resampling import AdaptiveUpsampling
from umei.models.backbones.swin import SwinLayer
from umei.models.init import init_linear_conv
from umei.models.layers import Norm

class SwinVQDecoder(nn.Module):
    layers: Sequence[SwinLayer] | nn.ModuleList
    upsamplings: Sequence[AdaptiveUpsampling] | nn.ModuleList

    def __init__(
        self,
        layer_channels: Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        layer_depths: Sequence[int],
        num_heads: Sequence[int],
        post_upsampling_channels: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        num_layers = len(layer_depths)
        if isinstance(layer_channels, int):
            layer_channels = [layer_channels << i for i in range(num_layers)]

        self.layers = nn.ModuleList([
            SwinLayer(
                layer_channels[i],
                layer_depths[i],
                num_heads[i],
                _max_window_size := kernel_sizes[i],
                _drop_path := 0.,
                mlp_ratio,
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                Norm.LAYER,
                use_checkpoint,
            )
            for i in range(num_layers)
        ])

        self.upsamplings = nn.ModuleList([
            AdaptiveUpsampling(
                layer_channels[i + 1],
                layer_channels[i],
            )
            for i in range(num_layers - 1)
        ])

        self.norms = nn.ModuleList([
            get_norm_layer(Norm.LAYERND, 3, layer_channels[i])
            for i in range(num_layers)
        ])

        self.post_upsampling = nn.Sequential()
        for i, channels in enumerate(post_upsampling_channels):
            self.post_upsampling.append(
                AdaptiveUpsampling(
                    post_upsampling_channels[i - 1] if i > 0 else layer_channels[0],
                    post_upsampling_channels[i],
                )
            )
            if i < len(post_upsampling_channels) - 1:
                self.post_upsampling.append(get_norm_layer(Norm.INSTANCE, 3, post_upsampling_channels[i]))

        self.apply(init_linear_conv)

    def no_weight_decay(self):
        ret = set()
        for name, _ in self.named_parameters():
            if 'relative_position_bias_table' in name:
                ret.add(name)
        return ret

    def forward(self, x: torch.Tensor, target_z_size: int):
        for i in reversed(range(len(self.layers))):
            x = self.layers[i](x)
            x = self.norms[i](x)
            if i:
                x = self.upsampling[i - 1](x, x.shape[-1] < target_z_size)
        x = self.post_upsampling(x)
        return x
