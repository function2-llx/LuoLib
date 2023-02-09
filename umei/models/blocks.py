from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution, get_output_padding, get_padding
from monai.networks.layers import Act, DropPath, Norm, get_act_layer

def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    groups: int = 1,
    dropout: tuple | str | float | None = None,
    norm: tuple | str | None = Norm.LAYERND,
    act: tuple | str | None = Act.GELU,
    adn_ordering: str = "DNA",
    bias: bool = False,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        3,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        adn_ordering=adn_ordering,
        act=act,
        norm=norm,
        dropout=dropout,
        groups=groups,
        bias=bias,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

class ResBasicBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
        norm: tuple | str = Norm.LAYERND,
        act: tuple | str = Act.GELU,
        drop_path: float = .0,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            channels,
            channels,
            kernel_size,
            dropout=dropout,
            norm=norm,
            act=act,
        )
        self.conv2 = get_conv_layer(
            channels,
            channels,
            kernel_size,
            dropout=dropout,
            norm=norm,
            act=None,
        )
        self.act2 = get_act_layer(act)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop_path(x) + res
        x = self.act2(x)
        return x

class ResLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        channels: int,
        kernel_size: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
        norm: tuple | str = Norm.LAYERND,
        act: tuple | str = Act.GELU,
        drop_path: list[float] = None,
    ):
        super().__init__()
        if drop_path is None:
            drop_path = [0.] * num_blocks
        self.blocks = nn.Sequential(*[
            ResBasicBlock(channels, kernel_size, dropout, norm, act, drop_path=dp)
            for dp in drop_path
        ])

    def forward(self, x: torch.Tensor):
        return self.blocks(x)
