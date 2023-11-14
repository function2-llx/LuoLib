from typing import Sequence

import numpy as np
import torch
from torch import nn

from monai.networks.blocks import Convolution, get_output_padding, get_padding
from monai.networks.layers import DropPath, Pool, get_act_layer

from luolib.types import spatial_param_t
from ..layers import Act, Norm

__all__ = [
    'get_conv_layer',
    'BasicConvLayer',
    'BasicConvBlock',
    'UNetUpLayer',
]

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    groups: int = 1,
    norm: tuple | str | None = (Norm.INSTANCE, {'affine': True}),
    act: tuple | str | None = Act.LEAKYRELU,
    adn_ordering: str = "DNA",
    bias: bool = False,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        adn_ordering=adn_ordering,
        act=act,
        norm=norm,
        groups=groups,
        bias=bias,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

class BasicConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: spatial_param_t[int],
        stride: spatial_param_t[int],
        norm: tuple | str = (Norm.INSTANCE, {'affine': True}),
        act: tuple | str = Act.LEAKYRELU,
        drop_path: float = .0,
        res: bool = True,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm=norm,
            act=act,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size,
            norm=norm,
            act=None,
        )
        if res:
            if in_channels != out_channels or np.prod(stride) > 1:
                self.res = nn.Sequential(
                    Pool[Pool.AVG, spatial_dims](stride, stride),
                    get_conv_layer(
                        spatial_dims, in_channels, out_channels, 1, 1, norm=norm, act=None,
                    ),
                )
            else:
                self.res = nn.Identity()
        else:
            self.res = None
        self.act2 = get_act_layer(act)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        res = None if self.res is None else self.res(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop_path(x)
        if res is not None:
            x += res
        x = self.act2(x)
        return x

class BasicConvLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: spatial_param_t[int],
        stride: spatial_param_t[int],
        norm: tuple | str = (Norm.INSTANCE, {'affine': True}),
        act: tuple | str = Act.LEAKYRELU,
        res_block: bool = True,
        drop_paths: float | Sequence[float] = 0.,
    ):
        super().__init__()
        if isinstance(drop_paths, float):
            drop_paths = [drop_paths] * num_blocks
        assert len(drop_paths) == num_blocks
        self.blocks = nn.Sequential(
            BasicConvBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm, act, drop_paths[0], res_block),
            *[
                BasicConvBlock(spatial_dims, out_channels, out_channels, kernel_size, 1, norm, act, drop_path, res_block)
                for drop_path in drop_paths[1:]
            ],
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)

class UNetUpLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: spatial_param_t[int],
        upsample_stride: spatial_param_t[int],
        norm: tuple | str = (Norm.INSTANCE, {'affine': True}),
        act: tuple | str = Act.LEAKYRELU,
        upsample_norm: tuple | str | None = (Norm.INSTANCE, {'affine': True}),
        upsample_act: tuple | str | None = Act.LEAKYRELU,
        res: bool = False,
    ):
        super().__init__()
        self.upsample = get_conv_layer(
            spatial_dims, in_channels, out_channels, upsample_stride, upsample_stride,
            norm=upsample_norm, act=upsample_act, is_transposed=True,
        )
        self.conv = BasicConvBlock(
            spatial_dims,
            out_channels << 1, out_channels,
            kernel_size, 1,
            norm=norm, act=act,
            res=res,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
