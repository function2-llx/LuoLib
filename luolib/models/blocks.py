import itertools as it
import warnings
from typing import Sequence, Literal

import einops
import numpy as np
import torch
from torch import nn

from monai.networks.blocks import Convolution, get_output_padding, get_padding
from monai.networks.layers import Conv, DropPath, Pool, get_act_layer, get_norm_layer

from luolib.models.layers import Act, Norm
from luolib.types import spatial_param_t

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    groups: int = 1,
    norm: tuple | str | None = Norm.INSTANCE,
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
        norm: tuple | str = Norm.INSTANCE,
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
                    Conv[Conv.CONV, spatial_dims](in_channels, out_channels, kernel_size=1, stride=1),
                    get_norm_layer(norm, spatial_dims, out_channels),
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
        norm: tuple | str = Norm.INSTANCE,
        act: tuple | str = Act.LEAKYRELU,
        drop_paths: float | Sequence[float] = 0.,
        res_block: bool = True,
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
        norm: tuple | str = Norm.INSTANCE,
        act: tuple | str = Act.LEAKYRELU,
        res: bool = False,
    ):
        super().__init__()
        self.upsample = Conv[Conv.CONVTRANS, spatial_dims](
            in_channels,
            out_channels,
            upsample_stride,
            upsample_stride,
            padding := get_padding(upsample_stride, upsample_stride),
            get_output_padding(upsample_stride, upsample_stride, padding),
        )
        self.conv = BasicConvBlock(
            spatial_dims,
            out_channels << 1,
            out_channels,
            kernel_size,
            stride=1,
            norm=norm,
            act=act,
            res=res,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class InflatableConv3d(nn.Conv3d):
    def __init__(self, *args, d_inflation: Literal['average', 'center'] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if d_inflation is None:
            if self.stride[0] == 2 and self.kernel_size[0] == 3:
                d_inflation = 'average'
            else:
                d_inflation = 'center'
        assert d_inflation in ['average', 'center']
        self.inflation = d_inflation

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        weight_key = f'{prefix}weight'
        if (weight := state_dict.get(weight_key)) is not None and weight.ndim + 1 == self.weight.ndim:
            d = self.kernel_size[0]
            match self.inflation:
                case 'average':
                    weight = einops.repeat(weight / d, 'co ci ... -> co ci d ...', d=d)
                case 'center':
                    new_weight = weight.new_zeros(*weight.shape[:2], d, *weight.shape[2:])
                    if d & 1:
                        new_weight[:, :, d >> 1] = weight
                    else:
                        new_weight[:, :, [d - 1 >> 1, d >> 1]] = weight[:, :, None] / 2
                    weight = new_weight
                case _:
                    raise ValueError
            state_dict[weight_key] = weight
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class InflatableInputConv3d(InflatableConv3d):
    def __init__(self, *args, force: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force = force

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.get(weight_key := f'{prefix}weight')) is not None \
        and (weight.shape[1] != (ci := self.weight.shape[1]) or self.force):
            state_dict[weight_key] = einops.repeat(
                einops.reduce(weight, 'co ci ... -> co ...', 'sum') / ci,
                'co ... -> co ci ...', ci=ci,
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
class InflatableOutputConv3d(InflatableConv3d):
    def __init__(self, *args, force: bool = False, c_inflation: Literal['RGB_L', 'average'] = 'RGB_L', **kwargs):
        super().__init__(*args, **kwargs)
        self.force = force
        assert c_inflation in ['RGB_L', 'average']
        self.c_inflation = c_inflation

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.get(weight_key := f'{prefix}weight')) is not None \
        and (weight.shape[0] != (co := self.weight.shape[0]) or self.force):
            match self.c_inflation:
                case 'average':
                    weight = einops.repeat(
                        einops.reduce(weight, 'co ci ... -> ci ...', 'sum') / co,
                        'ci ... -> co ci ...', co=co,
                    )
                case 'RGB_L':
                    assert weight.shape[0] == 3 and self.out_channels == 1
                    weight = einops.einsum(
                        weight.new_tensor([0.299, 0.587, 0.114]), weight,
                        'c, c ... -> ...'
                    )[None]

            state_dict[weight_key] = weight

        if (bias := state_dict.get(bias_key := f'{prefix}bias')) is not None \
        and (bias.shape[0] != (co := self.bias.shape[0]) or self.force):
            match self.c_inflation:
                case 'average':
                    bias = einops.repeat(
                        einops.reduce(bias, 'co -> ', 'sum') / co,
                        ' -> co', co=co,
                    )
                case 'RGB_L':
                    bias = torch.dot(bias.new_tensor([0.299, 0.587, 0.114]), bias)[None]
            state_dict[bias_key] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
