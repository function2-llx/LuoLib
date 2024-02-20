from typing import Literal
import warnings

import einops
import torch
from torch import nn
from torch.nn import functional as nnf

from monai.networks.blocks import get_output_padding, get_padding
from monai.utils import InterpolateMode, ensure_tuple_rep

from luolib.types import param3_t
from luolib.utils import RGB_TO_GRAY_WEIGHT
from .tensor import SpatialTensor

__all__ = [
    'Conv3d',
    'InputConv3D',
    'OutputConv3D',
    'TransposedConv3d',
    'MaxPool',
]

class Conv3d(nn.Conv3d):
    @staticmethod
    def _check_depth_adaptable(kernel_size: int, stride: int):
        assert stride & stride - 1 == 0, 'only power of 2 is supported'
        assert kernel_size == stride or (kernel_size, stride) in {(3, 1), (3, 2), (4, 2)}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 1,
        padding: str | param3_t[int] = 0,
        dilation: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        adaptive: bool = True,
        d_inflation: Literal['average', 'center'] = 'average',
        **kwargs,
    ):
        if padding != 0:
            warnings.warn('specifying padding will have no effect', DeprecationWarning, stacklevel=2)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype,
        )
        self.adaptive = adaptive
        if adaptive:
            # TODO: remove padding parameter, always calculate it dynamically
            assert self.stride[0] == self.stride[1] == self.stride[2], 'only isotropic stride is supported'
            self._check_depth_adaptable(self.kernel_size[0], self.stride[0])
            self.num_downsamples = self.stride[0].bit_length() - 1
            self.padding = get_padding(self.kernel_size, self.stride)
            assert self.padding_mode == 'zeros'
        self.inflation = d_inflation
        assert d_inflation in ['average', 'center']

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        weight_key = f'{prefix}weight'
        if (weight := state_dict.get(weight_key)) is not None and weight.ndim + 1 == self.weight.ndim:
            if weight.shape[2:] != self.kernel_size[1:]:
                weight = nnf.interpolate(weight, self.kernel_size[1:], mode='bicubic')
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

    def forward(self, x: torch.Tensor):
        """
        output shape: x.shape / adapted stride
        """
        if self.adaptive:
            x: SpatialTensor
            stride = list(self.stride)
            padding = list(self.padding)
            if (self.kernel_size[0], self.stride[0]) == (1, 1) or x.num_pending_hw_downsamples == 0:
                # no adaption
                weight = self.weight
            elif self.kernel_size[0] == self.stride[0]:
                # KpSp -> KqSq, where q | p
                stride[0] = self.stride[0] >> min(x.num_pending_hw_downsamples, self.num_downsamples)
                weight = einops.reduce(
                    self.weight,
                    'co ci (dr dc) ... -> co ci dr ...',
                    'sum',
                    dr=stride[0],
                )
            else:
                assert (self.kernel_size[0], self.stride[0]) in {(3, 1), (3, 2), (4, 2)}
                # K3S1, K3S2, K4S2 -> K1S1
                stride[0] = 1
                weight = self.weight.sum(dim=2, keepdim=True)
                padding[0] = 0
            x: SpatialTensor = nnf.conv3d(x, weight, self.bias, stride, padding, self.dilation, self.groups)
            x.num_downsamples += self.num_downsamples
            return x
        else:
            return super().forward(x)

class InputConv3D(Conv3d):
    """This class additionally handles pre-trained weights for input stem layer"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 1,
        padding: str | param3_t[int] = 0,
        dilation: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        adaptive: bool = True,
        d_inflation: Literal['average', 'center'] = 'average',
        force: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype,
            adaptive, d_inflation, **kwargs,
        )
        self.force = force

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.get(weight_key := f'{prefix}weight')) is not None \
        and (weight.shape[1] != (ci := self.weight.shape[1]) or self.force):
            state_dict[weight_key] = einops.repeat(
                einops.reduce(weight, 'co ci ... -> co ...', 'sum') / ci,
                'co ... -> co ci ...', ci=ci,
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class OutputConv3D(Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 1,
        padding: str | param3_t[int] = 0,
        dilation: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        adaptive: bool = True,
        d_inflation: Literal['average', 'center'] = 'average',
        force: bool = False,
        c_inflation: Literal['RGB_L', 'average'] = 'RGB_L',
        **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype,
            adaptive, d_inflation, **kwargs,
        )
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
                    # RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
                    weight = einops.einsum(
                        weight.new_tensor(RGB_TO_GRAY_WEIGHT), weight,
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
                    bias = torch.dot(bias.new_tensor(RGB_TO_GRAY_WEIGHT), bias)[None]
            state_dict[bias_key] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class AdaptiveInterpolationDownsample(nn.Module):
    def __init__(self, mode: InterpolateMode = InterpolateMode.AREA, antialias: bool = False):
        super().__init__()
        self.mode = mode
        self.antialias = antialias

    def forward(self, x: SpatialTensor):
        x = nnf.interpolate(
            x,
            scale_factor=(0.5 if x.can_downsample_d else 1, 0.5, 0.5),
            mode=self.mode,
            antialias=self.antialias,
        )
        x.num_downsamples += 1
        return x

class AdaptiveConvDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: param3_t[int] = 2,
        bias: bool = True,
        conv_t: type[Conv3d] = Conv3d,
        d_inflation: Literal['average', 'center'] = 'average',
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = conv_t(
            in_channels, out_channels, kernel_size,
            stride=2,
            padding=tuple(k - 1 >> 1 for k in ensure_tuple_rep(kernel_size, 3)),
            bias=bias,
            d_inflation=d_inflation,
        )

    def forward(self, x: SpatialTensor):
        return self.conv(x)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}weight', None)) is not None:
            state_dict[f'{prefix}conv.weight'] = weight
            if (bias := state_dict.pop(f'{prefix}bias', None)) is not None:
                state_dict[f'{prefix}conv.bias'] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class AdaptiveInterpolationUpsample(nn.Module):
    def __init__(self, mode: InterpolateMode = InterpolateMode.TRILINEAR):
        super().__init__()
        self.mode = mode

    def upsample(self, x: SpatialTensor) -> SpatialTensor:
        return nnf.interpolate(x, scale_factor=(2. if x.can_upsample_d else 1., 2., 2.), mode=self.mode)

    def forward(self, x: SpatialTensor):
        x = self.upsample(x)
        x.num_downsamples -= 1
        return x

class AdaptiveInterpolationUpsampleWithPostConv(AdaptiveInterpolationUpsample):
    # following VQGAN's implementation
    def __init__(self, in_channels: int, out_channels: int | None = None, mode: InterpolateMode = InterpolateMode.NEAREST_EXACT):
        super().__init__(mode)
        if out_channels is None:
            out_channels = in_channels
        self.conv = Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: SpatialTensor):
        x = super().forward(x)
        x = self.conv(x)
        return x

class TransposedConv3d(nn.ConvTranspose3d):
    @staticmethod
    def _check_depth_adaptable(kernel_size: int, stride: int):
        assert stride & stride - 1 == 0, 'only power of 2 is supported'
        assert kernel_size == stride or (kernel_size, stride) in {(3, 2), (4, 2)}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: param3_t[int] = 1,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, 0,
            groups, bias, dilation, padding_mode, device, dtype,
        )
        assert self.stride[0] == self.stride[1] == self.stride[2], 'only isotropic stride is supported'
        self._check_depth_adaptable(self.kernel_size[0], self.stride[0])
        self.num_upsamples = self.stride[1].bit_length() - 1
        self.padding = get_padding(self.kernel_size, self.stride)
        self.output_padding = get_output_padding(self.kernel_size, self.stride, self.padding)

    def forward(self, x: SpatialTensor, output_size=None):
        assert output_size is None
        stride = list(self.stride)
        padding = list(self.padding)
        output_padding = list(self.output_padding)
        if self.num_upsamples <= x.num_remained_d_upsamples:
            # no adaption
            weight = self.weight
        elif self.kernel_size[0] == self.stride[0]:
            # KpSp -> KqSq, where q | p
            stride[0] = 1 << x.num_remained_d_upsamples
            weight = einops.reduce(
                self.weight,
                'co ci (dr dc) ... -> co ci dr ...',
                'sum',
                dr=stride[0],
            )
        else:
            # K4S2, K3S2 -> K1S1
            stride[0] = 1
            weight = self.weight.sum(dim=2, keepdim=True)
            padding[0] = output_padding[0]
        x: SpatialTensor = nnf.conv_transpose3d(
            x, weight, self.bias, stride, padding, output_padding,
            self.groups, self.dilation,
        )
        x.num_downsamples -= self.num_upsamples
        return x

class AdaptiveTransposedConvUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.transposed_conv = TransposedConv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.conv = nn.Sequential(
            Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: SpatialTensor):
        x = self.transposed_conv(x)
        return self.conv(x)

class MaxPool(nn.MaxPool3d):
    def __init__(
        self,
        kernel_size: param3_t,
        stride: param3_t | None = None,
        padding: param3_t = 0,
        dilation: param3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        assert self.kernel_size == (2, 2, 2) or self.kernel_size == 2
        assert stride is None
        assert self.padding == (0, 0, 0) or self.padding == 0
        assert self.dilation == (1, 1, 1) or self.dilation == 1

    def forward(self, x: SpatialTensor):
        x: SpatialTensor
        kernel_size = list(self.kernel_size)
        kernel_size[0] = max(self.kernel_size[0] >> x.num_pending_hw_downsamples, 1)
        x = nnf.max_pool3d(
            x, kernel_size, kernel_size,
            self.padding, self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices,
        )
        x.num_downsamples += 1
        return x
