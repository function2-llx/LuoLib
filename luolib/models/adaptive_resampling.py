from typing import Literal

import cytoolz
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as nnf

from monai.networks.blocks import get_output_padding, get_padding

from luolib.models.blocks import InflatableConv3d
from luolib.types import param3_t

class AdaptiveDownsampling(nn.Conv3d):
    STRIDE = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride := self.STRIDE,
            get_padding(kernel_size, stride),
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        if spatial_shape[-1] * 2 > spatial_shape[0]:
            return super().forward(x)
        x = nnf.conv2d(
            rearrange(x, 'n c h w d -> (n d) c h w').contiguous(),
            self.weight[..., self.kernel_size[-1] >> 1],
            self.bias,
            self.stride[:-1],
            self.padding[:-1],
            groups=self.groups,
        )
        return rearrange(x, '(n d) c h w -> n c h w d', n=batch_size).contiguous()

class AdaptiveUpsampling(nn.ConvTranspose3d):
    STRIDE = AdaptiveDownsampling.STRIDE

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride := self.STRIDE,
            padding := get_padding(kernel_size, stride),
            get_output_padding(kernel_size, stride, padding),
            groups,
            bias,
        )

    # packed for sequential forwarding
    def forward(self, input: tuple[torch.Tensor, bool]):   # noqa
        x, upsample_z = input
        if upsample_z:
            return super().forward(x)
        batch_size = x.shape[0]
        x = nnf.conv_transpose2d(
            rearrange(x, 'n c h w d -> (n d) c h w').contiguous(),
            self.weight[..., self.kernel_size[-1] >> 1],
            self.bias,
            self.stride[:-1],
            self.padding[:-1],
            self.output_padding[:-1],
            self.groups,
        )
        return rearrange(x, '(n d) c h w -> n c h w d', n=batch_size).contiguous()

class AdaptiveDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: param3_t[int] = 2,
        conv_t: type[InflatableConv3d] = InflatableConv3d,
        bias: bool = True,
        d_inflation: Literal['average', 'center'] = 'center',
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = conv_t(in_channels, out_channels, kernel_size, stride=2, bias=bias, d_inflation=d_inflation)

    @property
    def in_channels(self):
        return self.conv.out_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}weight', None)) is not None:
            state_dict[f'{prefix}conv.weight'] = weight
            if (bias := state_dict.pop(f'{prefix}bias', None)) is not None:
                state_dict[f'{prefix}conv.bias'] = bias

        if len(state_dict) == 0 and self.in_channels == self.out_channels:
            def gaussian_kernel(sigma: float):
                kernel = torch.tensor(1)
                for size in self.kernel_size:
                    if size == 1:
                        continue
                    dist = torch.linspace(-(size - 1) / 2, (size - 1) / 2, size)
                    exp = torch.exp(-dist ** 2 / (2 * sigma ** 2))
                    kernel = kernel.view(-1).outer(exp)
                kernel = kernel.view(self.kernel_size)
                kernel /= kernel.sum()
                kf = kernel.flatten()
                diff = (kf[0::2].sum() - kf[1::2].sum()).abs()
                return kernel, diff
            # find a gaussian kernel balance the weights for odd and even positions
            # maybe there's a better way to solve it, but this is fast enough and I don't have time to find it
            low, high = 0, 10
            for _ in range(200):
                m1 = low + (high - low) / 3
                m2 = low + (high - low) / 3 * 2
                k1, d1 = gaussian_kernel(m1)
                k2, d2 = gaussian_kernel(m2)
                if d1 < d2:
                    high = m2
                    kernel = k1
                else:
                    low = m1
                    kernel = k2
            weight = torch.zeros_like(self.conv.weight)
            weight[torch.eye(self.out_channels, dtype=torch.bool, device=weight.device)] = kernel.to(weight)
            state_dict[f'{prefix}conv.weight'] = weight
            if self.conv.bias is not None:
                state_dict[f'{prefix}conv.bias'] = torch.zeros_like(self.conv.bias)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor):
        if x.shape[0] != 1:
            raise NotImplementedError('only support batch size of 1')
        spacing = spacing[0]
        downsample_mask = spacing < 2 * spacing.amin()
        stride = [2 if downsample else 1 for downsample in downsample_mask]
        if not downsample_mask.all():
            weight = self.conv.weight.sum(dim=[i + 2 for i, downsample in enumerate(downsample_mask) if not downsample], keepdim=True)
        else:
            weight = self.conv.weight
        pad = [
            (0, 0) if s == 1
            else (k - 2 >> 1, k - 1 >> 1)
            for k, s in zip(weight.shape[2:], stride)
        ]
        pad = list(cytoolz.concat(pad[::-1]))
        x = nnf.pad(x, pad)
        x = nnf.conv3d(x, weight, self.conv.bias, stride)
        new_spacing = spacing.clone()
        new_spacing[downsample_mask] *= 2
        return x, new_spacing[None], downsample_mask[None]

# following VQGAN's implementation
class AdaptiveUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = InflatableConv3d(in_channels, out_channels, kernel_size=3, padding=1)

    @property
    def in_channels(self):
        return self.conv.out_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if len(state_dict) == 0 and self.in_channels == self.out_channels:
            # identity
            weight = torch.zeros_like(self.conv.weight)
            weight[:, :, *(k - 1 >> 1 for k in self.conv.kernel_size)] = torch.eye(self.out_channels)
            state_dict[f'{prefix}conv.weight'] = weight
            state_dict[f'{prefix}conv.bias'] = torch.zeros_like(self.conv.bias)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, upsample_mask: torch.Tensor):
        if upsample_mask.ndim == 2:
            if upsample_mask.shape[0] != 1:
                assert x.shape[0] == upsample_mask.shape[0]
                if not (upsample_mask[0:1] == upsample_mask).all():
                    raise NotImplementedError('only support consistent mask')
            upsample_mask = upsample_mask[0]
        scale_factor = tuple(2. if upsample else 1. for upsample in upsample_mask)
        x = nnf.interpolate(x, scale_factor=scale_factor, mode='nearest-exact')
        x = self.conv(x)
        return x
