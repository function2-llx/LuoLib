import cytoolz
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.models.blocks import InflatableConv3d
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import get_output_padding, get_padding

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
    def __init__(self, in_channels: int, out_channels: int | None = None, kernel_size: param3_t[int] = 2):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = InflatableConv3d(in_channels, out_channels, kernel_size)

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

class AdaptiveUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = InflatableConv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, upsample_mask: torch.Tensor):
        if x.shape[0] != 1:
            raise NotImplementedError('only support batch size of 1')
        upsample_mask = upsample_mask[0]
        scale_factor = tuple(2. if upsample else 1. for upsample in upsample_mask)
        x = nnf.interpolate(x, scale_factor=scale_factor, mode="nearest-exact")
        x = self.conv(x)
        return x
