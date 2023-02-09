from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as torch_f

from monai.networks.blocks import get_output_padding, get_padding

class AdaptiveDownsampling(nn.Conv3d):
    STRIDE = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = True,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        spatial_shape = x.shape[2:]
        if spatial_shape[-1] * 2 > spatial_shape[0]:
            return super().forward(x), True
        d = x.shape[-1]
        x = rearrange(x, 'n c h w d -> (n d) c h w')
        x = torch_f.conv2d(
            x,
            self.weight[..., self.kernel_size[-1] >> 1],
            self.bias,
            self.stride[:-1],
            self.padding[:-1],
            groups=self.groups,
        )
        return rearrange(x, '(n d) c h w -> n c h w d', d=d), False

class AdaptiveUpsampling(nn.ConvTranspose3d):
    STRIDE = AdaptiveDownsampling.STRIDE

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = True,
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

    def forward(self, x: torch.Tensor, upsample_z: bool):   # noqa
        if upsample_z:
            return super().forward(x)
        d = x.shape[-1]
        x = rearrange(x, 'n c h w d -> (n d) c h w')
        x = torch_f.conv_transpose2d(
            x,
            self.weight[..., self.kernel_size[-1] >> 1],
            self.bias,
            self.stride[:-1],
            self.padding[:-1],
            self.output_padding[:-1],
            self.groups,
        )
        return rearrange(x, '(n d) c h w -> n c h w d', d=d)
