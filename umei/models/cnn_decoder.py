from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import UnetResBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.umei import UDecoderBase, UDecoderOutput

# Modified UnetUpBlock
class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            2 * in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetResBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to in_channels
        out = torch.cat((inp, skip), dim=1)
        out = self.transp_conv(out)
        out = self.conv_block(out)
        return out


class CnnDecoder(UDecoderBase):
    def __init__(
        self,
        encoder_feature_sizes: list[int],
        out_channels: int,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        self.bottleneck = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_feature_sizes[4],
            out_channels=encoder_feature_sizes[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder4 = UpBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_feature_sizes[4],
            out_channels=encoder_feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder3 = UpBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_feature_sizes[3],
            out_channels=encoder_feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder2 = UpBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_feature_sizes[2],
            out_channels=encoder_feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder1 = UpBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_feature_sizes[1],
            out_channels=encoder_feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder0 = UpBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_feature_sizes[0],
            out_channels=out_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

    def forward(self, x_in, hidden_states: list[torch.Tensor]):
        dec4 = self.bottleneck(hidden_states[4])
        dec3 = self.decoder4(dec4, hidden_states[4])
        dec2 = self.decoder3(dec3, hidden_states[3])
        dec1 = self.decoder2(dec2, hidden_states[2])
        dec0 = self.decoder1(dec1, hidden_states[1])
        out = self.decoder0(dec0, hidden_states[0])
        return UDecoderOutput([dec3, dec2, dec1, dec0, out])
