from torch import nn

from monai.networks.layers import get_act_layer

from ..layers import LayerNormNd
from .conv import Conv3d, InputConv3D, TransposedConv3d

__all__ = [
    'PatchEmbed',
    'InversePatchEmbed',
]

class PatchEmbed(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        kernel_size: int,
        hierarchical: bool,
        act: str = '',
    ):
        """
        Args:
            kernel_size: convolution kernel size in hierarchical structure
        """
        assert patch_size & patch_size - 1 == 0, 'only power of 2 is supported'
        if hierarchical:
            padding = kernel_size - 1 >> 1
            num_downsamples = patch_size.bit_length() - 1
            super().__init__(
                InputConv3D(
                    in_channels, out_channels >> num_downsamples - 1,
                    kernel_size, 2, padding,
                ),
                *[
                    nn.Sequential(
                        LayerNormNd(out_channels >> num_downsamples - i),
                        get_act_layer(act),
                        Conv3d(
                            out_channels >> num_downsamples - i,
                            out_channels >> num_downsamples - i - 1,
                            kernel_size, 2, padding,
                        ),
                    )
                    for i in range(1, num_downsamples)
                ],
            )
        else:
            super().__init__(InputConv3D(in_channels, out_channels, patch_size, patch_size))

class InversePatchEmbed(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        hierarchical: bool,
        act: str = '',
    ):
        super().__init__()
        assert patch_size & patch_size - 1 == 0, 'only power of 2 is supported'
        if not hierarchical or patch_size <= 4:
            self.append(TransposedConv3d(in_channels, out_channels, patch_size, patch_size))
        else:
            num_upsamples = patch_size.bit_length() - 3
            for i in range(num_upsamples):
                self.extend([
                    TransposedConv3d(in_channels >> i, in_channels >> i + 1, 2, 2),
                    LayerNormNd(in_channels >> i + 1),
                    get_act_layer(act),
                ])
            self.append(TransposedConv3d(in_channels >> num_upsamples, out_channels, 4, 4))
