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
        hierarchical: bool,
        act: str = '',
    ):
        super().__init__()
        assert patch_size & patch_size - 1 == 0, 'only power of 2 is supported'
        if not hierarchical or patch_size <= 4:
            self.append(InputConv3D(in_channels, out_channels, patch_size, patch_size))
        else:
            num_downsamples = patch_size.bit_length() - 3
            self.append(InputConv3D(in_channels, out_channels >> num_downsamples, 4, 4))
            for i in range(num_downsamples):
                self.extend([
                    LayerNormNd(out_channels >> num_downsamples - i),
                    get_act_layer(act),
                    Conv3d(
                        out_channels >> num_downsamples - i,
                        out_channels >> num_downsamples - i - 1,
                        2, 2,
                    ),
                ])

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
