from typing import Sequence

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
import torch

__all__ = [
    'ChannelFirst',
    'channel_first',
    'ChannelLast',
    'channel_last',
    'flatten',
    'spatialize',
]

class ChannelFirst(Rearrange):
    def __init__(self):
        super().__init__('n ... c -> n c ...')

def channel_first(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'n ... c -> n c ...')

class ChannelLast(Rearrange):
    def __init__(self):
        super().__init__('n c ... -> n ... c')

def channel_last(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'n c ... -> n ... c')

def flatten(x: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(x, 'n c ... -> n (...) c')

def spatialize(x: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    spatial_dims = len(spatial_shape)
    spatial_pattern = ' '.join(map(lambda i: f's{i}', range(spatial_dims)))
    spatial_dict = {
        f's{i}': s
        for i, s in enumerate(spatial_shape)
    }
    return einops.rearrange(x, f'n ({spatial_pattern}) d -> n d {spatial_pattern}', **spatial_dict)
