from collections.abc import Sequence

from einops.layers.torch import Rearrange
import torch
from einops import rearrange

def channel_first(x: torch.Tensor):
    return rearrange(x, 'n h w d c -> n c h w d')

def channel_last(x: torch.Tensor):
    return rearrange(x, 'n c h w d -> n h w d c')

class ChannelFirst(Rearrange):
    def __init__(self):
        super().__init__('n h w d c -> n c h w d')

class ChannelLast(Rearrange):
    def __init__(self):
        super().__init__('n c h w d -> n h w d c')

def patchify(x: torch.Tensor, patch_shape: Sequence[int]):
    return rearrange(
        x,
        'n c (h ph) (w pw) (d pd) -> n h w d (ph pw pd c)',
        **patch_axes_lengths(patch_shape),
    )

def unpatchify(x: torch.Tensor, patch_shape: Sequence[int]):
    return rearrange(
        x,
        'n h w d (ph pw pd c) -> n c (h ph) (w pw) (d pd)',
        **patch_axes_lengths(patch_shape),
    )

def patch_axes_lengths(patch_shape: Sequence[int]) -> dict:
    return {
        'ph': patch_shape[0],
        'pw': patch_shape[1],
        'pd': patch_shape[2],
    }
