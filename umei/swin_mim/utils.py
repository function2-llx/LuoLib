import torch
from einops import rearrange

def channel_first(x: torch.Tensor):
    return rearrange(x, 'n h w d c -> n c h w d')

def channel_last(x: torch.Tensor):
    return rearrange(x, 'n c h w d -> n h w d c')
