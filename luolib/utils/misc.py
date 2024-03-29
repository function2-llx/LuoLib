from typing import TypeVar

from einops import einops
import torch

__all__ = [
    'fall_back_none',
    'RGB_TO_GRAY_WEIGHT',
    'ema_update',
    'as_tensor',
]

from monai.data import MetaTensor

T = TypeVar('T')
U = TypeVar('U')

def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)

def ema_update(ema: torch.Tensor, x: torch.Tensor, decay: float):
    return ema.mul_(decay).add_(x, alpha=1 - decay)

def ensure_rgb(x: torch.Tensor, batched: bool = False) -> tuple[torch.Tensor, bool]:
    if x.shape[batched] == 3:
        not_rgb = False
    else:
        assert x.shape[batched] == 1
        maybe_batch = 'n' if batched else ''
        x = einops.repeat(x, f'{maybe_batch} 1 ... -> c ...', c=3)
        not_rgb = True
    return x, not_rgb

def as_tensor(x: torch.Tensor):
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    return x
