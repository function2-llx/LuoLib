from typing import TypeVar

import torch

__all__ = [
    'fall_back_none',
    'RGB_TO_GRAY_WEIGHT',
    'ema_update',
]

T = TypeVar('T')
U = TypeVar('U')

def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)

def ema_update(ema: torch.Tensor, x: torch.Tensor, decay: float):
    return ema.mul_(decay).add_(x, alpha=1 - decay)
