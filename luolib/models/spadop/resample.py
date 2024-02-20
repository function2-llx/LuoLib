import numpy as np
import torch
from torch.nn import functional as nnf

from luolib.types import tuple3_t

__all__ = [
    'resample',
]

def resample(x: torch.Tensor, shape: tuple[int, ...], upsample_mode: str | None = None, scale: bool = False):
    """
    Args:
        scale: whether to scale the values based on size, this can be useful for adapting convolution weights
    """
    scale_ratio = np.prod(x.shape[2:]) / np.prod(shape) if scale else 1.
    # without `.tolist()`, PyTorch will complain it is not int
    downsample_shape = tuple(np.minimum(x.shape[2:], shape).tolist())
    if downsample_shape != x.shape[2:]:
        x = nnf.interpolate(x, downsample_shape, mode='area')
    if shape != x.shape[2:]:
        if upsample_mode is None:
            upsample_mode = 'trilinear' if x.ndim == 5 else 'bicubic'
        x = nnf.interpolate(x, shape, mode=upsample_mode)
    if scale:
        x *= scale_ratio
    return x
