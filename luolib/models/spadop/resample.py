import numpy as np
import torch
from torch.nn import functional as nnf

from luolib.types import tuple3_t

__all__ = [
    'resample',
]

def resample(x: torch.Tensor, shape: tuple3_t[int]):
    # without `.tolist()`, PyTorch will complain it is not int
    downsample_shape = tuple(np.minimum(x.shape[2:], shape).tolist())
    if downsample_shape != x.shape[2:]:
        x = nnf.interpolate(x, downsample_shape, mode='area')
    if shape != x.shape[2:]:
        x = nnf.interpolate(x, shape, mode='trilinear')
    return x
