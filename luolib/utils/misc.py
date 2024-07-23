from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from einops import einops
import torch

__all__ = [
    'fall_back_none',
    'RGB_TO_GRAY_WEIGHT',
    'ema_update',
    'as_tensor',
    'to_nifti',
    'pairwise_forward',
]

from luolib.types import PathLike
from monai.data import MetaTensor

T = TypeVar('T')
U = TypeVar('U')

def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)

def ema_update(ema: torch.Tensor, x: torch.Tensor, decay: float):
    return ema.mul_(decay).add_(x, alpha=1 - decay)

def ensure_rgb(x: torch.Tensor, batched: bool = False, contiguous: bool = False) -> tuple[torch.Tensor, bool]:
    if x.shape[batched] == 3:
        not_rgb = False
    else:
        assert x.shape[batched] == 1
        maybe_batch = 'n' if batched else ''
        x = einops.repeat(x, f'{maybe_batch} 1 ... -> c ...', c=3)
        not_rgb = True
    if contiguous:
        x = x.contiguous()
    return x, not_rgb

def as_tensor(x: torch.Tensor):
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    return x

def to_nifti(path: PathLike, output_path: Path | None = None):
    import monai.transforms as mt
    import nibabel as nib
    loader = mt.LoadImage()
    path = Path(path)
    x: MetaTensor = loader(path)
    if output_path is None:
        output_path = path.with_name(path.name + '.nii.gz')
    nib.save(
        nib.Nifti1Image(x.numpy(), x.affine.numpy()),
        output_path,
    )

def ceil_divide(a: T, b: U) -> T | U:
    return -(a // -b)

def pairwise_forward(forward: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    n, m = x.shape[0], y.shape[0]
    # flatten the prefixed dimension, in case the forward function does not support arbitrary prefix
    x = einops.repeat(x, 'n ... -> (n m) ...', m=m)
    y = einops.repeat(y, 'm ... -> (n m) ...', n=n)
    # NOTE: make sure that the results of forward is not reduced, may be ensured by kwargs
    ret = forward(x, y, **kwargs)
    ret = einops.reduce(ret, '(n m) ... -> n m', 'mean', n=n, m=m)
    return ret

def hash_tensor(x: torch.Tensor) -> int:
    return hash(tuple(x.flatten().tolist()))

def min_stem(path: Path):
    suffix_len = sum(map(len, path.suffixes))
    return path.name[:-suffix_len]
