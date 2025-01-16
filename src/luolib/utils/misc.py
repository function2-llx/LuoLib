from __future__ import annotations

from pathlib import Path
from typing import TypeVar, TYPE_CHECKING, Callable, Hashable, Iterable

import cytoolz
import numpy as np
import pandas as pd
import torch
from einops import einops
from einops.einops import Reduction
from monai.data import MetaTensor
from torch import nn

from luolib.types import PathLike

if TYPE_CHECKING:
    from pandas._typing import DropKeep

__all__ = [
    'fall_back_none',
    'RGB_TO_GRAY_WEIGHT',
    'EMA_update',
    'ensure_rgb',
    'as_tensor',
    'to_nifti',
    'ceil_divide',
    'pairwise_forward',
    'hash_tensor',
    'min_stem',
    'concat_drop_dup',
    'compute_grad_norm',
    'partition_by_predicate',
    'import_object',
]

T = TypeVar('T')
U = TypeVar('U')

def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)

def EMA_update(ema: torch.Tensor, x: torch.Tensor, decay: float):
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

def pairwise_forward(forward: Callable, x: torch.Tensor, y: torch.Tensor, reduction: Reduction = 'mean', **kwargs) -> torch.Tensor:
    """Apply a forward function pairwise to two tensors and optionally reduce the results.

    This function takes two tensors, `x` and `y`, and applies a given `forward`
    function to each pair of elements from `x` and `y`. The results can then
    be reduced using the specified reduction method across the pairwise combinations.

    :param forward: A callable function that takes two tensors and additional
                    keyword arguments, and returns a tensor.
    :param x: A tensor of shape (n, ...), where n is the number of elements
              to pair with elements from `y`.
    :param y: A tensor of shape (m, ...), where m is the number of elements
              to pair with elements from `x`.
    :param reduction: The reduction method to apply. Options include 'mean', 'sum', etc.
                      If None, no reduction is performed.
    :param kwargs: Additional keyword arguments to pass to the `forward` function.
    :return: A tensor of shape (n, m) containing the reduced results from
             applying the `forward` function to each pair of elements from `x` and `y`.
    """
    n, m = x.shape[0], y.shape[0]
    # flatten the prefixed dimension, in case the forward function does not support arbitrary prefix
    x = einops.repeat(x, 'n ... -> (n m) ...', m=m)
    y = einops.repeat(y, 'm ... -> (n m) ...', n=n)
    # NOTE: make sure that the results of forward is not reduced when reduction is not None, this may be ensured by kwargs
    ret = forward(x, y, **kwargs)
    if reduction is not None:
        ret = einops.reduce(ret, '(n m) ... -> n m', reduction, n=n, m=m)
    return ret

def hash_tensor(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return hash(x.tobytes())

def min_stem(path: Path) -> str:
    """Gets the stem of a path without any suffixes.

    Unlike :meth:`pathlib.Path.stem` which only removes the last suffix, this function removes all suffixes.

    Args:
        path: A Path object representing a file path

    Returns:
        The filename with all suffixes removed

    Examples:
        >>> min_stem(Path('file.tar.gz'))
        'file'
        >>> min_stem(Path('image.001.jpg'))
        'image'
    """
    suffix_len = sum(map(len, path.suffixes))
    return path.name[:-suffix_len]

def concat_drop_dup(
    objs: Iterable[pd.Series | pd.DataFrame],
    keep: DropKeep = 'last',
) -> pd.DataFrame | pd.Series:
    """Concatenates pandas objects and drops duplicate indices.

    Args:
        objs: An iterable of pandas Series or DataFrames to concatenate
        keep: How to handle duplicate indices:
            * 'first': Keep first occurrence of duplicated index
            * 'last': Keep last occurrence of duplicated index
            * False: Drop all duplicates including first/last occurrence

    Returns:
        A concatenated pandas DataFrame or Series with duplicate indices removed

    Example:
        >>> df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
        >>> df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 2])
        >>> concat_drop_dup([df1, df2], keep='last')
           A
        0  1
        1  3
        2  4
    """
    ret = pd.concat(objs)
    return ret.loc[~ret.index.duplicated(keep)]

@torch.no_grad()
def compute_grad_norm(m: nn.Module):
    norm = 0.
    for name, p in m.named_parameters():
        if p.grad is not None:
            grad = p.grad.flatten()
            norm += torch.dot(grad, grad)
    return norm ** 0.5

def partition_by_predicate(pred: Callable[[T], bool] | Hashable, seq: Iterable[T]) -> tuple[list[T], list[T]]:
    """Partitions a sequence into two lists based on a predicate.

    Args:
        pred: A callable that takes an element of the sequence and returns a boolean,
            or a hashable object used for grouping
        seq: An iterable sequence of elements to be partitioned

    Returns:
        A tuple of two lists:
        - First list contains elements for which the predicate returns False
        - Second list contains elements for which the predicate returns True
    """
    groups: dict[bool, list] = cytoolz.groupby(pred, seq)
    assert set(groups.keys()).issubset({False, True})
    return groups.get(False, []), groups.get(True, [])

def import_object(name: str):
    """Imports an object from a module using its fully qualified name.

    The implementation is copied from :func:`jsonargparse._util.import_object` (v4.35.0)

    Args:
        name: A dot-separated string representing the full import path of the object.
            For example: ``os.path.join`` or ``torch.nn.Linear``

    Returns:
        The imported object

    Raises:
        ValueError: If the name is not a valid dot-separated import path string
        ModuleNotFoundError: If the module cannot be found
        AttributeError: If the object cannot be found in the module
    """
    if not isinstance(name, str) or "." not in name:
        raise ValueError(f"Expected a dot import path string: {name}")
    if not all(x.isidentifier() for x in name.split(".")):
        raise ValueError(f"Unexpected import path format: {name}")
    name_module, name_object = name.rsplit(".", 1)
    try:
        parent = __import__(name_module, fromlist=[name_object])
    except ModuleNotFoundError as ex:
        if "." not in name_module:
            raise ex
        name_module, name_object1 = name_module.rsplit(".", 1)
        parent = getattr(__import__(name_module, fromlist=[name_object1]), name_object1)
    return getattr(parent, name_object)
