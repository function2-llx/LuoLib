from collections.abc import Callable, Hashable, Iterable, Sequence
import os
from typing import TypeVar

import cytoolz
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
import torch

from .enums import DataSplit, DataKey
from .index_tracker import IndexTracker
from .device_map import *
from .process_map import *

PathLike = str | bytes | os.PathLike

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

T = TypeVar('T')
def partition_by_predicate(pred: Callable[[T], bool] | Hashable, seq: Iterable[T]) -> tuple[list[T], list[T]]:
    groups: dict[bool, list] = cytoolz.groupby(pred, seq)
    assert set(groups.keys()).issubset({False, True})
    return groups.get(False, []), groups.get(True, [])

class SimpleReprMixin(object):
    """A mixin implementing a simple __repr__."""
    def __repr__(self):
        return "<{cls} @{id:x} {attrs}>".format(
            cls=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

U = TypeVar('U')
def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x
