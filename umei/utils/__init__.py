from collections.abc import Hashable
import operator
import os
from typing import Callable, Iterable, Union

import cytoolz
from einops import rearrange
from einops.layers.torch import Rearrange
import torch

from .argparse import UMeIParser
from .enums import DataSplit, DataKey

PathLike = Union[str, bytes, os.PathLike]

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

def partition_by_predicate(pred: Callable | Hashable, seq: Iterable):
    groups = cytoolz.groupby(pred, seq)
    return tuple(groups.get(k, []) for k in [False, True])

class SimpleReprMixin(object):
    """A mixin implementing a simple __repr__."""
    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )
