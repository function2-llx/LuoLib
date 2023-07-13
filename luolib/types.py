from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, TypedDict, NamedTuple, Generic

from torch import nn
from lightning.pytorch.utilities.types import LRSchedulerConfig as LRSchedulerConfigBase

T = TypeVar('T')
tuple2_t = tuple[T, T]
param2_t = T | tuple2_t[T]
tuple3_t = tuple[T, T, T]
param3_t = T | tuple3_t[T]
spatial_param_t = T | tuple2_t[T] | tuple3_t[T]
spatial_param_seq_t = Sequence[param2_t[T]] | Sequence[param3_t[T]]

def check_tuple(obj, n: int, t: type):
    if not isinstance(obj, tuple):
        return False
    if len(obj) != n:
        return False
    return all(isinstance(x, t) for x in obj)


# set total=False to comfort IDE
class ParamGroup(TypedDict, total=False):
    params: list[nn.Parameter] | None
    names: list[str]
    lr_scale: float  # inserted by timm
    lr: float
    weight_decay: float

@dataclass
class LRSchedulerConfig(LRSchedulerConfigBase):
    scheduler: dict  # bypass jsonargparse check

Tr = TypeVar('Tr', bound=int | float)
class RangeTuple(NamedTuple, Generic[Tr]):
    min: Tr
    max: Tr
