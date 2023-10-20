from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from torch import nn

T = TypeVar('T')
tuple2_t: TypeAlias = tuple[T, T]
param2_t: TypeAlias = T | tuple2_t[T]
tuple3_t: TypeAlias = tuple[T, T, T]
param3_t: TypeAlias = T | tuple3_t[T]
spatial_param_t: TypeAlias = T | tuple2_t[T] | tuple3_t[T]
spatial_param_seq_t: TypeAlias = Sequence[param2_t[T]] | Sequence[param3_t[T]]
maybe_seq_t: TypeAlias = T | Sequence[T]

def check_tuple(obj, n: int, t: type):
    if not isinstance(obj, tuple):
        return False
    if len(obj) != n:
        return False
    return all(isinstance(x, t) for x in obj)

@dataclass
class RangeTuple:
    min: float | int
    max: float | int

    def __iter__(self):
        yield self.min
        yield self.max

# explicitly indicate that a parameter requires no weight decay
class NoWeightDecayParameter(nn.Parameter):
    pass

F = TypeVar('F', bound=type)
partial_t: TypeAlias = type[F] | tuple[type[F], dict]

def call_partial(partial: partial_t[F], *args, **kwargs):
    if not isinstance(partial, tuple):
        partial = (partial, {})
    return partial[0](*args, **partial[1], **kwargs)
