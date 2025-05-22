from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
from typing import TypeAlias, TypeVar

from torch import nn

from monai.networks.layers import Conv

type tuple2_t[T] = tuple[T, T]
type param2_t[T] = T | tuple2_t[T]
type tuple3_t[T] = tuple[T, T, T]
type param3_t[T] = T | tuple3_t[T]
type spatial_param_t[T] = T | tuple2_t[T] | tuple3_t[T]
type spatial_param_seq_t[T] = Sequence[param2_t[T]] | Sequence[param3_t[T]]
type maybe_seq_t[T] = T | Sequence[T]

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

type partial_t[F] = type[F] | tuple[type[F], dict]

def call_partial[F](partial: partial_t[F], *args, **kwargs):
    if not isinstance(partial, tuple):
        partial = (partial, {})
    return partial[0](*args, **partial[1], **kwargs)

def get_conv_t(spatial_dims) -> type[nn.Conv2d | nn.Conv3d]:
    assert spatial_dims != 1
    return Conv[Conv.CONV, spatial_dims]

type spatial_shape_t = tuple2_t[int] | tuple3_t[int]

type named_param_t = tuple[str, nn.Parameter]
type PathLike = str | bytes | os.PathLike
