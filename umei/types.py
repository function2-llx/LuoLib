from collections.abc import Iterable, Sequence
from typing import TypeVar, TypedDict

from torch import nn

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

class ParameterGroup(TypedDict):
    params: Iterable[nn.Parameter]
    param_names: Iterable[str]
