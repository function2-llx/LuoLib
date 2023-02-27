from typing import TypeVar

T = TypeVar('T')
tuple2_t = tuple[T, T]
scalar_tuple2_t = T | tuple2_t[T]
tuple3_t = tuple[T, T, T]
