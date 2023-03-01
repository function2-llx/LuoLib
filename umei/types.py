from typing import TypeVar

T = TypeVar('T')
tuple2_t = tuple[T, T]
scalar_tuple2_t = T | tuple2_t[T]
tuple3_t = tuple[T, T, T]

def check_tuple(obj, n: int, t: type):
    if not isinstance(obj, tuple):
        return False
    if len(obj) != n:
        return False
    return all(isinstance(x, t) for x in obj)
