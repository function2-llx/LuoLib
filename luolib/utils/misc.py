from typing import TypeVar

from luolib.utils import T

U = TypeVar('U')

def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x
