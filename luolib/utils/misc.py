from typing import TypeVar

__all__ = [
    'fall_back_none',
]

T = TypeVar('T')
U = TypeVar('U')

def fall_back_none(x: T | None, default: U) -> T | U:
    return default if x is None else x
