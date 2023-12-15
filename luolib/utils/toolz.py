from typing import Callable, Hashable, Iterable, TypeVar

import cytoolz

__all__ = [
    'partition_by_predicate',
]

T = TypeVar('T')

def partition_by_predicate(pred: Callable[[T], bool] | Hashable, seq: Iterable[T]) -> tuple[list[T], list[T]]:
    groups: dict[bool, list] = cytoolz.groupby(pred, seq)
    assert set(groups.keys()).issubset({False, True})
    return groups.get(False, []), groups.get(True, [])
