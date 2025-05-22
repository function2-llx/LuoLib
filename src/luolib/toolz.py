from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING

# vscode does not support .pyx very well
if TYPE_CHECKING:
    from toolz import *
else:
    from cytoolz import *

def partition_by_predicate[T](pred: Callable[[T], bool] | Hashable, seq: Iterable[T]) -> tuple[list[T], list[T]]:
    """Partitions a sequence into two lists based on a predicate.

    Args:
        pred: A callable that takes an element of the sequence and returns a boolean,
            or a hashable object used for grouping
        seq: An iterable sequence of elements to be partitioned

    Returns:
        A tuple of two lists:
        - First list contains elements for which the predicate returns False
        - Second list contains elements for which the predicate returns True
    """
    groups: dict[bool, list] = groupby(pred, seq)
    assert set(groups.keys()).issubset({False, True})
    return groups.get(False, []), groups.get(True, [])
