from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from monai.config import NdarrayOrTensor
from monai import transforms as mt
from monai.data import MetaTensor
from monai.transforms.lazy.functional import apply_pending_transforms

__all__ = ["OneOf", "execute_compose"]

def execute_compose(
    data: NdarrayOrTensor | Sequence[NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor],
    transforms: Sequence[Any],
    map_items: bool = True,
    unpack_items: bool = False,
    start: int = 0,
    end: int | None = None,
    lazy: bool | None = False,
    overrides: dict | None = None,
    threading: bool = False,
    log_stats: bool | str = False,
    apply_pending: bool = False,
) -> NdarrayOrTensor | Sequence[NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor]:
    end_ = len(transforms) if end is None else end
    if start is None:
        raise ValueError(f"'start' ({start}) cannot be None")
    if start < 0:
        raise ValueError(f"'start' ({start}) cannot be less than 0")
    if start > end_:
        raise ValueError(f"'start' ({start}) must be less than 'end' ({end_})")
    if end_ > len(transforms):
        raise ValueError(f"'end' ({end_}) must be less than or equal to the transform count ({len(transforms)}")

    # no-op if the range is empty
    if start == end:
        return data

    for _transform in transforms[start:end]:
        if threading:
            _transform = deepcopy(_transform) if isinstance(_transform, mt.ThreadUnsafe) else _transform
        data = mt.apply_transform(
            _transform, data, map_items, unpack_items, lazy=lazy, overrides=overrides, log_stats=log_stats
        )
    if apply_pending:
        data = apply_pending_transforms(data, None, overrides, logger_name=log_stats)
    return data

class OneOf(mt.OneOf):
    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        weights: Sequence[float] | float | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool | str = False,
        lazy: bool | None = False,
        overrides: dict | None = None,
        apply_pending: bool = False,
    ) -> None:
        super().__init__(transforms, weights, map_items, unpack_items, log_stats, lazy, overrides)
        self.apply_pending = apply_pending

    def __call__(self, data, start=0, end=None, threading=False, lazy: bool | None = None):
        if self.apply_pending:
            return super().__call__(data, start, end, threading, lazy)

        if start != 0:
            raise ValueError(f"OneOf requires 'start' parameter to be 0 (start set to {start})")
        if end is not None:
            raise ValueError(f"OneOf requires 'end' parameter to be None (end set to {end}")

        if len(self.transforms) == 0:
            return data

        index = self.R.multinomial(1, self.weights).argmax()
        _transform = self.transforms[index]
        _lazy = self._lazy if lazy is None else lazy

        data = execute_compose(
            data,
            [_transform],
            start=start,
            end=end,
            map_items=self.map_items,
            unpack_items=self.unpack_items,
            lazy=_lazy,
            overrides=self.overrides,
            threading=threading,
            log_stats=self.log_stats,
            apply_pending=False,
        )

        # if the data is a mapping (dictionary), append the OneOf transform to the end
        if isinstance(data, MetaTensor):
            self.push_transform(data, extra_info={"index": index})
        elif isinstance(data, Mapping):
            for key in data:  # dictionary not change size during iteration
                if isinstance(data[key], MetaTensor):
                    self.push_transform(data[key], extra_info={"index": index})
        return data
