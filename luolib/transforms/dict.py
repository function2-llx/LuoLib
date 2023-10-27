from collections.abc import Callable, Hashable, Mapping
import inspect
from typing import Protocol
import warnings

import numpy as np

from monai import transforms as mt
from monai.config import KeysCollection
from monai.transforms import Randomizable

__all__ = [
    'DictWrapper',
    'RandDictWrapper',
    'RandUniformDictWrapper',
]

from monai.utils import convert_to_tensor

class RandomizableTransformProtocol(Protocol):
    def __call__(self, data, randomize: bool, *args, **kwargs) -> ...:
        ...

class DictWrapper(mt.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        trans: Callable[..., ...],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.trans = trans

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        data = dict(data)
        for key in self.key_iterator(data):
            data[key] = self.trans(data[key], *args, **kwargs)
        return data

class RandDictWrapper(DictWrapper, mt.Randomizable):
    def __init__(
        self,
        keys: KeysCollection,
        trans: RandomizableTransformProtocol,
        allow_missing_keys: bool = False,
    ):
        DictWrapper.__init__(self, keys, trans, allow_missing_keys)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        if isinstance(self.trans, mt.Randomizable):
            self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data=None):
        pass  # do nothing, each key will randomize independently

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        return super().__call__(data, *args, **kwargs, randomize=True)

class RandUniformDictWrapper(DictWrapper, mt.RandomizableTransform):
    def __init__(
        self,
        keys: KeysCollection,
        prob: float,
        trans: RandomizableTransformProtocol,
        allow_missing_keys: bool = False
    ):
        DictWrapper.__init__(self, keys, trans, allow_missing_keys)
        mt.RandomizableTransform.__init__(self, prob)
        if isinstance(trans, mt.RandomizableTransform):
            if trans.prob != 1:
                warnings.warn(
                    'uniform dict transform should handle transform probability at dict-level'
                    'fixed by setting trans.prob=1',
                    stacklevel=2,
                )
                trans.prob = 1

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        super().set_random_state(seed, state)
        if isinstance(self.trans, mt.Randomizable):
            self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: dict):
        if isinstance(self.trans, mt.Randomizable):
            self.trans.randomize(data[self.first_key(data)])

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        data = dict(data)
        self.randomize(data)
        if self._do_transform:
            data = super()(data, *args, **kwargs, randomize=False)
        else:
            for key in self.key_iterator(data):
                data[key] = convert_to_tensor(data[key])
        return data
