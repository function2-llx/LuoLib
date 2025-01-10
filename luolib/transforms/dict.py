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

class DictWrapper(mt.MapTransform):
    """Wraps a transform to handle dictionary data, applies the given transform to specified keys in a dictionary input.
    
    Args:
        keys: Keys to pick data from dictionary for transformation.
        trans: Transform to be applied on dictionary values.
    """
    def __init__(
        self,
        keys: KeysCollection,
        trans: Callable,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.trans = trans

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        """
        Args:
            data: Dictionary containing data to transform.
            *args: Additional positional arguments to pass to transform.
            **kwargs: Additional keyword arguments to pass to transform.
            
        Returns:
            Transformed dictionary data.
        """
        data = dict(data)
        for key in self.key_iterator(data):
            data[key] = self.trans(data[key], *args, **kwargs)
        return data

class RandDictWrapper(DictWrapper, mt.Randomizable):
    """Wraps a randomizable transform to handle dictionary data.
    
    This transform applies the given random transform to specified keys in a dictionary input.
    Each key's data is transformed independently with its own random parameters.
    
    Args:
        keys: Keys to pick data from dictionary for transformation.
        trans: Random transform to be applied on dictionary values.
    """
    def __init__(
        self,
        keys: KeysCollection,
        trans: Callable,
        allow_missing_keys: bool = False,
    ):
        DictWrapper.__init__(self, keys, trans, allow_missing_keys)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        if isinstance(self.trans, mt.Randomizable):
            self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data=None):
        pass  # do nothing, will randomize for each key independently

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        return super().__call__(data, *args, **kwargs, randomize=True)

class RandUniformDictWrapper(DictWrapper, mt.RandomizableTransform):
    """Wraps a randomizable transform with uniform probability across keys.
    
    This transform applies the given random transform to all specified keys with the same
    random parameters and probability. If transform is applied, it's applied to all keys.
    
    Args:
        keys: Keys to pick data from dictionary for transformation.
        prob: Probability of applying the transform.
        trans: Random transform to be applied on dictionary values.
        allow_missing_keys: Don't raise exception if key is missing.
    """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float,
        trans: Callable,
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
            # Initialize random state with first key's data
            self.trans.randomize(data[self.first_key(data)])

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        data = dict(data)
        self.randomize(data)
        if self._do_transform:
            data = super().__call__(data, *args, **kwargs, randomize=False)
        else:
            for key in self.key_iterator(data):
                data[key] = convert_to_tensor(data[key])
        return data
