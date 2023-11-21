from collections.abc import Hashable
from pathlib import Path

import numpy as np
import pandas as pd

from monai import transforms as mt
from monai.data import NumpyReader

__all__ = [
    'RandomizableLoadImage',
    'RandomizableLoadImageD',
    'nnUNetLoader',
    'nnUNetLoaderD',
]

class RandomizableLoadImage(mt.Randomizable, mt.LoadImage):
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        for reader in self.readers:
            if isinstance(reader, mt.Randomizable):
                reader.set_random_state(seed, state)
        return self

class RandomizableLoadImageD(mt.Randomizable, mt.LoadImageD):
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        RandomizableLoadImage.set_random_state(self._loader, seed, state)

class nnUNetLoader(mt.Transform):
    def __init__(self, data_dir: Path | None = None, img_key: Hashable = 'img', seg_key: Hashable | None = 'seg'):
        self.data_dir = data_dir
        self.img_key = img_key
        self.seg_key = seg_key
        keys = [img_key]
        if seg_key is not None:
            keys.append(seg_key)
        self.img_loader = mt.LoadImageD(
            keys,
            reader=NumpyReader,
            dtype=None,
            ensure_channel_first=False,
        )

    def __call__(self, key: str, data_dir: Path | None = None):
        data_dir = self.data_dir if data_dir is None else data_dir
        path_data = {self.img_key: data_dir / f'{key}.npy'}
        if self.seg_key is not None:
            path_data[self.seg_key] = data_dir / f'{key}_seg.npy'
        img_data = self.img_loader(path_data)
        meta = pd.read_pickle(data_dir / f'{key}.pkl')
        return {**img_data, **meta, 'path_base': str(data_dir / key)}

class nnUNetLoaderD(mt.Transform):
    def __init__(self, key: Hashable, data_dir: Path | None = None, img_key: Hashable = 'img', seg_key: Hashable | None = 'seg'):
        self.key = key
        self.loader = nnUNetLoader(data_dir, img_key, seg_key)

    def __call__(self, data: dict, data_dir: Path | None = None):
        data = dict(data)
        return {
            **data,
            **self.loader(data[self.key], data_dir),
        }
