from collections.abc import Hashable
from pathlib import Path

import cytoolz
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
    def __init__(
        self,
        data_dir: Path | None = None,
        img_key: Hashable | None = 'img',
        seg_key: Hashable | None = 'seg',
        unravel_class_locations: bool = False,
        remove_label: bool = True,
        allow_missing: bool = False,
    ):
        self.data_dir = data_dir
        self.img_key = img_key
        self.seg_key = seg_key
        keys = [*filter(lambda key: key is not None, [img_key, seg_key])]
        self.unravel_class_locations = unravel_class_locations
        self.remove_label = remove_label
        self.img_loader = mt.LoadImageD(
            keys,
            reader=NumpyReader,
            dtype=None,
            ensure_channel_first=False,
            allow_missing_keys=allow_missing,
        )

    def __call__(self, key: str, data_dir: Path | None = None):
        data_dir = self.data_dir if data_dir is None else data_dir
        path_data = {}
        if self.img_key is not None and (img_path := data_dir / f'{key}.npy').exists():
            path_data[self.img_key] = img_path
        if self.seg_key is not None and (seg_path := data_dir / f'{key}_seg.npy').exists():
            path_data[self.seg_key] = seg_path
        img_data = self.img_loader(path_data)
        if self.remove_label and (seg := img_data.get('seg')) is not None:
            # https://github.com/MIC-DKFZ/nnUNet/blob/v2.3.1/nnunetv2/preprocessing/cropping/cropping.py#L43
            seg[seg < 0] = 0

        meta: dict = pd.read_pickle(data_dir / f'{key}.pkl')
        if self.unravel_class_locations:
            class_locations: dict = meta['class_locations']
            shape = cytoolz.first(img_data.values()).shape[1:]
            meta['class_locations'] = [
                np.ravel_multi_index(locations[:, 1:].T, shape)
                if len(locations := class_locations[k]) > 0 else np.array([])
                for k in sorted(class_locations.keys())
            ]

        return {**img_data, **meta, 'path_base': str(data_dir / key)}

class nnUNetLoaderD(mt.Transform):
    def __init__(
        self,
        key: Hashable,
        data_dir: Path | None = None,
        img_key: Hashable | None = 'img',
        seg_key: Hashable | None = 'seg',
        unravel_class_locations: bool = False,
        allow_missing: bool = False,
    ):
        """
        Args:
            key: the key to obtain nnU-Net case key
        """
        self.key = key
        self.loader = nnUNetLoader(
            data_dir=data_dir,
            img_key=img_key,
            seg_key=seg_key,
            unravel_class_locations=unravel_class_locations,
            allow_missing=allow_missing,
        )

    def __call__(self, data: dict, data_dir: Path | None = None):
        data = dict(data)
        return {
            **data,
            **self.loader(data[self.key], data_dir),
        }
