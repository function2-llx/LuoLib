import itertools as it

import torch

from monai import transforms as mt
from monai.config import NdarrayOrTensor
from monai.data import get_track_meta
from monai.utils import convert_to_tensor

from luolib.types import tuple2_t

__all__ = [
    'RandAdjustContrast',
]

class RandAdjustContrast(mt.RandomizableTransform):
    def __init__(
        self,
        prob: float,
        contrast_range: tuple2_t[float],
        preserve_intensity_range: bool = True,
    ):
        mt.RandomizableTransform.__init__(self, prob)
        self.contrast_range = contrast_range
        self.preserve_intensity_range = preserve_intensity_range

    def randomize(self, num_channels: int):
        super().randomize(None)
        if not self._do_transform:
            return
        self.factor = self.R.uniform(*self.contrast_range, (num_channels, 1))

    def __call__(self, img_in: NdarrayOrTensor, randomize: bool = True):
        img: torch.Tensor = convert_to_tensor(img_in, track_meta=get_track_meta())
        num_channels = img.shape[0]
        if randomize:
            self.randomize(num_channels)
        if not self._do_transform:
            return img
        spatial_size = img.shape[1:]
        img = img.view(num_channels, -1)
        factor = img.new_tensor(self.factor)
        min_v = img.min(1, True)
        max_v = img.max(1, True)
        mean = img.mean(1, True)
        img.mul_(factor).add_(mean, alpha=1 - factor)
        if self.preserve_intensity_range:
            img.clamp_(min_v, max_v)
        return img.view(num_channels, *spatial_size)
