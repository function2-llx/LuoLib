import itertools as it

import einops
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

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True):
        img_t: torch.Tensor = convert_to_tensor(img, track_meta=get_track_meta())
        num_channels = img_t.shape[0]
        if randomize:
            self.randomize(num_channels)
        if not self._do_transform:
            return img_t
        spatial_size = img_t.shape[1:]
        img_t = einops.rearrange(img_t, 'c ... -> c (...)')
        if self.preserve_intensity_range:
            min_v = img_t.amin(1, True)
            max_v = img_t.amax(1, True)
        mean = img_t.mean(1, True)
        factor = img_t.new_tensor(self.factor)
        ret = img_t * factor + mean * (1 - factor)
        if self.preserve_intensity_range:
            ret.clamp_(min_v, max_v)
        return ret.view(num_channels, *spatial_size)
