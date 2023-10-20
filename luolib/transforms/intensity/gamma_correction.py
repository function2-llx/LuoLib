# MONAI seems to have an unconventional name for gamma correction
# https://github.com/Project-MONAI/MONAI/discussions/6027
# ref: batchgenerators

import numpy as np
import torch

from monai import transforms as mt
from monai.config import NdarrayOrTensor
from monai.data import get_track_meta
from monai.utils import convert_to_tensor

from luolib.types import tuple2_t

__all__ = [
    'RandGammaCorrection',
]

class RandGammaCorrection(mt.RandomizableTransform):
    def __init__(
        self,
        prob: float,
        gamma_range: tuple2_t[float],
        prob_invert: float,
        retain_stats: bool,
        eps: float = 1e-7,
    ) -> None:
        mt.RandomizableTransform.__init__(self, prob)
        self.gamma_range = gamma_range
        self.prob_invert = prob_invert
        self.retain_stats = retain_stats
        self.eps = eps

    def randomize(self, num_channels: int):
        super().randomize(None)
        if not self._do_transform:
            return
        self.gamma = np.empty((num_channels, 1))
        for i in range(num_channels):
            if self.gamma_range[0] < 1 and self.R.uniform() < 0.5:
                self.gamma[i] = self.R.uniform(self.gamma_range[0], 1)
            else:
                self.gamma[i] = self.R.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
        self.invert = self.R.uniform() < self.prob_invert

    def __call__(self, img_in: NdarrayOrTensor):
        img: torch.Tensor = convert_to_tensor(img_in, track_meta=get_track_meta())
        self.randomize(img.shape[0])
        if not self._do_transform:
            return img
        spatial_shape = img.shape[1:]
        img = img.view(img.shape[0], -1)
        if self.invert:
            img = -img
        if self.retain_stats:
            mean = img.mean(1, True)
            std = img.std(1, keepdim=True, correction=True)
        min_v = img.min(1, True)
        range_v = img.max(1, True) - min_v + self.eps
        img = ((img - min_v) / range_v).pow(self.gamma)
        if self.retain_stats:
            img = (img - img.mean(1, True)) / (img.std(1, True) + 1e-8)
            img = img * std + mean
        else:
            img = img * range_v + min_v
        if self.invert:
            img = -img
        return img.view(img.shape[0], *spatial_shape)
