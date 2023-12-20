# MONAI seems to have an unconventional name for gamma correction
# https://github.com/Project-MONAI/MONAI/discussions/6027
# ref: batchgenerators
import einops
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
        rescale: bool,
        eps: float = 1e-7,
    ):
        """
        Args:
            prob:
            gamma_range:
            prob_invert:
            retain_stats:
            rescale: whether to rescale the intensity to [0, 1], only use this if the intensity is between [0, 1]
            eps:
        """
        mt.RandomizableTransform.__init__(self, prob)
        self.gamma_range = gamma_range
        self.prob_invert = prob_invert
        self.retain_stats = retain_stats
        self.rescale = rescale
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

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True):
        img_t: torch.Tensor = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize(img_t.shape[0])
        if not self._do_transform:
            return img_t
        spatial_shape = img_t.shape[1:]
        img_t = einops.rearrange(img_t, 'c ... -> c (...)')
        if self.retain_stats:
            mean = img_t.mean(1, True)
            std = img_t.std(1, keepdim=True, correction=0)
        if self.rescale:
            min_v = img_t.amin(1, True)
            range_v = img_t.amax(1, True) - min_v + self.eps
            img_t = (img_t - min_v) / range_v
        if self.invert:
            img_t = 1 - img_t
        img_t = img_t.pow(img_t.new_tensor(self.gamma))
        if self.invert:
            img_t = 1 - img_t
        if self.rescale:
            img_t = img_t * range_v + min_v
        if self.retain_stats:
            img_t = (img_t - img_t.mean(1, True)) / (img_t.std(keepdim=True, correction=0) + 1e-8)
            img_t = img_t * std + mean
        return img_t.view(img_t.shape[0], *spatial_shape)
