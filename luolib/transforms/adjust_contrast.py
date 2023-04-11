from typing import Hashable, Mapping

import torch

from monai import transforms as monai_t
from monai.config import KeysCollection

class RandAdjustContrastD(monai_t.RandomizableTransform, monai_t.MapTransform):
    def __init__(self, keys: KeysCollection, contrast_range: tuple[float, float], prob: float, preserve_range: bool = True, allow_missing: bool = False):
        monai_t.RandomizableTransform.__init__(self, prob)
        monai_t.MapTransform.__init__(self, keys, allow_missing)
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        self.randomize(None)
        if not self._do_transform:
            return data
        factor = self.R.uniform(*self.contrast_range)
        d = dict(data)
        sample_x = d[self.first_key(d)]
        spatial_dims = sample_x.ndim - 1
        reduce_dims = tuple(range(1, spatial_dims + 1))
        for key in self.key_iterator(d):
            x = d[key]
            # mean = einops.reduce(x, 'c ... -> c', 'mean')
            mean = x.mean(dim=reduce_dims, keepdim=True)
            if self.preserve_range:
                min_v = x.amin(dim=reduce_dims, keepdim=True)
                max_v = x.amax(dim=reduce_dims, keepdim=True)
            x = x * factor + mean * (1 - factor)
            if self.preserve_range:
                x.clamp_(min_v, max_v)

            d[key] = x
        return d
