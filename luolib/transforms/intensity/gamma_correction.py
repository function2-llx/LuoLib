# Fix name for MONAI: https://github.com/Project-MONAI/MONAI/discussions/6027
from collections.abc import Hashable, Mapping

import torch

from monai import transforms as mt
from monai.config import KeysCollection, NdarrayOrTensor
from monai.utils import TransformBackends

def gamma_correction(img: torch.Tensor, gamma: float, invert: bool = False):
    if invert:
        img = 1 - img
    ret = img ** gamma
    if invert:
        ret = 1 - ret
    return ret

class RandGammaCorrectionD(mt.RandomizableTransform, mt.MapTransform):
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        gamma: tuple[float, float] | float = (0.5, 4.5),
        invert_image: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        mt.RandomizableTransform.__init__(self, prob)
        mt.MapTransform.__init__(self, keys, allow_missing_keys)
        self.gamma = gamma
        self.gamma_value = None
        self.invert_image = invert_image

    def randomize(self, _) -> None:
        super().randomize(None)
        if self._do_transform:
            self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        data = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(data):
                data[key] = gamma_correction(data[key], self.gamma_value, self.invert_image)
        return data
