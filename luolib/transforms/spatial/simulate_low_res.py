from collections.abc import Sequence

import numpy as np
import torch

from monai import transforms as mt

from luolib.data import track_meta

__all__ = [
    'RandSimulateLowResolution',
]

from monai.config import NdarrayOrTensor

from monai.utils import convert_to_tensor

class RandSimulateLowResolution(mt.RandomizableTransform):

    backend = mt.Affine.backend

    def __init__(
        self,
        prob: float = 0.25,
        prob_per_channel: float = 0.25,
        downsample_mode: str | int = 0,
        upsample_mode: str | int = 3,
        zoom_range: Sequence[float] = (0.5, 1.0),
    ):
        """
        Args:
            prob: probability of performing this augmentation
            downsample_mode: interpolation mode for downsampling operation
            upsample_mode: interpolation mode for upsampling operation
            zoom_range: range from which the random zoom factor for the downsampling and upsampling operation is
            sampled. It determines the shape of the downsampled tensor.
        """
        super().__init__(prob)

        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.zoom_range = zoom_range
        self.prob_per_channel = prob_per_channel

    def randomize(self, img: torch.Tensor) -> None:
        super().randomize(img)
        if not self._do_transform:
            return
        self.zoom_factor = [
            self.R.uniform(self.zoom_range[0], self.zoom_range[1]) if self.prob_per_channel < self.R.uniform()
            else None
        ]

    def __call__(self, img_in: NdarrayOrTensor, randomize: bool = True):
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            randomize: whether to execute `randomize()` function first, defaults to True.
        """
        img: torch.Tensor = convert_to_tensor(img_in)
        if randomize:
            self.randomize(img)

        if self._do_transform:
            input_shape = img.shape[1:]
            target_shape = np.round(np.array(input_shape) * self.zoom_factor).astype(np.int32)
            with track_meta(False):
                for ci, zoom_factor in enumerate(self.zoom_factor):
                    if zoom_factor is None:
                        continue
                    downsample = mt.Affine(
                        scale_params=np.full(len(input_shape), zoom_factor),
                        spatial_size=target_shape,
                        mode=self.downsample_mode,
                        image_only=True,
                    )
                    upsample = mt.Affine(
                        scale_params=np.full(len(input_shape), 1 / zoom_factor),
                        spatial_size=input_shape,
                        mode=self.upsample_mode,
                        image_only=True,
                    )
                    # temporarily disable metadata tracking, since we do not want to invert the two Resize functions during
                    # post-processing
                    with track_meta(False):
                        img_downsampled = downsample(img[ci:ci + 1])
                        img[ci:ci + 1] = upsample(img_downsampled)
        return img
