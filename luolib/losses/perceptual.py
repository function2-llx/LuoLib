from functools import partial

import torch

from monai.losses import PerceptualLoss
from monai.losses.perceptual import RadImageNetPerceptualSimilarity, normalize_tensor, spatial_average

__all__ = [
    'SlicePerceptualLoss',
]

class FixedRadImageNetPerceptualSimilarity(RadImageNetPerceptualSimilarity):
    # fix https://github.com/Project-MONAI/GenerativeModels/issues/450
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The input is supposed to have a range of [-1, 1] with RGB channels"""
        # Change order from 'RGB' to 'BGR'
        input = input[:, [2, 1, 0], ...]
        target = target[:, [2, 1, 0], ...]

        # Get model outputs
        outs_input = self.model(input)
        outs_target = self.model(target)

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        results: torch.Tensor = (feats_input - feats_target) ** 2
        results = spatial_average(results.sum(dim=1, keepdim=True), keepdim=True)

        return results

class SlicePerceptualLoss(PerceptualLoss):
    def __init__(self, *args, max_slices: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_fake_3d
        if isinstance(self.perceptual_function, RadImageNetPerceptualSimilarity):
            self.perceptual_function.forward = partial(FixedRadImageNetPerceptualSimilarity.forward, self.perceptual_function)
        self.max_slices = max_slices

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.max_slices >= input.shape[2]:
            self.fake_3d_ratio = 1.
        else:
            self.fake_3d_ratio = self.max_slices / input.shape[2] + 1e-8
        loss = self._calculate_axis_loss(input, target, spatial_axis=2)
        return loss
