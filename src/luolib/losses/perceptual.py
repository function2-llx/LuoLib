import torch

from monai.losses import PerceptualLoss

__all__ = [
    'SlicePerceptualLoss',
]

class SlicePerceptualLoss(PerceptualLoss):
    # TODO: intensity rescale
    def __init__(self, *args, max_slices: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_fake_3d
        self.max_slices = max_slices

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.max_slices >= input.shape[2]:
            self.fake_3d_ratio = 1.
        else:
            self.fake_3d_ratio = self.max_slices / input.shape[2] + 1e-8
        loss = self._calculate_axis_loss(input, target, spatial_axis=2)
        return loss
