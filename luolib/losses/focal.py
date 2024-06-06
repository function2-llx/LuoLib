import torch

from monai.losses.focal_loss import sigmoid_focal_loss as _monai_sigmoid_focal_loss

__all__ = [
    'sigmoid_focal_loss',
]

def sigmoid_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    return _monai_sigmoid_focal_loss(input, target.float(), gamma, alpha)
