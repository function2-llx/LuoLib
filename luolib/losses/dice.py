import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as nnf

from monai.networks import one_hot
from .focal import bce_with_binary_label, sigmoid_focal_loss

__all__ = [
    'DiceFocalLoss',
]

_EPS = 1e-8

def dice(prob: torch.Tensor, target: torch.Tensor | None):
    """
    copy from monai.losses.DiceLoss.forward, but fix the smooth issue, fix: https://github.com/MIC-DKFZ/nnUNet/issues/812
    """
    if target is None:
        return prob.new_ones(prob.shape[:2])

    if target.shape[1] == 1:
        target = one_hot(target, prob.shape[1])
    intersection = einops.reduce(target * prob, 'n c ... -> n c', 'sum')
    ground_o = einops.reduce(target, 'n c ... -> n c ', 'sum')
    pred_o = einops.reduce(prob, 'n c ... -> n c', 'sum')
    denominator = ground_o + pred_o
    # NOTE: no smooth item for nominator, or it will become unfortunate
    f: torch.Tensor = 1.0 - 2.0 * intersection / torch.clip(denominator, min=_EPS)
    return f

class DiceFocalLoss(nn.Module):
    """
    fix smooth issue of dice
    """
    def __init__(
        self,
        *,
        dice_weight: float,
        focal_binary: bool = False,
        focal_weight: float,
        focal_gamma: float,
        focal_alpha: float | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_binary = focal_binary
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        assert focal_gamma >= 0
        self.focal_weight = focal_weight

    def focal(self, input: torch.Tensor, target: torch.Tensor | None):
        # let's be happy
        if self.focal_gamma < _EPS:
            if self.focal_binary:
                return bce_with_binary_label(input, target)
            else:
                return nnf.cross_entropy(input, target[:, 0].long(), reduction='none')
        else:
            if target is None:
                target = torch.zeros_like(input)
            if not self.focal_binary:
                raise NotImplementedError
            return sigmoid_focal_loss(input, target, self.focal_gamma, self.focal_alpha)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.BoolTensor | None = None,
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        assert input.ndim == 5 and target.ndim == 5
        if target is not None:
            assert input.shape[0] == target.shape[0]
            assert input.shape[2:] == target.shape[2:]
            if self.focal_binary:
                assert input.shape[1] == target.shape[1]
            else:
                assert target.shape[1] == 1
        input = input.float()
        prob = input.sigmoid() if self.focal_binary else input.softmax(dim=1)
        dice_loss = dice(prob, target)
        focal_loss = self.focal(input, target)
        dice_loss = dice_loss.mean()
        focal_loss = focal_loss.mean()
        total_loss: torch.Tensor = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        if return_dict:
            focal_key = 'ce' if self.focal_gamma < _EPS else f'focal-{self.focal_gamma:.1f}'
            return {
                'dice': dice_loss,
                focal_key: focal_loss,
                'total': total_loss,
            }
        else:
            return total_loss
