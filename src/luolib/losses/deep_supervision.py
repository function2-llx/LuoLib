from functools import cache

import torch
from torch import nn
import torch.nn.functional as nnf

from monai.utils import InterpolateMode

class DeepSupervisionWrapper(nn.Module):
    """Wrapper for applying deep supervision loss to multi-scale predictions.
    
    This module computes weighted losses for predictions at different scales compared to 
    the ground truth label. The label is interpolated to match each prediction's spatial 
    dimensions. Losses at deeper layers are weighted less than those at shallower layers.

    Args:
        loss: Base loss module to compute loss at each scale
        label_interp_mode: Interpolation mode for resizing labels to match prediction scales, defaults to nearest-exact
    """

    def __init__(self, loss: nn.Module, label_interp_mode: InterpolateMode = InterpolateMode.NEAREST_EXACT):
        super().__init__()
        self.loss = loss
        self.mode = label_interp_mode

    @staticmethod
    def prepare_labels(label: torch.Tensor, spatial_shapes: tuple[torch.Size, ...], mode: InterpolateMode) -> list[torch.Tensor]:
        """Prepares labels by interpolating to match prediction spatial dimensions.
        
        Args:
            label: Input label tensor
            spatial_shapes: Tuple of spatial dimensions for each prediction scale
            mode: Interpolation mode to use when resizing labels
        
        Returns:
            List of label tensors interpolated to match each prediction scale
        """
        label_shape = label.shape[2:]
        return [
            nnf.interpolate(label.byte(), shape, mode=mode).to(dtype=label.dtype) if label_shape != shape else label
            for shape in spatial_shapes
        ]

    def forward(self, deep_logits: list[torch.Tensor], label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes weighted deep supervision loss across multiple scales.
        
        Args:
            deep_logits: List of prediction tensors at different scales, ordered from 
                shallowest to deepest
            label: Ground truth label tensor
        
        Returns:
            Tuple containing:
                - Weighted sum of losses across all scales
                - Individual loss values for each scale before weighting
        """
        spatial_shapes = tuple([logits.shape[2:] for logits in deep_logits])
        deep_labels = self.prepare_labels(label, spatial_shapes, self.mode)
        ds_losses = torch.stack([
            self.loss(logits, deep_label)
            for logits, deep_label in zip(deep_logits, deep_labels)
        ])
        weight = ds_losses.new_tensor([1 / (1 << i) for i in range(len(deep_logits))])
        weight /= weight.sum()
        ds_loss = torch.dot(weight, ds_losses)
        return ds_loss, ds_losses
