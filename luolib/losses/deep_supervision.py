from functools import cache

import torch
from torch import nn
import torch.nn.functional as nnf

from monai.utils import InterpolateMode

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss: nn.Module, label_interp_mode: InterpolateMode = InterpolateMode.NEAREST_EXACT):
        super().__init__()
        self.loss = loss
        self.mode = label_interp_mode

    @staticmethod
    def prepare_labels(label: torch.Tensor, spatial_shapes: tuple[torch.Size, ...], mode: InterpolateMode) -> list[torch.Tensor]:
        label_shape = label.shape[2:]
        return [
            nnf.interpolate(label.byte(), shape, mode=mode).to(dtype=label.dtype) if label_shape != shape else label
            for shape in spatial_shapes
        ]

    def forward(self, deep_logits: list[torch.Tensor], label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
