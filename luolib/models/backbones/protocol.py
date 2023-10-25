from typing import Protocol, runtime_checkable

import torch

__all__ = [
    'BackboneProtocol',
]

@runtime_checkable
class BackboneProtocol(Protocol):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: input image
        Returns:
            list of feature maps, high → low resolution
        """
        ...
