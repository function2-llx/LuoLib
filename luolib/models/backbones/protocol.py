from typing import Protocol, runtime_checkable

import torch

__all__ = [
    'BackboneProtocol',
    # 'FeatureMapProcessorProtocol',
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

# @runtime_checkable
# class FeatureMapProcessorProtocol(Protocol):
#     def forward(self, feature_maps: list[torch.Tensor], x: torch.Tensor) -> list[torch.Tensor]:
#         """
#         Args:
#             feature_maps: feature maps extracted by previous backbone, high → low resolution
#             x: input image
#         Returns:
#             list of feature maps, high → low resolution
#         """
#         ...
#
