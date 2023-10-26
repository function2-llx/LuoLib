import torch
from torch import nn

from ..backbones import BackboneProtocol

class NestedBackbone(nn.Module):
    inner: BackboneProtocol

    def __init__(self, *, inner: nn.Module):
        super().__init__()
        assert isinstance(inner, BackboneProtocol)
        self.inner = inner

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.process(self.inner(x), x)

    def process(self, feature_maps: list[torch.Tensor], x: torch.Tensor) -> list[torch.Tensor]:
        """
        Process the feature maps output by the inner backbone and produce feature maps processed by this module
        Args:
            feature_maps: feature maps (output by some backbone), high → low resolution
            x: original input image
        Returns:
            list of feature maps, high → low resolution
        """
        raise NotImplementedError
