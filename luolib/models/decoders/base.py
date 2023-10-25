import torch
from torch import nn

from ..backbones import BackboneProtocol

class BackboneWithDecoder(nn.Module):
    backbone: BackboneProtocol

    def __init__(self, *, backbone: nn.Module):
        super().__init__()
        assert isinstance(backbone, BackboneProtocol)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.decode(self.backbone(x), x)

    def decode(self, feature_maps: list[torch.Tensor], x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            feature_maps: feature maps (output by some backbone), high → low resolution
            x: original input image
        Returns:
            list of feature maps, high → low resolution
        """
        raise NotImplementedError
