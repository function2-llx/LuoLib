from collections.abc import Sequence

import cytoolz
import torch
from torch import nn

from luolib.lightning import LightningModule
from .backbones import BackboneProtocol
from .masked_attention_decoder import MaskedAttentionDecoder

__all__ = [
    'MaskFormer',
]

class MaskFormer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        mask_decoder: MaskedAttentionDecoder,
        key_levels: Sequence[int] = (-1, -2, -3),
        pixel_embedding_level: int = 0,
    ):
        super().__init__()
        assert isinstance(backbone, BackboneProtocol)
        self.backbone = backbone
        self.mask_decoder = mask_decoder
        self.key_levels = key_levels
        assert len(key_levels) == self.mask_decoder.num_feature_levels
        self.pixel_embedding_level = pixel_embedding_level

    def forward(self, x: torch.Tensor, manual_mask: torch.Tensor | None = None):
        feature_maps: list[torch.Tensor] = self.backbone(x)
        layers_mask_embeddings, layers_mask_logits = self.mask_decoder(
            cytoolz.get(self.key_levels, feature_maps),
            feature_maps[self.pixel_embedding_level],
            manual_mask,
        )
        return layers_mask_embeddings, layers_mask_logits
