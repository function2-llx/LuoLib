from collections.abc import Sequence

import cytoolz
import torch
from torch import nn

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
        pixel_embedding_levels: list[int],
        key_levels: Sequence[int] = (-1, -2, -3),
        **kwargs,
    ):
        """
        Args:
            pixel_embedding_levels： high → low res
            key_levels: low → high res
        """
        super().__init__(**kwargs)
        assert isinstance(backbone, BackboneProtocol)
        self.backbone = backbone
        self.mask_decoder = mask_decoder
        self.key_levels = key_levels
        assert len(key_levels) == self.mask_decoder.num_feature_levels
        self.pixel_embedding_levels = pixel_embedding_levels

    def forward(self, x: torch.Tensor, manual_mask: torch.Tensor | None = None):
        feature_maps: list[torch.Tensor] = self.backbone(x)
        layers_mask_embeddings, layers_mask_logits = self.mask_decoder(
            cytoolz.get(self.key_levels, feature_maps),
            cytoolz.get(self.pixel_embedding_levels, feature_maps),
            manual_mask,
        )
        return layers_mask_embeddings, layers_mask_logits
