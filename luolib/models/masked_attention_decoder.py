import random
from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from monai.networks.blocks import MLPBlock

from luolib.types import NoWeightDecayParameter, spatial_shape_t
from luolib.utils import fall_back_none, flatten
from .blocks import (
    transformer_block_forward, MemoryEfficientAttention, SpatialSinusoidalPositionEmbedding, with_pos_embed,
)
from .init import init_common

__all__ = [
    'MaskedAttentionDecoder',
]

class MaskedAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attention_heads: int,
        dim_feedforward: int,
        pre_norm: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        ffn_drop: float = 0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.cross_attn = MemoryEfficientAttention(
            embed_dim, num_attention_heads, True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        # set qkv_bias=True following the reference implementation
        self.self_attn = MemoryEfficientAttention(
            embed_dim, num_attention_heads, True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embed_dim, dim_feedforward, ffn_drop)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @property
    def post_norm(self):
        return not self.pre_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        key_position_embeddings: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ):
        hidden_states = transformer_block_forward(
            hidden_states, self.cross_attn, self.cross_attn_layer_norm, self.pre_norm,
            kwargs=dict(
                key=with_pos_embed(key_hidden_states, key_position_embeddings),
                value=key_hidden_states,
                attn_bias=attn_bias,
            ),
        )
        hidden_states = transformer_block_forward(
            hidden_states, self.self_attn, self.self_attn_layer_norm, self.pre_norm,
        )
        hidden_states = transformer_block_forward(hidden_states, self.mlp, self.final_layer_norm, self.pre_norm)

        return hidden_states

class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim: int, pixel_embedding_dim: int, norm: bool = False):
        """
        Args:
            norm: usually used for pre-norm
        """
        super().__init__()
        self.pixel_embedding_dim = pixel_embedding_dim
        self.mask_query_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pixel_embedding_dim + 1),
        )
        if norm:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, pixel_embedding: torch.Tensor) -> torch.Tensor:
        wandb = self.mask_query_projection(self.norm(hidden_states))
        weight = wandb[..., :self.pixel_embedding_dim]
        bias = einops.rearrange(
            wandb[..., self.pixel_embedding_dim],
            f"... -> ... {' '.join('1' * (pixel_embedding.ndim - 2))}",
        )
        mask_logits = einops.einsum(weight, pixel_embedding, 'n nq c, n c ... -> n nq ...') + bias
        return mask_logits

# modified from transformers.models.mask2former.modeling_mask2former.Mask2FormerMaskedAttentionDecoder
class MaskedAttentionDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        feature_channels: int,
        num_attention_heads: int,
        pixel_embedding_dim: int,
        dim_feedforward: int | None = None,
        num_decoder_layers: int = 9,
        mask_start_layer: int = 3,
        num_feature_levels: int = 3,
        num_queries: int = 100,
        pre_norm: bool = True,
        layer_drop: float = 0.,
        grad_ckpt: bool = False,
    ):
        """
        Args:
            mask_start_layer: the layer from which to start restricting the cross-attention area using predicted mask.
        """
        super().__init__()
        # attention mask interpolation
        self.interpolate_mode = 'bilinear' if spatial_dims == 2 else 'trilinear'
        self.hidden_dim = hidden_dim = feature_channels
        self.num_attention_heads = num_attention_heads
        dim_feedforward = fall_back_none(dim_feedforward, feature_channels * 4)

        self.num_feature_levels = num_feature_levels
        self.key_position_embedding = SpatialSinusoidalPositionEmbedding(feature_channels, normalize=True, flatten=True)
        self.level_embedding = NoWeightDecayParameter(torch.empty(num_feature_levels, hidden_dim))
        self.query_embedding = NoWeightDecayParameter(torch.empty(num_queries, hidden_dim))

        self.layers: Sequence[MaskedAttentionDecoderLayer] | nn.ModuleList = nn.ModuleList([
            MaskedAttentionDecoderLayer(feature_channels, num_attention_heads, dim_feedforward, pre_norm)
            for _ in range(num_decoder_layers)
        ])
        self.layer_drop = layer_drop

        self.mask_start_layer = mask_start_layer
        self.mask_predictor = MaskPredictor(hidden_dim, pixel_embedding_dim, pre_norm)

        self.gradient_checkpointing = grad_ckpt

        # follows nn.Embedding
        nn.init.normal_(self.query_embedding)
        nn.init.normal_(self.level_embedding)
        self.apply(init_common)

    @torch.no_grad()
    def get_attn_bias(
        self,
        target_shape: spatial_shape_t,
        mask_logits: torch.Tensor,
        manual_mask: torch.Tensor | None = None,
    ):
        if manual_mask is None:
            mask_prob = nnf.interpolate(mask_logits, target_shape, mode=self.interpolate_mode).sigmoid()
        else:
            mask_prob = nnf.interpolate(manual_mask.float(), target_shape, mode=self.interpolate_mode)
        ignore_mask = einops.rearrange(mask_prob < 0.5, 'n nq ... -> n nq (...)')
        # no restriction for empty mask
        ignore_mask.masked_fill_(ignore_mask.all(dim=-1, keepdim=True), False)
        attn_bias = torch.where(ignore_mask, -torch.inf, 0.)
        return einops.repeat(attn_bias, 'n nq nk -> n M nq nk', M=self.num_attention_heads)

    def forward(
        self,
        key_feature_maps: list[torch.Tensor],
        pixel_embedding: torch.Tensor,
        manual_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            key_feature_maps: list of (n, c, *spatial), c is identical across all levels
                resolution: low → high (Mask2Former)
            pixel_embedding: feature map to produce the segmentation output
            manual_mask: restrict the cross-attention area with the specified mask
        Returns:
            (mask_embedding, mask_logits) of all layers
        """
        batch_size = pixel_embedding.shape[0]
        key_hidden_states = list(map(flatten, key_feature_maps))
        key_position_embeddings = [
            self.key_position_embedding(x.shape[2:]) + self.level_embedding[i]
            for i, x in enumerate(key_feature_maps)
        ]

        hidden_states = einops.repeat(self.query_embedding, '... -> n ...', n=batch_size)
        layers_mask_embeddings = [hidden_states]
        mask_logits = self.mask_predictor(hidden_states, pixel_embedding)
        layers_mask_logits = [mask_logits]

        for idx, layer in enumerate(self.layers):
            if self.training and (random.uniform(0, 1) < self.layer_drop):
                continue
            level_index = idx % self.num_feature_levels
            if manual_mask is not None or idx >= self.mask_start_layer:
                attn_bias = self.get_attn_bias(key_feature_maps[level_index].shape[2:], mask_logits, manual_mask)
            else:
                attn_bias = None
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint.checkpoint(
                    layer, hidden_states,
                    key_hidden_states[level_index], key_position_embeddings[level_index], attn_bias,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    key_hidden_states[level_index], key_position_embeddings[level_index], attn_bias,
                )

            layers_mask_embeddings.append(hidden_states)
            mask_logits = self.mask_predictor(hidden_states, pixel_embedding)
            layers_mask_logits.append(mask_logits)

        return layers_mask_embeddings, layers_mask_logits
