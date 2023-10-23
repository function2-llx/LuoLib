import random
from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from monai.networks.blocks import MLPBlock

from luolib.utils import fall_back_none, flatten
from ..blocks import MemoryEfficientAttention
from ..layers import SpatialPositionEmbedding
from ..init import init_common

def with_pe(x: torch.Tensor, pe: torch.Tensor | None):
    return x if pe is None else x + pe

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
            embed_dim, num_attention_heads, False,
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
        query_position_embeddings: torch.Tensor | None = None,
        key_position_embeddings: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ):
        # cross-attention
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states = self.cross_attn(
            query=with_pe(hidden_states, query_position_embeddings),
            key=with_pe(key_hidden_states, key_position_embeddings),
            value=key_hidden_states,
            attention_bias=attention_bias,
        )
        hidden_states = residual + hidden_states
        if self.post_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        # self-attention
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(with_pe(hidden_states, query_position_embeddings))
        hidden_states = residual + hidden_states
        if self.post_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # fully connected
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if self.post_norm:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim: int, pixel_embedding_dim: int, norm: bool = False):
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

    def forward(self, hidden_states: torch.Tensor, pixel_embedding: torch.Tensor):
        wandb = self.mask_query_projection(self.norm(hidden_states))
        weight = wandb[..., :self.pixel_embedding_dim]
        bias = wandb[..., self.pixel_embedding_dim, *([None] * (pixel_embedding.ndim - 2))]
        mask_logits = einops.einsum(weight, pixel_embedding, 'n nq c, n c ... -> n nq ...') + bias
        return mask_logits

# modified from transformers.models.mask2former.modeling_mask2former.Mask2FormerMaskedAttentionDecoder
class MaskedAttentionDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        feature_channels: int,
        num_attention_heads: int,
        dim_feedforward: int | None = None,
        num_decoder_layers: int = 9,
        num_feature_levels: int = 3,
        num_queries: int = 100,
        pixel_embedding_dim: int = None,
        pre_norm: bool = True,
        layer_drop: float = 0.,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        # attention mask interpolation
        self.interpolate_mode = 'bilinear' if spatial_dims == 2 else 'trilinear'
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim = feature_channels
        self.query_embedding = nn.Embedding(num_queries, hidden_dim)
        self.num_feature_levels = num_feature_levels
        self.level_embedding = nn.Embedding(num_feature_levels, hidden_dim)
        self.key_position_embedding = SpatialPositionEmbedding(feature_channels, spatial_dims, flatten=True)
        if dim_feedforward is None:
            dim_feedforward = feature_channels * 4
        self.pre_norm = pre_norm
        self.layers: Sequence[MaskedAttentionDecoderLayer] | nn.ModuleList = nn.ModuleList([
            MaskedAttentionDecoderLayer(feature_channels, num_attention_heads, dim_feedforward, pre_norm)
            for _ in range(num_decoder_layers)
        ])
        # this is redundant for post-norm, but let's follow the reference implementation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_drop = layer_drop

        pixel_embedding_dim = fall_back_none(pixel_embedding_dim, feature_channels)
        self.mask_query_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pixel_embedding_dim),
        )

        self.gradient_checkpointing = gradient_checkpointing

        self.apply(init_common)

    def predict_mask(
        self,
        hidden_states: torch.Tensor,
        pixel_embedding: torch.Tensor,
        manual_mask: torch.Tensor | None = None,
        attention_mask_shape: Sequence[int] | None = None,
    ):
        mask_embeddings = self.layer_norm(hidden_states)
        projected_mask_embeddings = self.mask_query_projection(mask_embeddings)
        mask_logits = einops.einsum(projected_mask_embeddings, pixel_embedding, 'n nq c, n c ... -> n nq ...')
        if attention_mask_shape is None:
            return mask_logits
        with torch.no_grad():
            if manual_mask is None:
                attention_mask = nnf.interpolate(mask_logits, attention_mask_shape, mode=self.interpolate_mode).sigmoid()
            else:
                attention_mask = nnf.interpolate(manual_mask.float(), attention_mask_shape, mode=self.interpolate_mode)
            # `attn_mask` parameter of https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
            # "For a binary mask, a True value indicates that the corresponding position is not allowed to attend."
            attention_mask = einops.rearrange((attention_mask < 0.5).bool(), 'n nq ... -> n nq (...)')
            # no restriction for empty mask
            attention_mask.masked_fill_(attention_mask.all(dim=-1, keepdim=True), False)
            attention_mask = einops.repeat(attention_mask, 'n nq nk -> (n M) nq nk', M=self.num_attention_heads)
        return mask_embeddings, mask_logits, attention_mask

    def forward(
        self,
        key_feature_maps: list[torch.Tensor],
        pixel_embedding: torch.Tensor,
        manual_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size = pixel_embedding.shape[0]
        hidden_states = einops.repeat(self.query_embedding.weight, '... -> n ...', n=batch_size)
        # query_position_embeddings = einops.repeat(self.query_position_embedding.weight, '... -> n ...', n=batch_size)
        key_hidden_states = [
            flatten(feature_map) + self.level_embedding.weight[i]
            for i, feature_map in enumerate(key_feature_maps)
        ]
        key_position_embeddings = [self.key_position_embedding(x) for x in key_feature_maps]
        pixel_embedding = self.pixel_embedding_projection(pixel_embedding)
        mask_embeddings, mask_logits, attention_mask = self.predict_mask(
            hidden_states,
            pixel_embedding,
            manual_mask,
            key_feature_maps[0].shape[2:]
        )
        layers_mask_embeddings = [mask_embeddings]
        layers_mask_logits = [mask_logits]
        from transformers import OneFormerModel

        for idx, layer in enumerate(self.layers):
            if self.training and (random.uniform(0, 1) < self.layer_drop):
                continue
            level_index = idx % self.num_feature_levels
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint.checkpoint(
                    layer, hidden_states, query_position_embeddings, key_hidden_states[level_index],
                    key_position_embeddings[level_index], attention_mask,
                )
            else:
                hidden_states = layer(
                    hidden_states, query_position_embeddings, key_hidden_states[level_index],
                    key_position_embeddings[level_index], attention_mask,
                )

            mask_embeddings, mask_logits, attention_mask = self.predict_mask(
                hidden_states,
                pixel_embedding,
                manual_mask,
                key_feature_maps[(idx + 1) % self.num_feature_levels].shape[2:],
            )
            layers_mask_embeddings.append(mask_embeddings)
            layers_mask_logits.append(mask_logits)

        return layers_mask_embeddings, layers_mask_logits

def main():
    bs = 2
    sp = 3
    feature_channels = 192
    t_decoder = MaskedAttentionDecoder(sp, feature_channels).cuda()
    spatial_shapes = [
        (s, ) * sp
        for s in [48, 24, 12, 6]
    ]
    feature_maps = [
        torch.randn(bs, feature_channels, *shape, device='cuda')
        for shape in spatial_shapes
    ]
    t_decoder.forward(feature_maps[1:], feature_maps[0])

if __name__ == '__main__':
    main()
