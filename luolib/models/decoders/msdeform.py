import warnings
from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from monai.networks.blocks import MLPBlock

from luolib.types import NoWeightDecayParameter, get_conv_t
from luolib.utils import fall_back_none, flatten
from ..init import init_common
from ..blocks import SpatialSinusoidalPositionEmbedding, transformer_block_forward

__all__ = [
    'MultiscaleDeformablePixelDecoder',
]

def get_spatial_pattern(spatial_shape: Sequence[int]):
    spatial_dims = len(spatial_shape)
    spatial_pattern = ' '.join(map(lambda i: f's{i}', range(spatial_dims)))
    spatial_dict = {
        f's{i}': s
        for i, s in enumerate(spatial_shape)
    }
    return spatial_pattern, spatial_dict

class MultiscaleDeformableSelfAttention(nn.Module):
    """
    Multiscale deformable attention originally proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int, spatial_dims: int, proj_drop: float = 0.):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        dim_per_head = embed_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.spatial_dims = spatial_dims
        self.proj_drop = nn.Dropout(proj_drop)

        # TODO: unify spatial_dims of 2 & 3
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * spatial_dims)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        """
        nq = total length of all flattened feature maps
        Args:
            hidden_states: (n, nq, d)
            position_embeddings: (nq, d)
            reference_points: (nq, spatial_dims), normalized to [-1, 1], no align corners
            spatial_shapes: (L, spatial_dims)
        """
        values = self.value_proj(hidden_states)
        value_list: tuple[torch.Tensor, ...] = values.split(spatial_shapes.prod(dim=-1).tolist(), dim=1)
        # add position embeddings to the hidden states before projecting to queries and keys
        hidden_states = hidden_states + position_embeddings
        sampling_offsets = einops.rearrange(
            self.sampling_offsets(hidden_states),
            'n nq (L M K sp) -> L n nq M K sp', L=self.n_levels, M=self.n_heads, K=self.n_points,
        )
        sampling_points = einops.rearrange(reference_points, 'nq sp -> nq 1 1 sp') + sampling_offsets
        # DHW -> xyz
        sampling_points = sampling_points.flip(dims=[-1])
        sampling_value_list = []
        # dummy_dim: for grid sampling, PyTorch forces sampling on N-d input to have an N-d output, very annoying
        dummy_spatial_pattern = ' '.join('1' * (self.spatial_dims - 1))
        for level_id, spatial_shape in enumerate(spatial_shapes):
            spatial_pattern, spatial_dict = get_spatial_pattern(spatial_shape)
            level_value = einops.rearrange(
                value_list[level_id],
                f'n ({spatial_pattern}) (M d) -> (n M) d {spatial_pattern}',
                **spatial_dict, M=self.n_heads,
            )
            sampling_value = nnf.grid_sample(
                level_value,
                einops.rearrange(
                    sampling_points[level_id],
                    f'n nq M K sp -> (n M) (nq K) {dummy_spatial_pattern} sp',
                ),
                mode='bilinear', padding_mode='zeros', align_corners=False,
            )
            sampling_value_list.append(sampling_value)
        sampling_values = einops.rearrange(
            sampling_value_list,
            f'L (n M) d (nq K) {dummy_spatial_pattern} -> n nq M (L K) d', M=self.n_heads, K=self.n_points,
        )
        attention_weights = einops.rearrange(
            self.attention_weights(hidden_states),
            'n nq (M LK) -> n nq M 1 LK', M=self.n_heads,
        ).softmax(dim=-1)
        output = einops.rearrange(attention_weights @ sampling_values, 'n nq M 1 d -> n nq (M d)')
        output = self.output_proj(output)
        output = self.proj_drop(output)

        return output

class MultiscaleDeformablePixelDecoderLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        embed_dim: int,
        num_heads: int,
        n_levels: int,
        n_points: int,
        mlp_dim: int,
        pre_norm: bool = False,
        dropout: float = 0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pre_norm = pre_norm

        self.ms_deform_sa = MultiscaleDeformableSelfAttention(embed_dim, num_heads, n_levels, n_points, spatial_dims, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout, 'RELU')
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @property
    def post_norm(self):
        return not self.pre_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = transformer_block_forward(
            hidden_states, self.ms_deform_sa, self.self_attn_layer_norm, self.pre_norm,
            kwargs=dict(
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                position_embeddings=position_embeddings
            ),
        )
        hidden_states = transformer_block_forward(hidden_states, self.mlp, self.final_layer_norm, self.pre_norm)

        # if self.training:
        #     if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
        #         clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        #         hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states

class MultiscaleDeformablePixelDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        backbone_feature_channels: Sequence[int],
        feature_dim: int,
        num_gn_groups: int,
        num_heads: int,
        num_feature_levels: int = 3,
        n_points: int = 4,
        num_layers: int = 6,
        mlp_dim: int | None = None,
        grad_ckpt: bool = False,
        **kwargs,
    ):
        """
            Args:
                backbone_feature_channels: feature channels of high → low resolution backbone feature maps
                n_points: key points for each level & head
        """
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.interpolate_mode = 'bilinear' if spatial_dims == 2 else 'trilinear'
        conv_t = get_conv_t(spatial_dims)

        self.input_projections = nn.ModuleList([
            nn.Sequential(
                conv_t(backbone_feature_channel, feature_dim, 1),
                nn.GroupNorm(num_gn_groups, feature_dim),
            )
            for backbone_feature_channel in backbone_feature_channels
        ])

        self.position_embedding = SpatialSinusoidalPositionEmbedding(feature_dim, normalize=True, flatten=True)
        self.level_embedding = NoWeightDecayParameter(torch.empty(num_feature_levels, feature_dim))
        mlp_dim = fall_back_none(mlp_dim, 4 * feature_dim)
        self.layers: Sequence[MultiscaleDeformablePixelDecoderLayer] | nn.ModuleList = nn.ModuleList([
            MultiscaleDeformablePixelDecoderLayer(spatial_dims, feature_dim, num_heads, num_feature_levels, n_points, mlp_dim)
            for _ in range(num_layers)
        ])

        self.num_fpn_levels = len(backbone_feature_channels) - num_feature_levels
        self.fpn_output_convs = nn.ModuleList([
            nn.Sequential(
                # TODO: use sac
                conv_t(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(num_gn_groups, feature_dim),
                nn.ReLU(),
            )
            for _ in range(self.num_fpn_levels)
        ])

        self.gradient_checkpointing = grad_ckpt

        # follows nn.Embedding
        nn.init.normal_(self.level_embedding)
        self.apply(init_common)

    @staticmethod
    def get_reference_points(spatial_shapes: torch.Tensor) -> torch.Tensor:
        """
        Get normalized point coordinates for each spatial position
        Args:
            spatial_shapes: spatial shape of feature map of each level (num_levels * spatial_dims)
        Returns:
            reference points coordinates of each query, normalized to [-1, 1], align_corners=False
        """
        device = spatial_shapes.device
        reference_points = torch.cat(
            [
                torch.cartesian_prod(*[
                    torch.linspace(-1 + 1 / s, 1 - 1 / s, s, dtype=torch.float32, device=device)
                    for s in spatial_shape
                ])
                for spatial_shape in spatial_shapes
            ],
            dim=0,
        )
        return reference_points

    def forward_deformable_layers(self, feature_maps: list[torch.Tensor]):
        spatial_shapes = [feature_map.shape[2:] for feature_map in feature_maps]
        position_embeddings = list(map(self.position_embedding, spatial_shapes))
        hidden_states = torch.cat(list(map(flatten, feature_maps)), dim=1)
        position_embeddings = torch.cat(
            [
                position_embedding + self.level_embedding[i]
                for i, position_embedding in enumerate(position_embeddings)
            ],
            dim=0,
        )
        spatial_shapes = torch.tensor(spatial_shapes, device=hidden_states.device)
        reference_points = self.get_reference_points(spatial_shapes)

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_embeddings,
                    reference_points,
                    spatial_shapes,
                )
            else:
                hidden_states = layer(hidden_states, position_embeddings, reference_points, spatial_shapes)
        return [
            einops.rearrange(x, f'n ({spatial_pattern}) d -> n d {spatial_pattern}', **spatial_dict)
            for x, (spatial_pattern, spatial_dict) in zip(
                hidden_states.split(spatial_shapes.prod(dim=-1).tolist(), dim=1),
                [get_spatial_pattern(spatial_shape) for spatial_shape in spatial_shapes]
            )
        ]

    def forward(self, feature_maps: list[torch.Tensor], *args) -> list[torch.Tensor]:
        feature_maps = [
            projection(feature_map)
            for projection, feature_map in zip(self.input_projections, feature_maps)
        ]
        outputs = self.forward_deformable_layers(feature_maps[self.num_fpn_levels:])
        # turn to resolution: low → high, for appending FPN
        outputs = outputs[::-1]
        for lateral, output_conv in zip(feature_maps, self.fpn_output_convs):
            output = lateral + nnf.interpolate(outputs[-1], lateral.shape[2:], mode=self.interpolate_mode)
            outputs.append(output_conv(output))
        return outputs[::-1]
