import warnings
from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as torch_f

def with_if(x, y):
    return x if y is None else x + y

def get_spatial_pattern(spatial_shape: Sequence[int]):
    spatial_dims = len(spatial_shape)
    spatial_pattern = ' '.join(map(lambda i: f's{i}', range(spatial_dims)))
    spatial_dict = {
        f's{i}': s
        for i, s in enumerate(spatial_shape)
    }
    return spatial_pattern, spatial_dict

# modified from Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
class MultiscaleDeformableSelfAttention(nn.Module):
    """
    Multiscale deformable attention originally proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int, spatial_dims: int):
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

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * spatial_dims)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,  # (n, n_seq, L, sp), normalized to [-1, 1], no align corners
        spatial_shapes=None,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        hidden_states = with_if(hidden_states, position_embeddings)
        values = self.value_proj(hidden_states)
        values = einops.rearrange(values, '... (M d) -> ... M d', M=self.n_heads)
        # value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states)
        sampling_offsets = einops.rearrange(
            sampling_offsets, '... (M L K sp) -> ... M L K sp',
            M=self.n_heads, L=self.n_levels, K=self.n_points,
        )
        offset_normalizer = spatial_shapes
        sampling_points = einops.rearrange(reference_points, '... L sp -> ... 1 L 1 sp') \
            + sampling_offsets / einops.rearrange(offset_normalizer, 'L sp -> 1 L 1 sp')
        sampling_points = sampling_points.flip(dims=(-1, ))
        value_list = values.split(spatial_shapes.prod(dim=-1).tolist(), dim=1)
        sampling_value_list = []
        dummy_dim = ' 1 ' if self.spatial_dims == 3 else ' '
        for level_id, spatial_shape in enumerate(spatial_shapes):
            spatial_pattern, spatial_dict = get_spatial_pattern(spatial_shape)
            value_l_ = einops.rearrange(
                value_list[level_id],
                f'n ({spatial_pattern}) M d -> (n M) d {spatial_pattern}',
                **spatial_dict,
            )
            sampling_value = torch_f.grid_sample(
                value_l_,
                einops.rearrange(
                    sampling_points[:, :, :, level_id],
                    f'n nq M K sp -> (n M){dummy_dim}nq K sp'
                ),
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampling_value_list.append(sampling_value)
        sampling_values = einops.rearrange(sampling_value_list, f'L (n M) d{dummy_dim}nq K -> n nq M (L K) d', M=self.n_heads)
        # nk = L * K
        attention_weights = einops.rearrange(
            self.attention_weights(hidden_states), '... (M nk) -> ... M 1 nk', M=self.n_heads,
        ).softmax(dim=-1)
        output = einops.rearrange(attention_weights @ sampling_values, '... M 1 d -> ... (M d)')
        output = self.output_proj(output)

        return output

def main():
    bs = 2
    M = 8
    L = 3
    K = 4
    sp = 3
    dim = 256
    spatial_shapes = torch.tensor([
        [6] * sp,
        [12] * sp,
        [24] * sp,
    ]).cuda()
    nq = spatial_shapes.prod(dim=1).sum().item()
    ms_deform_sa = MultiscaleDeformableSelfAttention(dim, M, L, K, sp).cuda()
    hidden_states = torch.randn(bs, nq, dim).cuda()
    reference_points = torch.rand(bs, nq, L, sp).cuda() * 2 - 1
    ms_deform_sa.forward(hidden_states, reference_points=reference_points, spatial_shapes=spatial_shapes)

if __name__ == '__main__':
    main()
