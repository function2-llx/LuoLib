import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from xformers import ops as xops

from luolib.models.blocks import SpatialRotaryEmbedding
from luolib.types import NoWeightDecayParameter

__all__ = [
    'MemoryEfficientAttention',
]

from luolib.utils import fall_back_none

class MemoryEfficientAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        rope: SpatialRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim, _rem = divmod(dim, num_heads)
        assert _rem == 0
        self.scale = fall_back_none(qk_scale, self.head_dim ** -0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        if qkv_bias:
            self.q_bias = NoWeightDecayParameter(torch.zeros(dim))
            self.v_bias = NoWeightDecayParameter(torch.zeros(dim))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('v_bias', None)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.rope = rope

    def expand_head(self, x: torch.Tensor):
        return einops.rearrange(x, 'n l (nh d) -> n l nh d', nh=self.num_heads)

    def apply_rope(self, x: torch.Tensor):
        if self.rope is not None:
            x = torch.cat([x[:, :1], self.rope(x[:, 1:])], dim=1)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_bias: torch.Tensor | xops.AttentionBias | None = None,
    ):
        key = fall_back_none(key, query)
        value = fall_back_none(value, key)
        q = self.expand_head(nnf.linear(query, self.q_proj.weight, self.q_bias))
        k = self.expand_head(nnf.linear(key, self.k_proj.weight))
        v = self.expand_head(nnf.linear(value, self.v_proj.weight, self.v_bias))
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        x = einops.rearrange(
            # if using amp, v.dtype here will be the autocast dtype
            xops.memory_efficient_attention(q.type_as(v), k.type_as(v), v, attn_bias, self.attn_drop, self.scale),
            'n l nh d -> n l (nh d)',
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
