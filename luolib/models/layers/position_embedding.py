from functools import lru_cache
from typing import TypeAlias

import math

import einops
import torch
from torch import nn

from luolib.types import tuple2_t, tuple3_t
from luolib.utils import flatten

spatial_shape_t: TypeAlias = tuple2_t[int] | tuple3_t[int]

def broadcast_cat(a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    tensors = torch.broadcast_tensors(a, b)
    return torch.cat(tensors, dim)

class SpatialSinusoidalEmbeddingHelper(nn.Module):
    @property
    def ω(self) -> list[torch.Tensor]:
        # let's wish we can have nn.BufferList soon:
        # - https://github.com/pytorch/pytorch/issues/35735
        # - https://github.com/pytorch/pytorch/issues/37386
        return [self.get_buffer(f'ω{i}') for i in range(3)]

    def __init__(self, dim: int, base: tuple3_t[float] = (2333., 1e4, 1e4)):
        super().__init__()
        assert dim & 3 == 0
        # every 2 elements share the same θ
        dim >>= 1
        for i in range(3):
            # θ(h, w, d) = θ(h, w) + θ(d)
            if i > 0:
                exp = -torch.arange(0, dim, 2) / dim
            else:
                exp = -torch.arange(1, dim, 2) / dim
                exp = einops.repeat(exp, '... -> (r ...)', r=2)
            self.register_buffer(f'ω{i}', torch.pow(base[i], exp), False)

    @lru_cache
    def get_rotation(self, spatial_shape: spatial_shape_t) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_shape_3d = spatial_shape if len(spatial_shape) == 3 else (1, ) + spatial_shape
        θ = [
            torch.outer(
                torch.arange(spatial_shape_3d[i], device=self.ω[i].device),
                self.ω[i],
            )
            for i in range(3)
        ]
        θ_hw = broadcast_cat(θ[1][:, None], θ[2][None, :])
        θ = θ[0][:, None, None] + θ_hw[None, :]
        if len(spatial_shape) == 2:
            θ = θ[0]
        # (*spatial_shape, d)
        return θ.cos(), θ.sin()

class SpatialAbsolutePositionEmbedding(SpatialSinusoidalEmbeddingHelper):
    def __init__(self, embed_dim: int, flatten: bool = False):
        super().__init__(embed_dim)
        self.flatten = flatten

    def forward(self, spatial_shape: spatial_shape_t, batch_size: int = 1) -> torch.Tensor:
        cos, sin = super().get_rotation(spatial_shape)
        ret = einops.repeat([sin, cos], 'l2 ... d -> n (d l2) ...', l2=2, n=batch_size)
        if self.flatten:
            ret = flatten(ret)
        return ret
