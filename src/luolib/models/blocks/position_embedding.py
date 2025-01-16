from functools import lru_cache

import einops
import torch
from torch import nn

from luolib.types import spatial_shape_t, tuple3_t

__all__ = [
    'SpatialSinusoidalHelper',
    'SpatialSinusoidalPositionEmbedding',
]

def broadcast_cat(a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    tensors = torch.broadcast_tensors(a, b)
    return torch.cat(tensors, dim)

class SpatialSinusoidalHelper(nn.Module):
    """
        θ(h, w, d) = θ(h, w) + θ(d)
        θ(h, w) = θ(h) || θ(w)
    """
    @property
    def ω(self) -> list[torch.Tensor]:
        # let's wish we can have nn.BufferList soon:
        # - https://github.com/pytorch/pytorch/issues/35735
        # - https://github.com/pytorch/pytorch/issues/37386
        return [self.get_buffer(f'ω{i}') for i in range(3)]

    def __init__(self, dim: int, normalize: bool, scale: float | None = None, base: tuple3_t[float] = (2333., 1e4, 1e4)):
        """
        Args:
            dim: embedding dimension.
            normalize: whether to normalize the position to [-1, 1] w.r.t. the spatial shape. Choosing [-1, 1] instead
            of [0, 1] so that the only position of a dimension with shape=1 will be normalized to 0. Normalization may
            be important when relative positions are used.
            scale: position scaling factor (after normalization, of course).
        """
        super().__init__()
        if scale is None:
            scale = 1e2 if normalize else 1.
        assert dim & 3 == 0
        # every 2 elements share the same θ
        dim >>= 1
        for i in range(3):
            if i > 0:
                exp = -torch.arange(0, dim, 2) / dim
            else:
                exp = -torch.arange(1, dim, 2) / dim
                exp = einops.repeat(exp, '... -> (r ...)', r=2)
            self.register_buffer(f'ω{i}', torch.pow(base[i], exp), False)
        self.normalize = normalize
        self.scale = scale

    @lru_cache
    def get_rotation(self, spatial_shape: spatial_shape_t) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: cos(θ), sin(θ), where θ is of shape (*spatial_shape, d)
        """
        spatial_shape_3d = spatial_shape if len(spatial_shape) == 3 else (1, ) + spatial_shape
        θ = [
            torch.outer(
                torch.linspace(-1 + 1 / s, 1 - 1 / s, s, device=self.ω[i].device) if self.normalize
                else torch.arange(s, device=self.ω[i].device),
                self.ω[i],
            )
            for i, s in enumerate(spatial_shape_3d)
        ]
        θ_hw = broadcast_cat(θ[1][:, None], θ[2][None, :])
        θ = θ[0][:, None, None] + θ_hw[None, :]
        if len(spatial_shape) == 2:
            θ = θ[0]
        θ *= self.scale
        return θ.cos(), θ.sin()

class SpatialSinusoidalPositionEmbedding(SpatialSinusoidalHelper):
    def __init__(self, dim: int, normalize: bool, flatten: bool, **kwargs):
        super().__init__(dim, normalize, **kwargs)
        self.flatten = flatten

    def forward(self, spatial_shape: spatial_shape_t) -> torch.Tensor:
        cos, sin = super().get_rotation(spatial_shape)
        ret = einops.repeat([sin, cos], 'l2 ... d -> (d l2) ...', l2=2)
        if self.flatten:
            ret = einops.rearrange(ret, 'c ... -> (...) c')
        return ret
