from collections.abc import Mapping, Sequence
from functools import lru_cache
import itertools as it

import einops
import numpy as np
import torch
from torch import nn
from xformers import ops as xops

from monai.networks.blocks import MLPBlock
from monai.networks.layers import DropPath
from monai.utils import ensure_tuple_rep

from luolib.utils import channel_first, channel_last
from ..layers import Act
from ..utils import forward_maybe_grad_ckpt
from ..param import NoWeightDecayParameter
from .tensor import SpatialTensor

__all__ = [
    'SwinLayer',
]

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            max_window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = NoWeightDecayParameter(
            torch.empty(num_heads, np.prod(2 * np.array(max_window_size) - 1))
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor | None, relative_position_index: torch.LongTensor):
        qkv = einops.rearrange(
            self.qkv(x), 'n l (qkv nh ch) -> qkv n l nh ch', qkv=3, nh=self.num_heads,
        ).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make TorchScript happy (cannot use tensor as tuple)
        attn_bias = einops.repeat(
            self.relative_position_bias_table[:, relative_position_index],
            'nh (l1 l2) -> n nh l1 l2',
            n=x.shape[0], l1=x.shape[1],
        ).clone()
        if mask is not None:
            attn_bias.masked_fill_(~mask[:, None], -torch.inf)
        x = einops.rearrange(
            xops.memory_efficient_attention(
                q, k, v, attn_bias.type(v.dtype), self.attn_drop, self.scale
            ),
            'n l nh d -> n l (nh d)',
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    """
    Args:
        x: channel-last feature map, (batch_size, d, h, w, c)
    Returns:
        - window-partitioned channel-last feature map, (batch_size * window_num, window_size, c)
        - dict of window_num for each spatial dimension
    """
    num_windows = {
        f'n{i}': x.shape[i + 1] // ws
        for i, ws in enumerate(window_size)
    }
    return einops.rearrange(x, 'n (n0 w0) (n1 w1) (n2 w2) c -> (n n0 n1 n2) (w0 w1 w2) c', **num_windows), num_windows

def window_reverse(x: torch.Tensor, window_size: Sequence[int], num_windows: Mapping[str, int]):
    return einops.rearrange(
        x, '(n n0 n1 n2) (w0 w1 w2) c -> n (n0 w0) (n1 w1) (n2 w2) c',
        **num_windows,
        **{
            f'w{i}': ws
            for i, ws in enumerate(window_size)
        }
    )

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: tuple | str | None = Act.GELU,
        # norm_layer: tuple | str | None = Norm.LAYER,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads: number of attention heads.
            max_window_size: local window size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
        """

        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            num_heads,
            max_window_size,
            qkv_bias,
            attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(
            hidden_size=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer,
            dropout_rate=drop,
            dropout_mode="swin",
        )

    def forward(
        self,
        x: torch.Tensor,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        attn_mask: torch.BoolTensor,
        relative_position_index: torch.LongTensor,
    ):
        shortcut = x
        x = self.norm1(x)

        shift_size = np.array(shift_size)
        if np.any(shift_size):  # a little faster
            x = x.roll(tuple(shift_size), dims=(1, 2, 3))   # shift towards pad
        parted_x, num_windows = window_partition(x, window_size)
        x = self.attn(parted_x, attn_mask, relative_position_index)
        x = window_reverse(x, window_size, num_windows)
        if np.any(shift_size):
            x = x.roll(tuple(-shift_size), dims=(1, 2, 3))
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@lru_cache(128)
def compute_shift_attn_mask(
    spatial_shape: Sequence[int],
    window_size: Sequence[int],
    shift_size: Sequence[int],
    device: torch.device,
) -> torch.BoolTensor | None:
    """
    TODO: there are at most 8 kinds of attention mask for 3D
    Returns:
        attention mask of (window_num, window_size, window_size)
    """
    img_mask = torch.zeros(spatial_shape, device=device, dtype=torch.int8)
    for i, slices in enumerate(
        it.product(
            *[
                [slice(s, None), slice(s)] if s else [slice(None)]
                for s in shift_size
            ]
        )
    ):
        img_mask[slices] = i
    mask_windows, _ = window_partition(img_mask[None, ..., None], window_size)
    mask_windows = mask_windows.squeeze(-1)  # squeeze the dummy channel
    attn_mask = mask_windows[:, None, :] == mask_windows[:, :, None]
    return attn_mask    # type: ignore

@lru_cache(maxsize=128)
def compute_relative_position_index(window_size: tuple[int, ...]) -> torch.LongTensor:
    window_size = torch.tensor(window_size)
    # create coordinates of w0 * w1 * w2, 3
    coords_flatten = torch.cartesian_prod(*map(torch.arange, window_size))
    # compute axis-wise relative distance & shift to start from 0
    relative_coords = coords_flatten[:, None] - coords_flatten[None, :] + window_size - 1
    # flatten 3D coordinates to 1D (which might be faster & more convenient for indexing)
    # PyTorch does not support negative stride yet: https://github.com/pytorch/pytorch/issues/59786
    relative_coords[..., :-1] *= (2 * window_size[1:] - 1).flip(dims=[0]).cumprod(dim=0).flip(dims=[0])
    relative_position_index = relative_coords.sum(dim=-1)
    return relative_position_index.view(-1)

def _to_tuple(x: np.ndarray):
    return tuple(x.tolist())

class SwinLayer(nn.Module):
    blocks: Sequence[SwinTransformerBlock] | nn.ModuleList

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        max_window_size: int | Sequence[int],
        drop_path_rates: list[float] | float = 0.,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        last_norm: bool = False,
        slide: bool = True,
        grad_ckpt: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            depth: number of blocks
            num_heads: number of attention heads.
            max_window_size: local window size.
            drop_path_rates: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            last_norm: whether to apply a norm to the output, this may be useful when using pre-norm
            slide: whether to enable sliding window
            grad_ckpt: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.max_window_size = np.array(ensure_tuple_rep(max_window_size, 3))
        assert (self.max_window_size & self.max_window_size - 1 == 0).all(), 'only power of 2 is supported'
        assert (self.max_window_size[1:] > 1).any(), 'unit in-plane window is not supported'

        self.grad_ckpt = grad_ckpt
        if isinstance(drop_path_rates, float):
            drop_path_rates = [drop_path_rates] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim,
                num_heads,
                self.max_window_size,
                mlp_ratio,
                qkv_bias,
                drop,
                attn_drop,
                drop_path_rate,
            )
            for drop_path_rate in drop_path_rates
        ])
        if last_norm:
            self.last_norm = nn.LayerNorm(dim)
        else:
            self.last_norm = nn.Identity()
        self.slide = slide

    def forward(self, x: SpatialTensor):
        spatial_shape = np.array(x.shape[2:])
        window_size = self.max_window_size.copy()
        window_size[0] = min(max(window_size[0] >> x.num_pending_hw_downsamples, 1), x.shape[2])
        relative_position_index = compute_relative_position_index(_to_tuple(window_size))
        assert (spatial_shape % window_size == 0).all(), "I don't want to support padding, do you?"
        if self.slide:
            shift_size = window_size >> 1
            shift_size[spatial_shape == window_size] = 0
        else:
            shift_size = np.zeros_like(window_size)
        if self.slide and shift_size.any():
            shift_attn_mask = compute_shift_attn_mask(
                _to_tuple(spatial_shape),
                _to_tuple(window_size),
                _to_tuple(shift_size),
                x.device,
            )
            shift_attn_mask = einops.repeat(shift_attn_mask, 'nw ... -> (n nw) ...', n=x.shape[0])
        else:
            shift_attn_mask = None
        x = channel_last(x).contiguous()
        for block, block_shift_size, block_attn_mask in zip(
            self.blocks,
            it.cycle([np.zeros_like(shift_size), shift_size]),
            it.cycle([None, shift_attn_mask]),
        ):
            block: SwinTransformerBlock
            x = forward_maybe_grad_ckpt(
                block, self.training and self.grad_ckpt,
                x, window_size, block_shift_size, block_attn_mask, relative_position_index,
            )
        x = self.last_norm(x)
        return channel_first(x).contiguous()

    def extra_repr(self) -> str:
        return f'max_window_size={_to_tuple(self.max_window_size)}, slide={self.slide}'
