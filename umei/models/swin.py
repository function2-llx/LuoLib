from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools as it

from einops import rearrange
import numpy as np
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils import checkpoint
from torch.nn import functional as torch_f

from monai.networks.layers import DropPath, get_norm_layer
from monai.networks.blocks import MLPBlock
from monai.umei import UEncoderBase, BackboneOutput

from umei.models.adaptive_resampling import AdaptiveDownsampling
from umei.models.blocks import ResLayer, get_conv_layer
from umei.models.layers import Act, LayerNormNd, Norm


from umei.utils import channel_first, channel_last

# used in blocks, x is channel last
def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    num_windows = {
        f'n{i}': x.shape[i + 1] // ws
        for i, ws in enumerate(window_size)
    }
    return rearrange(x, 'n (n0 w0) (n1 w1) (n2 w2) c -> (n n0 n1 n2) (w0 w1 w2) c', **num_windows), num_windows

def window_reverse(x: torch.Tensor, window_size: Sequence[int], num_windows: Mapping[str, int]):
    return rearrange(x, '(n n0 n1 n2) (w0 w1 w2) c -> n (n0 w0) (n1 w1) (n2 w2) c', **{
        **num_windows,
        **{
            f'w{i}': ws
            for i, ws in enumerate(window_size)
        }
    })

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_window_size: Sequence[int],
        relative_position_index: torch.Tensor,
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
        self.scale = np.power(head_dim, -0.5)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                np.product(2 * np.array(max_window_size) - 1),
                num_heads,
            )
        )
        self.relative_position_index = relative_position_index.view(-1)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor):
        qkv = rearrange(self.qkv(x), 'n l (qkv nh ch) -> qkv n nh l ch', qkv=3, nh=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = rearrange(
            self.relative_position_bias_table[self.relative_position_index],
            '(l1 l2) nh -> nh l1 l2',
            l1=x.shape[1],
        )
        attn = attn + relative_position_bias[None]
        # attn_view = rearrange(attn, '(n nw) nh l1 l2 -> nw l1 l2 n nh', nw=mask.shape[0])
        # assert attn.data_ptr() == attn_view.data_ptr()
        rearrange(attn, '(n nw) nh l1 l2 -> nw l1 l2 n nh', nw=mask.shape[0])[mask] = -torch.inf

        attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = rearrange(attn @ v, 'n nh l d -> n l (nh d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_window_size: Sequence[int],
        relative_position_index: torch.Tensor,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: tuple | str | None = Act.GELU,
        norm_layer: tuple | str | None = Norm.LAYER,
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
            norm_layer: normalization layer.
        """

        super().__init__()
        self.num_heads = num_heads
        self.norm1 = get_norm_layer(norm_layer, 3, dim)
        self.attn = WindowAttention(
            dim,
            num_heads,
            max_window_size,
            relative_position_index,
            qkv_bias,
            attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = get_norm_layer(norm_layer, 3, dim)
        self.mlp = MLPBlock(
            hidden_size=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer,
            dropout_rate=drop,
            dropout_mode="swin",
        )

    def forward(self, x: torch.Tensor, window_size: Sequence[int], shift_size: Sequence[int], attn_mask: torch.BoolTensor):
        shortcut = x
        x = self.norm1(x)

        shift_size = np.array(shift_size)
        shifted_x = x.roll(tuple(shift_size), dims=(1, 2, 3))
        parted_x, num_windows = window_partition(shifted_x, window_size)
        attn_x = self.attn.forward(parted_x, mask=attn_mask)
        shifted_x = window_reverse(attn_x, window_size, num_windows)
        x = shifted_x.roll(tuple(-shift_size), dims=(1, 2, 3))
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# TODO: there are at most 4 kinds of attention mask
def compute_attn_mask(img_mask: torch.Tensor, window_size: Sequence[int]) -> torch.BoolTensor:
    mask_windows, _ = window_partition(img_mask[None, ..., None], window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows[:, None, :] == mask_windows[:, :, None]
    return attn_mask    # type: ignore

class SwinLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    blocks: Sequence[SwinTransformerBlock] | nn.ModuleList

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        max_window_size: Sequence[int],
        drop_path_rates: list[float] | float = 0.,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: tuple | str | None = Norm.LAYER,
        use_checkpoint: bool = False,
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
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.max_window_size = np.array(max_window_size)
        # for tensor computation
        max_window_size = torch.tensor(max_window_size)
        coords_flatten = torch.cartesian_prod(*[
            torch.arange(s)
            for s in max_window_size
        ])  # create coordinates of w0 * w1 * w2, 3
        # compute axis-wise relative distance & shift to start from 0
        relative_coords = coords_flatten[None, :] - coords_flatten[:, None] + max_window_size - 1
        # flatten 3D coordinates to 1D (which might be faster & more convenient for indexing)
        # PyTorch does not support negative stride yet: https://github.com/pytorch/pytorch/issues/59786
        relative_coords[..., :-1] *= (2 * max_window_size[1:] - 1).flip(dims=(0, )).cumprod(dim=0)
        relative_position_index = relative_coords.sum(dim=-1)

        self.shift_size = self.max_window_size // 2
        self.use_checkpoint = use_checkpoint
        if isinstance(drop_path_rates, float):
            drop_path_rates = [drop_path_rates] * depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim,
                    num_heads,
                    self.max_window_size,
                    relative_position_index,
                    mlp_ratio,
                    qkv_bias,
                    drop,
                    attn_drop,
                    drop_path_rate,
                    norm_layer=norm_layer,
                )
                for drop_path_rate in drop_path_rates
            ]
        )

    def forward(self, x: torch.Tensor):
        spatial_shape = np.array(x.shape[2:])
        window_size = np.minimum(self.max_window_size, spatial_shape)
        pad_size = (window_size - spatial_shape % window_size) % window_size
        x = torch_f.pad(x, tuple(np.ravel([np.zeros_like(pad_size), np.flip(pad_size)], 'F')))

        img_mask = torch.zeros(x.shape[2:], dtype=torch.int32, device=x.device)
        img_mask[tuple(map(slice, spatial_shape))] = 1
        attn_mask = compute_attn_mask(img_mask, window_size)

        shift_size = np.where(window_size > self.max_window_size, self.shift_size, 0)    # type: ignore
        for i, slices in enumerate(
            it.product(*[
                [slice(s, None), slice(s)] if s else [slice(None)]
                for s in shift_size
            ])
        ):
            img_mask[slices] = i + 1
        shift_attn_mask = compute_attn_mask(img_mask, window_size)

        x = channel_last(x)
        for blk, shift_size, attn_mask in zip(
            self.blocks,
            it.cycle([np.zeros_like(self.shift_size), self.shift_size]),
            it.cycle([attn_mask, shift_attn_mask]),
        ):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, window_size, shift_size, attn_mask)
            else:
                x = blk.forward(x, window_size, shift_size, attn_mask)
        return channel_first(x)


class SwinBackbone(UEncoderBase):
    """
    Modify from MONAI implementation, support 3D only
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_channels: int,
        layer_channels: int | Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        layer_depths: Sequence[int],
        num_conv_layers: int,
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            layer_channels: number of channels for each layer.
            layer_depths: number of block in each layer.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        num_layers = len(layer_depths)
        if isinstance(layer_channels, int):
            layer_channels = [layer_channels << i for i in range(num_layers)]
        # number of channels for the last downsampled feature map
        if len(layer_channels) == num_layers:
            layer_channels.append(layer_channels[-1])
        self.stem = get_conv_layer(
            in_channels,
            layer_channels[0],
            kernel_sizes[0],
            stride=1,
        )
        layer_drop_path_rates = np.split(
            np.linspace(0, drop_path_rate, sum(layer_depths)),
            np.cumsum(layer_depths)[:-1],
        )

        self.layers = nn.ModuleList([
            ResLayer(
                layer_depths[i],
                layer_channels[i],
                kernel_sizes[i],
                drop_rate,
                Norm.LAYERND,
                Act.GELU,
                layer_drop_path_rates[i],
            )
            if i < num_conv_layers
            else SwinLayer(
                layer_channels[i],
                layer_depths[i],
                num_heads[i],
                kernel_sizes[i],    # max_window_size
                layer_drop_path_rates[i],
                mlp_ratio,
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                Norm.LAYER,
                use_checkpoint,
            )
            for i in range(num_layers)
        ])
        self.downsamplings = nn.ModuleList([
            AdaptiveDownsampling(
                layer_channels[i],  # in
                layer_channels[i + 1],  # out
                kernel_sizes[i],
            )
            for i in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            LayerNormNd(layer_channels[i])
            for i in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature_maps = []
        for layer, norm, downsampling in zip(self.layers, self.norms, self.downsamplings):
            x = layer(x)
            x = norm(x)
            feature_maps.append(x)
            x, downsampled = downsampling(x)
        return feature_maps

    def forward(self, x: torch.Tensor, *args) -> BackboneOutput:
        x = self.stem(x)
        feature_maps = self.forward_layers(x)
        return BackboneOutput(
            cls_feature=self.pool(feature_maps[-1]),
            hidden_states=feature_maps[:-1],
        )
