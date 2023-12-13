from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
from timm.layers import DropPath
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from monai.utils import ensure_tuple_rep

from luolib.types import param3_t, tuple2_t, tuple3_t
from luolib.models import spadop
from ..param import NoWeightDecayParameter
from ..blocks import SpatialRotaryEmbedding, MemoryEfficientAttention
from ..utils import load_ckpt

__all__ = [
    'ViT',
    'SimpleViTAdapter',
]

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: param3_t[int] = 16, in_chans: int = 3, embed_dim: int = 768, adaptive: bool = True, flatten: bool = True):
        super().__init__()
        self.patch_size: tuple3_t[int] = ensure_tuple_rep(patch_size, 3)
        self.adaptive = adaptive
        self.proj = spadop.InputConv3D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, adaptive=adaptive)
        self.flatten = flatten

    def forward(self, x: torch.Tensor, flatten: bool | None = None) -> torch.Tensor:
        flatten = self.flatten if flatten is None else flatten
        x = self.proj(x)
        if flatten:
            x = einops.rearrange(x, 'n c ... -> n (...) c')
        return x

class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        sub_ln: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.ffn_ln = nn.LayerNorm(hidden_features) if sub_ln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float | int = 4.,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        proj_drop: float = 0.,
        drop_path: float = 0.,
        sub_ln: bool = False,
        rope: SpatialRotaryEmbedding = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MemoryEfficientAttention(dim, num_heads, qkv_bias, qk_scale, proj_drop, sub_ln, rope=rope)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim, sub_ln=sub_ln)

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@dataclass
class Checkpoint:
    path: Path | None = None
    state_dict_key: str = 'state_dict'

class ViT(nn.Module):
    def __init__(
        self, *,
        in_channels: int = 3,
        patch_size: param3_t[int] = 16,
        adaptive_patch_embed: bool = True,
        embed_dim: int = 768,
        pos_embed_shape: tuple3_t[int],
        pretrained_pos_embed_shape: tuple2_t[int] | None = None,
        rope_rescale_shape: tuple3_t[int] = (-1, -1, -1),
        rope_base: tuple3_t[float] = (2333., 10000., 10000.),
        rope_merge_hw: bool = True,
        depth: int = 12,
        num_heads: int = 12,
        sub_ln: bool = True,
        mlp_ratio: float = 8 / 3,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        grad_ckpt: bool = False,
        patch_embed_grad_scale: float = 1.,
        pretrained_ckpt: Checkpoint,
    ):
        """
        Args:
            pretrained_pos_embed_shape: used for spatialize 2D pre-trained positional embedding (from EVA-02)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim, adaptive_patch_embed, False)
        self.cls_token = NoWeightDecayParameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = NoWeightDecayParameter(torch.empty(1, embed_dim, *pos_embed_shape))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.pretrained_pos_embed_shape = pretrained_pos_embed_shape
        self.pos_drop = nn.Dropout(drop_rate, inplace=True)
        self.rope = SpatialRotaryEmbedding(embed_dim // num_heads, rope_rescale_shape, rope_base, rope_merge_hw)
        self.num_heads = num_heads
        self.sub_ln = sub_ln
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks: Sequence[Block] | nn.ModuleList = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                proj_drop=drop_rate,
                drop_path=dpr[i],
                sub_ln=sub_ln,
                rope=self.rope,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.grad_ckpt = grad_ckpt
        self.patch_embed_grad_scale = patch_embed_grad_scale
        load_ckpt(self, pretrained_ckpt.path, pretrained_ckpt.state_dict_key)

    def prepare_seq_input(self, x: torch.Tensor):
        x = self.patch_embed(x)
        shape = x.shape[2:]
        x += spadop.resample(self.pos_embed, shape)
        x = self.pos_drop(x)
        x = torch.cat(
            [
                self.cls_token.expand(x.shape[0], -1, -1),
                einops.rearrange(x, 'n c ... -> n (...) c'),
            ],
            dim=1,
        )
        return x, shape

    def forward_features(self, x: torch.Tensor):
        x, shape = self.prepare_seq_input(x)
        self.rope.prepare(shape)
        states = []
        for block in self.blocks:
            if self.training and self.grad_ckpt:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
            states.append(x)
        # don't reset! or gradient checkpointing will fail
        # self.rope.reset()
        return states

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)[-1]
        return self.norm(x)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        for k in list(state_dict):
            if k.endswith('rope.freqs_cos') or k.endswith('rope.freqs_sin'):
                state_dict.pop(k)
        if (weight := state_dict.get('patch_embed.proj.weight')) is not None:
            d = self.patch_embed.patch_size[0]
            if weight.ndim == 4:
                # conv2d weight from EVA-02
                weight = nnf.interpolate(weight.float(), self.patch_embed.patch_size[1:], mode='bicubic')
                weight = einops.repeat(weight / d, 'co ci ... -> co ci d ...', d=d)
            else:
                # weight from PUMIT
                weight = einops.reduce(
                    weight,
                    'co ci (dr dc) ... -> co ci dr ...',
                    'sum',
                    dr=d,
                )
            state_dict['patch_embed.proj.weight'] = weight

        if (pos_embed := state_dict.get('pos_embed')) is not None:
            if pos_embed.ndim == 3:
                cls_pos_embed, pos_embed = pos_embed[:, 1], pos_embed[:, 1:]
                state_dict['cls_token'] += cls_pos_embed
                h, w = self.pretrained_pos_embed_shape
                pos_embed = einops.repeat(
                    pos_embed, '1 (h w) c -> 1 c d h w',
                    d=self.pos_embed.shape[2], h=h, w=w,
                )
            state_dict['pos_embed'] = spadop.resample(pos_embed, self.pos_embed.shape[2:])

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class SimpleViTAdapter(ViT):
    def __init__(self, *args, out_indexes: Sequence[int], **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.embed_dim
        patch_size = self.patch_embed.patch_size
        assert patch_size[1] == patch_size[2] == 16
        # TODO: handle in-plane patch size of 8
        assert patch_size[0] & patch_size[0] - 1 == 0
        aniso_d = max(0, (16 // patch_size[0]).bit_length() - 1)
        assert not self.patch_embed.adaptive
        get_args = lambda i: ((1 if aniso_d >= i else 2, 2, 2), ) * 2
        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(dim, dim, *get_args(4)),
                nn.InstanceNorm3d(dim, affine=True),
                nn.GELU(),
                nn.ConvTranspose3d(dim, dim, *get_args(3)),
            ),
            nn.ConvTranspose3d(dim, dim, *get_args(4)),
            nn.Identity(),
            nn.MaxPool3d(*get_args(5)),
        ])
        self.out_indexes = out_indexes
        assert len(out_indexes) == len(self.fpn)
        self.norms = nn.ModuleList([nn.InstanceNorm3d(dim, affine=True) for _ in range(len(out_indexes))])

    def forward(self, x: torch.Tensor):
        states = self.forward_features(x)
        d, h, w = np.array(x.shape[2:]) // np.array(self.patch_embed.patch_size)
        ret = []
        for out_id, norm, fpn in zip(self.out_indexes, self.norms, self.fpn):
            feature_map = einops.rearrange(states[out_id][:, 1:], 'n (d h w) c -> n c d h w', d=d, h=h, w=w)
            ret.append(fpn(norm(feature_map)))
        return ret
