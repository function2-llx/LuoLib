from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Type

from einops import rearrange
import torch
from torch import nn

from monai.networks.blocks import PatchEmbed, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import BasicLayer, PatchMerging
from umei import UDecoderBase, UEncoderBase
from umei.model import UDecoderOutput, UEncoderOutput

__all__ = ['SwinTransformer', 'SwinUnetrDecoder']

class SwinTransformer(UEncoderBase):
    """
    Modify from MONAI implementation, support 3D only
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        assert spatial_dims == 3
        num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=embed_dim << i_layer,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[:i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer + 1 < num_layers else None,
                use_checkpoint=use_checkpoint,
            )
            for i_layer in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            norm_layer(embed_dim << i)
            for i in range(num_layers)
        ])
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        hidden_states = []
        for layer, norm in zip(self.layers, self.norms):
            z, z_ds = layer(x)
            z = rearrange(z, "n c d h w -> n d h w c")
            z = norm(z)
            z = rearrange(z, "n d h w c -> n c d h w")
            hidden_states.append(z)
            x = z_ds

        return hidden_states

    def forward(self, x: torch.Tensor) -> UEncoderOutput:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        hidden_states = self.forward_layers(x)
        return UEncoderOutput(
            cls_feature=self.avg_pool(hidden_states[-1]).flatten(1),
            hidden_states=hidden_states,
        )

class SwinUnetrDecoder(UDecoderBase):
    def __init__(
        self,
        in_channels: int,
        feature_size: int = 24,
        num_layers: int = 4,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
        input_stride: Optional[Sequence[int] | int] = None,
    ) -> None:
        super().__init__()
        assert spatial_dims == 3

        self.bottleneck = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size << num_layers - 1,
            out_channels=feature_size << num_layers - 1,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.ups = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size << i,
                out_channels=feature_size << i - 1,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
            for i in range(1, num_layers)
        ])

        if input_stride is not None:
            self.input_encoder = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=input_stride,
                norm_name=norm_name,
                res_block=True,
            )
            self.ups.append(
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True,
                )
            )
        else:
            self.input_encoder = None

    def forward(self, x_in: torch.Tensor, hidden_states: list[torch.Tensor]) -> UDecoderOutput:
        if self.input_encoder is not None:
            hidden_states = [self.input_encoder(x_in), *hidden_states]
        x = self.bottleneck(hidden_states[-1])
        feature_maps = []
        for z, up in zip(hidden_states[-2::-1], self.ups[::-1]):
            up: UnetrUpBlock
            x = up(x, z if up.use_skip else None)
            feature_maps.append(x)
        return UDecoderOutput(feature_maps)
