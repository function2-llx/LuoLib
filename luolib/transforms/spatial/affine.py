from collections.abc import Sequence

import numpy as np
import torch

from monai import transforms as mt
from monai.config import KeysCollection, SequenceStr
from monai.data import get_track_meta, set_track_meta, MetaTensor
from monai.transforms.spatial.array import RandRange
from monai.utils import GridSampleMode, GridSamplePadMode, ensure_tuple_rep

from luolib.types import maybe_seq_t, tuple2_t

__all__ = [
    'RandAffineGridWithIsotropicScale',
    'RandAffineWithIsotropicScale',
    'RandAffineWithIsotropicScaleD',
]

class RandAffineGridWithIsotropicScale(mt.RandAffineGrid):
    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: tuple2_t[float] | None = None,
        *args,
        spatial_dims: int,
        ignore_dim: int | None = None,
        **kwargs
    ):
        """
        Args:
            scale_range: common scale value not subtracting 1
            ignore_dim: the spatial dimension keep unchanged
        """
        super().__init__(rotate_range, shear_range, translate_range, None, *args, **kwargs)
        assert scale_range is not None
        self.scale_range = scale_range
        self.spatial_dims = spatial_dims
        self.ignore_dim = ignore_dim

    def randomize(self, data=None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = [float(self.R.uniform(*self.scale_range))] * self.spatial_dims
        if self.ignore_dim is not None:
            self.scale_params[self.ignore_dim] = 1

class RandAffineWithIsotropicScale(mt.RandAffine):
    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: tuple2_t[float] = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
        lazy: bool = False,
        *,
        spatial_dims: int,
        ignore_dim: int | None = None,
    ):
        super().__init__(prob, rotate_range, shear_range, translate_range, None, spatial_size, mode, padding_mode, cache_grid, device, lazy)
        self.rand_affine_grid = RandAffineGridWithIsotropicScale(
            rotate_range,
            shear_range,
            translate_range,
            scale_range,
            device=device,
            lazy=lazy,
            spatial_dims=spatial_dims,
            ignore_dim=ignore_dim,
        )

class RandAffineWithIsotropicScaleD(mt.RandAffineD):
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: tuple2_t[float] | None = None,
        mode: maybe_seq_t[int | str] = GridSampleMode.BILINEAR,
        padding_mode: maybe_seq_t[int | str] = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        *,
        spatial_dims: int,
        ignore_dim: int | None = None,
    ) -> None:
        super().__init__(keys, spatial_size, prob, rotate_range, shear_range, translate_range, None, mode, padding_mode, cache_grid, device, allow_missing_keys, lazy)
        self.rand_affine = RandAffineWithIsotropicScale(
            1.0,  # because probability handled in this class
            rotate_range,
            shear_range,
            translate_range,
            scale_range,
            spatial_size,
            cache_grid=cache_grid,
            device=device,
            lazy=lazy,
            spatial_dims=spatial_dims,
            ignore_dim=ignore_dim,
        )
