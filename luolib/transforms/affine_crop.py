import math
from typing import Callable, Hashable, Mapping, Sequence

from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as torch_f

from monai import transforms as monai_t
from monai.config import KeysCollection, SequenceStr
from monai.networks.utils import meshgrid_ij
from monai.transforms import Randomizable, create_rotate, create_scale, create_translate
from monai.utils import GridSamplePadMode, TransformBackends, ensure_tuple_rep, get_equivalent_dtype

from luolib.types import param3_t, spatial_param_t, tuple2_t
from .utils import SpatialRangeGenerator

class RandAffineCropD(monai_t.Randomizable, monai_t.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        crop_size: Sequence[int],
        sample_mode: SequenceStr,
        rotate_range: param3_t[tuple2_t[float]],
        rotate_p: param3_t[float],
        scale_range: spatial_param_t[tuple2_t[float]],
        scale_p: spatial_param_t[float],
        spatial_dims: int = 3,
        dummy_dim: int | None = None,
        padding_mode: SequenceStr = GridSamplePadMode.ZEROS,
        allow_missing_keys: bool = False,
        *,
        center_generator: Callable[[Mapping[Hashable, torch.Tensor]], Sequence[int]],
    ):
        monai_t.MapTransform.__init__(self, keys, allow_missing_keys)
        self.crop_size = np.array(crop_size)
        self.sample_mode = ensure_tuple_rep(sample_mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.dummy_dim = dummy_dim
        self.center_generator = center_generator
        real_spatial_dims = spatial_dims - (dummy_dim is not None)
        self.rotate_generator = SpatialRangeGenerator(
            rotate_range,
            rotate_p,
            repeat=math.comb(real_spatial_dims, 2),
        )
        self.id_rotate = np.zeros(math.comb(real_spatial_dims, 2))
        self.scale_generator = SpatialRangeGenerator(
            scale_range,
            scale_p,
            default=1.,
            repeat=real_spatial_dims,
        )
        self.id_scale = np.ones(real_spatial_dims)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        super().set_random_state(seed, state)
        if isinstance(self.center_generator, monai_t.Randomizable):
            self.center_generator.set_random_state(seed, state)
        self.rotate_generator.set_random_state(seed, state)
        self.scale_generator.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)
        self.center = center = np.array(self.center_generator(d))
        sample_x = d[self.first_key(d)]
        spatial_size = np.array(sample_x.shape[1:])
        self.rotate_params = rotate_params = self.rotate_generator(d)
        self.scale_params = scale_params = self.scale_generator(d)
        if rotate_params is not None or scale_params is not None:
            if self.dummy_dim is not None:
                dummy_crop_size = self.crop_size[self.dummy_dim]
                crop_size = np.delete(self.crop_size, self.dummy_dim)
                spatial_size = np.delete(spatial_size, self.dummy_dim)
                dummy_center = center[self.dummy_dim]
                dummy_slice = slice(
                    dummy_center - (dummy_crop_size >> 1),
                    dummy_center + dummy_crop_size - (dummy_crop_size >> 1),
                )
                center = np.delete(center, self.dummy_dim)
            else:
                crop_size = self.crop_size

            spatial_dims = len(spatial_size)
            patch_grid = create_patch_grid(spatial_size, center, crop_size, sample_x.device)
            center_coord = center - (spatial_size - 1) / 2  # center's coordination in grid
            backend = TransformBackends.TORCH
            # shift the center to 0
            affine = create_translate(spatial_dims, -center_coord, backend=backend)
            if rotate_params is not None:
                affine = create_rotate(spatial_dims, rotate_params, backend=backend) @ affine
            if scale_params is not None:
                affine = create_scale(spatial_dims, scale_params, backend=backend) @ affine
            # shift center back
            affine = create_translate(spatial_dims, center_coord, backend=backend) @ affine
            # apply affine on patch grid
            patch_grid = patch_grid.view(spatial_dims + 1, -1)
            patch_grid = affine @ patch_grid
            patch_grid = patch_grid.view(spatial_dims + 1, *crop_size)
            patch_grid = patch_grid[list(reversed(range(spatial_dims)))]  # PyTorch believes D H W <-> z y x
            patch_grid = rearrange(patch_grid, 'sd ... -> 1 ... sd')
            # normalize grid, remember to flip the spatial size as well
            patch_grid /= torch.from_numpy(np.maximum((np.flip(spatial_size) - 1) / 2, 1)).to(patch_grid)

            # monai_t.Resample is not traceable, no better than resampling myself
            for key, mode, padding_mode in self.key_iterator(d, self.sample_mode, self.padding_mode):
                x = d[key]
                if self.dummy_dim is not None:
                    x = x.movedim(self.dummy_dim + 1, 1)
                    x = x[:, dummy_slice]  # directly crop along dummy dim
                    # merge dummy spatial dim to channel dim to share the dummy-2D transform
                    x = rearrange(x, 'c d ... -> (c d) ...')
                if padding_mode == GridSamplePadMode.ZEROS:
                    min_v = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
                    x -= min_v
                x = torch_f.grid_sample(
                    x[None],
                    patch_grid,
                    mode,
                    padding_mode,
                    align_corners=True,
                )[0]
                if padding_mode == GridSamplePadMode.ZEROS:
                    x += min_v
                if self.dummy_dim is not None:
                    x = rearrange(x, '(c d) ... -> c d ...', d=dummy_crop_size)
                    x = x.movedim(1, self.dummy_dim + 1)
                if hasattr(x, 'meta'):
                    x.meta['crop center'] = self.center
                    x.meta['rotate'] = self.id_rotate if rotate_params is None else np.array(rotate_params).tolist()
                    x.meta['scale'] = self.id_scale if scale_params is None else np.array(scale_params).tolist()
                d[key] = x
        else:
            crop = monai_t.Compose([
                monai_t.SpatialCrop(center, self.crop_size),
                monai_t.SpatialPad(self.crop_size),  # note: this does not guarantee the center
            ])
            for key in self.key_iterator(d):
                x = crop(d[key])
                if hasattr(x, 'meta'):
                    x.meta['crop center'] = self.center
                    x.meta['rotate'] = self.id_rotate
                    x.meta['scale'] = self.id_scale
                d[key] = x

        return d

# compatible with monai_t.create_grid, without normalization
def create_patch_grid(
    spatial_size: Sequence[int],
    center: Sequence[int],
    patch_size: Sequence[int],
    device: torch.device | None = None,
    dtype=torch.float32,
):
    spatial_size = np.array(spatial_size)
    center = np.array(center)
    patch_size = np.array(patch_size)
    front_shift = patch_size >> 1
    back_shift = patch_size - front_shift - 1
    start = center - front_shift - (spatial_size - 1) / 2
    end = center + back_shift - (spatial_size - 1) / 2
    ranges = [
        torch.linspace(
            start[i], end[i], patch_size[i],
            device=device,
            dtype=get_equivalent_dtype(dtype, torch.Tensor),
        )
        for i in range(len(patch_size))
    ]
    coords = meshgrid_ij(*ranges)
    return torch.stack([*coords, torch.ones_like(coords[0])])
