from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import torch

import monai
from monai import transforms as monai_t
from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor, SequenceStr
from monai.networks.utils import meshgrid_ij
from monai.transforms import Randomizable, create_rotate, create_scale, create_translate
from monai.utils import GridSampleMode, GridSamplePadMode, TransformBackends, ensure_tuple_rep, get_equivalent_dtype

from umei.types import tuple2_t

class SpatialSquarePad(monai.transforms.SpatialPad):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(-1, **kwargs)

    def __call__(self, data: NdarrayOrTensor, **kwargs):
        size = max(data.shape[1:3])
        self.spatial_size = [size, size, -1]
        return super().__call__(data, **kwargs)

# FIXME: set padder
class SpatialSquarePadD(monai.transforms.SpatialPadD):
    def __init__(
        self,
        keys: KeysCollection,
        **kwargs,
    ) -> None:
        super().__init__(keys, -1, **kwargs)

class SpatialRangeGenerator(monai_t.Randomizable):
    def __init__(
        self,
        rand_range: Sequence[tuple2_t[float]] | tuple2_t[float],
        prob: Sequence[float] | float = 1.,
        default: float = 0.,
        repeat: int | None = None,
        dtype: DtypeLike = np.float32,
    ):
        super().__init__()
        self.rand_range = rand_range = np.array(rand_range)
        self.prob = prob = np.array(prob)
        self.default = default
        self.dtype = dtype
        if rand_range.ndim == 1:
            assert repeat is not None
            self.repeat = repeat
            # shared transform
            assert prob.ndim == 0
        else:
            # independent transform
            self.repeat = rand_range.shape[0]
            assert repeat is None or repeat == self.repeat
            if prob.ndim > 0:
                # independent prob
                assert prob.shape[0] == rand_range.shape[0]

    def randomize(self, *_, **__):
        match self.rand_range.ndim, self.prob.ndim:
            case 1, 0:
                # shared transform & prob
                if self.R.uniform() >= self.prob:
                    return None
                return np.repeat(self.R.uniform(*self.rand_range), self.repeat)
            case _, 0:
                # independent transform, shared prob
                if self.R.uniform() >= self.prob:
                    return None
                return np.array([self.R.uniform(*r) for r in self.rand_range])
            case _, _:
                # independent transform, independent prob
                do_transform = self.R.uniform(size=self.repeat) < self.prob
                if np.any(do_transform):
                    return np.array([
                        self.R.uniform(*r) if do
                        else self.default
                        for r, do in zip(self.rand_range, do_transform)
                    ])
                else:
                    return None

    def __call__(self, *args, **kwargs):
        ret = self.randomize()
        if ret is not None:
            ret = ret.astype(self.dtype)
        return ret

class RandAffineCropD(monai_t.RandomizableTrait, monai_t.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        crop_size: Sequence[int],
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.ZEROS,
        allow_missing_keys: bool = False,
        *,
        center_generator: Callable[[Mapping[Hashable, torch.Tensor]], Sequence[int]],
        rotate_generator: Callable[[Mapping[Hashable, torch.Tensor]], Sequence[float] | None],
        scale_generator: Callable[[Mapping[Hashable, torch.Tensor]], Sequence[float] | None],
    ):
        monai_t.MapTransform.__init__(self, keys, allow_missing_keys)
        self.crop_size = crop_size
        self.rotate_generator = rotate_generator
        self.scale_generator = scale_generator
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.center_generator = center_generator

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)
        spatial_size = np.array(d[self.first_key(d)].shape[1:])
        self.center = center = np.array(self.center_generator(d))
        patch_grid = create_patch_grid(spatial_size, center, self.crop_size)
        self.rotate_params = rotate_params = self.rotate_generator(d)
        self.scale_params = scale_params = self.scale_generator(d)
        if rotate_params is not None or scale_params is not None:
            spatial_dims = len(spatial_size)
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
            patch_grid = patch_grid.view(spatial_dims + 1, -1)
            patch_grid = affine @ patch_grid
            patch_grid = patch_grid.view(spatial_dims + 1, *self.crop_size)

            resample = monai_t.Resample(dtype=None)
            for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
                d[key] = resample(d[key], patch_grid, mode, padding_mode)
        else:
            crop = monai_t.SpatialCrop(center, self.crop_size)
            for key in self.key_iterator(d):
                d[key] = crop(d[key])
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

class RandSpatialCenterGeneratorD(monai_t.Randomizable):
    def __init__(
        self,
        ref_key: str,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
    ):
        self.ref_key = ref_key
        self.dummy_rand_cropper = monai_t.RandSpatialCrop(roi_size, max_roi_size, random_center, random_size)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        super().set_random_state(seed, state)
        self.dummy_rand_cropper.set_random_state(seed, state)
        return self

    def randomize(self, spatial_size: Sequence[int]):
        self.dummy_rand_cropper.randomize(spatial_size)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> tuple[int, ...]:
        spatial_size = data[self.ref_key].shape[1:]
        self.randomize(spatial_size)
        if self.dummy_rand_cropper.random_center:
            slices = self.dummy_rand_cropper._slices
        else:
            slices = monai_t.CenterSpatialCrop(self.dummy_rand_cropper._size).compute_slices(spatial_size)
        return tuple(
            s.start + s.stop >> 1
            for s in slices
        )

class RandCenterGeneratorByLabelClassesD(monai_t.Randomizable):
    def __init__(
        self,
        label_key: str,
        roi_size: Sequence[int] | int,
        ratios: list[float | int] | None = None,
        num_classes: int | None = None,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        indices_key: str | None = None,
        allow_smaller: bool = False,
    ) -> None:
        self.label_key = label_key
        self.image_key = image_key
        self.indices_key = indices_key
        self.dummy_rand_cropper = monai_t.RandCropByLabelClasses(
            roi_size,
            ratios,
            num_classes=num_classes,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        super().set_random_state(seed, state)
        self.dummy_rand_cropper.set_random_state(seed, state)
        return self

    def randomize(
        self, label: torch.Tensor, indices: list[NdarrayOrTensor] | None = None, image: torch.Tensor | None = None
    ) -> None:
        self.dummy_rand_cropper.randomize(label=label, indices=indices, image=image)

    def __call__(self, data: Mapping[Hashable, Any]):
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        indices = d.pop(self.indices_key, None) if self.indices_key is not None else None
        self.randomize(label, indices, image)

        return self.dummy_rand_cropper.centers[0]
