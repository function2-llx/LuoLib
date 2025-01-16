from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import torch

from monai import transforms as mt
from monai.config import KeysCollection, NdarrayOrTensor
from monai.utils import convert_to_dst_type, convert_to_numpy, fall_back_tuple

__all__ = [
    'FilterClassIndicesD',
]

def filter_crop_centers(
    centers: NdarrayOrTensor,
    crop_size: Sequence[int] | int,
    img_shape: Sequence[int],
) -> NdarrayOrTensor:
    """
    Utility to filter the crop centers that are compatible with the image size and crop size.

    Args:
        centers: (spatial_dims, n) pre-computed crop centers of every dim, will correct based on the valid region.
        crop_size: spatial size of the ROIs to be sampled.
        img_shape: spatial shape of the original label data to compare with ROI.
    Returns:
        valid centers (spatial_dims, n_valid)
    """
    crop_size = fall_back_tuple(crop_size, default=img_shape)
    if any(np.subtract(img_shape, crop_size) < 0):
        raise ValueError(
            "The size of the proposed random crop ROI is larger than the image size, "
            f"got ROI size {crop_size} and label image size {img_shape} respectively."
        )

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(crop_size, 2)
    # add 1 for random
    valid_end = np.subtract(img_shape + np.array(1), crop_size / np.array(2)).astype(np.uint16)
    valid_start, *_ = convert_to_dst_type(valid_start, centers)
    valid_end, *_ = convert_to_dst_type(valid_end, centers)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    # need this because np.random.randint does not work with same start and end
    valid_end[valid_start == valid_end] += 1
    valid_mask = (valid_start[:, None] <= centers) & (centers <= valid_end[:, None] - 1)
    if isinstance(valid_mask, np.ndarray):
        valid_mask = valid_mask.all(axis=0)
    else:
        assert isinstance(valid_mask, torch.Tensor)
        valid_mask = valid_mask.all(dim=0)
    return centers[:, valid_mask]

class FilterClassIndicesD(mt.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        crop_size: Sequence[int] | int,
        shape_ref_key: Hashable,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.crop_size = crop_size
        self.shape_ref_key = shape_ref_key

    def __call__(self, data: Mapping[Hashable, ...]):
        data = dict(data)
        shape = data[self.shape_ref_key].shape[1:]
        for key in self.key_iterator(data):
            for i, indices in enumerate(data[key]):
                centers = mt.unravel_index(indices, shape)
                filtered = filter_crop_centers(centers, self.crop_size, shape)
                # TODO: handle torch.Tensor
                raveled = np.ravel_multi_index(convert_to_numpy(filtered), shape)
                data[key][i] = convert_to_dst_type(raveled, indices)
