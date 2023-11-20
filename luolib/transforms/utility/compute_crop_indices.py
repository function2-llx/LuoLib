from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as nnf

from monai import transforms as mt
from monai.config import KeysCollection, NdarrayOrTensor
from monai.networks import one_hot
from monai.utils import convert_to_dst_type, convert_to_tensor, fall_back_tuple

__all__ = [
    'sliding_window_sum',
    'ComputeCropIndicesD',
]

def sliding_window_sum(x: torch.Tensor, window_size: Sequence[int]) -> torch.LongTensor:
    """
    Be careful about value overflow
    """
    window_size = np.array(window_size)
    ret_spatial_shape = x.shape[1:] - window_size + 1
    assert (ret_spatial_shape > 0).all()
    spatial_dims = len(window_size)

    # multi-dimensional partial sum
    s = x.long()
    for i in range(1, spatial_dims + 1):
        s = s.cumsum(i)
    s = nnf.pad(s, (1, 0) * spatial_dims)
    ret = s.new_zeros(x.shape[0], *ret_spatial_shape)

    def dfs(dim: int, index: tuple[slice, ...], sign: int):
        nonlocal ret
        if dim == spatial_dims:
            ret += sign * s[:, *index]
            return
        dfs(dim + 1, index + (slice(window_size[dim], None), ), sign)
        dfs(dim + 1, index + (slice(None, -window_size[dim]), ), -sign)

    dfs(0, (), 1)
    return ret

def compute_crop_indices(
    label: NdarrayOrTensor,
    crop_size: Sequence[int] | int,
    th_abs: float,
    th_rel: float,
    num_classes: int | None = None,
    include_background: bool = False,
) -> list[NdarrayOrTensor]:
    mt.check_non_lazy_pending_ops(label, name="compute_crop_indices")
    label_t: torch.Tensor = convert_to_tensor(label)
    if label.shape[0] == 1:
        if num_classes is None:
            raise ValueError("channels==1 indicates not using One-Hot format label, must provide ``num_classes``.")
        label_t = one_hot(label_t, num_classes, dim=0, dtype=torch.bool)
        if not include_background:
            label_t = label_t[1:]
    spatial_size = label_t.shape[1:]
    crop_size = fall_back_tuple(crop_size, spatial_size)
    s = sliding_window_sum(label_t, crop_size)
    half: np.ndarray = np.floor_divide(crop_size, 2)
    # exclude invalid suffix & prefix patch (on any dimension) that cannot be completely contained within the image
    label_t = label_t[:, *(slice(None, s) for s in spatial_size - (crop_size - half) + 1)]
    label_t = label_t[:, *(slice(p, None) for p in half)]
    ret = []
    half_t = s.new_tensor(half)
    ravel_multiplier = s.new_tensor(spatial_size[-1:0:-1]).cumprod(dim=0).flip(dims=(0, ))
    for c in range(label_t.shape[0]):
        fg_s = s[c][s[c] > 0]
        if len(fg_s) == 0:
            ret.append(torch.empty(0))
            continue
        fg_s = fg_s.float()
        # min_s: 2.1 & 2.2
        min_s = min(mt.percentile(fg_s, 99.5) * th_abs, mt.percentile(fg_s, 100 * th_rel))
        valid_mask = label_t[c] | (s[c] >= min_s)
        centers = valid_mask.nonzero() + half_t
        # manually ravel_multi_index, make MONAI happy
        ret.append((ravel_multiplier * centers[:, :-1]).sum(dim=1) + centers[:, -1])
    dtype = torch.int64 if isinstance(label, torch.Tensor) else np.int64
    for i, indices in enumerate(ret):
        ret[i], *_ = convert_to_dst_type(indices, label, dtype=dtype)
    return ret

class ComputeCropIndicesD(mt.MapTransform):
    """
    a valid patch must be completely contained in the image, and at least one of the following holds:
      1. center is foreground
      2. 考虑所有至少存在一个前景点的 patch，以下至少一条成立：
        2.1 与前景点数为 99.5% 分位数的 patch 相比，至少有其 th_abs 的点数
        2.2 比 th_rel 的 patch 的前景点数多
    """
    def __init__(
        self,
        keys: KeysCollection,
        crop_size: int | Sequence[int],
        th_abs: float = 0.7,
        th_rel: float = 0.7,
        indices_postfix: str = "_indices",
        num_classes: int | None = None,
        include_background: bool = False,
        cache_path_base_key: Hashable | None = None,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            cache_path_base_key: the key to get the cache path base. This class won't validate the cache, make sure to
            manually clean the cache when changing parameters (e.g., crop_size, th_*)
        """
        super().__init__(keys, allow_missing_keys)
        self.indices_postfix = indices_postfix
        self.crop_size = crop_size
        self.th_abs = th_abs
        self.th_rel = th_rel
        self.num_classes = num_classes
        self.include_background = include_background
        self.cache_path_base_key = cache_path_base_key

    def __call__(self, data: Mapping):
        data = dict(data)
        cache_path_base = None if self.cache_path_base_key is None else Path(data[self.cache_path_base_key])
        name = None if cache_path_base is None else cache_path_base.name
        for key in self.key_iterator(data):
            indices_key = f'{key}{self.indices_postfix}'
            cache_path = None if name is None else cache_path_base.with_name(f'{name}_{indices_key}.pkl')
            if cache_path is not None and cache_path.exists():
                indices = pd.read_pickle(cache_path)
            else:
                indices = compute_crop_indices(
                    data[key],
                    self.crop_size, self.th_abs, self.th_rel,
                    self.num_classes, self.include_background,
                )
                if cache_path is not None:
                    pd.to_pickle(indices, cache_path)
            data[indices_key] = indices

        return data
