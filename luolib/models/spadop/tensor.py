from typing import Iterable, Sequence

import torch

__all__ = [
    'SpatialTensor',
]

class SpatialTensor(torch.Tensor):
    # gradient checkpointing will not work for this class
    # https://github.com/pytorch/pytorch/issues/105644

    @staticmethod
    def __new__(cls, x, aniso_d: int, num_downsamples: int = 0, *args, **kwargs):
        return torch.as_tensor(x, *args, **kwargs).as_subclass(SpatialTensor)

    def __init__(self, _x, aniso_d: int, num_downsamples: int = 0, *_args, **_kwargs):
        """
        Args:
            aniso_d: degree of anisotropy, i.e., log2(spacing_slice / spacing_in-plane)
            num_downsamples:  how many downsamples have been performed in-plane
        """
        super().__init__()
        self.aniso_d = aniso_d
        self.num_downsamples = num_downsamples

    def __repr__(self, *args, **kwargs):
        aniso_d = getattr(self, 'aniso_d', 'missing')
        num_downsamples = getattr(self, 'num_downsamples', 'missing')
        return f'{super().__repr__()}\nshape={self.shape}, aniso_d={aniso_d}, num_downsamples={num_downsamples}'

    @property
    def num_pending_hw_downsamples(self):
        return max(self.aniso_d - self.num_downsamples, 0)

    @property
    def can_downsample_d(self) -> bool:
        return self.num_pending_hw_downsamples == 0

    @property
    def num_remained_d_upsamples(self) -> int:
        return max(self.num_downsamples - self.aniso_d, 0)

    @property
    def can_upsample_d(self) -> bool:
        return self.num_remained_d_upsamples > 0

    @classmethod
    def find_meta_ref_iter(cls, iterable: Iterable):
        for x in iterable:
            if (ret := cls.find_meta_ref(x)) is not None:
                return ret
        return None

    @classmethod
    def find_meta_ref(cls, obj):
        match obj:
            case SpatialTensor():
                return obj
            case tuple() | list():
                return cls.find_meta_ref_iter(obj)
            case dict():
                return cls.find_meta_ref_iter(obj.values())
            case _:
                return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, Sequence):
            unpack = False
        else:
            unpack = True
            ret = [ret]
        if any(isinstance(x, SpatialTensor) for x in ret) and (
            (meta_ref := cls.find_meta_ref(args)) is not None
            or (meta_ref := cls.find_meta_ref(kwargs)) is not None
        ):
            meta_ref: SpatialTensor
            for x in ret:
                if isinstance(x, SpatialTensor):
                    x.aniso_d = meta_ref.aniso_d
                    x.num_downsamples = meta_ref.num_downsamples
        if unpack:
            ret = ret[0]
        return ret

    def as_tensor(self):
        return self.as_subclass(torch.Tensor)
