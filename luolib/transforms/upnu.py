# see mt.utils_pytorch_numpy_unification

import torch

__all__ = [
    'quantile'
]

def quantile(x: torch.Tensor, q: float, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    # workaround for https://github.com/pytorch/pytorch/issues/64947
    assert 0 <= q <= 1
    if dim is None:
        x = x.view(-1)
        k = round(x.numel() * q)
        dim = 0
    else:
        k = round(x.shape[dim] * q)
    if k == 0:
        k = 1
    return x.kthvalue(k, dim, keepdim).values
