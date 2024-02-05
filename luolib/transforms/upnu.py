# see mt.utils_pytorch_numpy_unification

import torch

__all__ = [
    'quantile'
]

def quantile(x: torch.Tensor, q: float, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    # workaround for https://github.com/pytorch/pytorch/issues/64947
    if dim is None:
        x = x.view(-1)
        k = round(x.numel() * q)
    else:
        k = round(x.shape[dim] * q)
    assert 0 <= q <= 1
    return x.kthvalue(k, dim, keepdim).values
