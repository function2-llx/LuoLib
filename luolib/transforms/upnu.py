# see mt.utils_pytorch_numpy_unification

import torch

__all__ = [
    'quantile'
]

def quantile(x: torch.Tensor, q: float, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    """Compute the q-th quantile of the input tensor along the specified dimension.
    
    This is a workaround for PyTorch's quantile size limitation (https://github.com/pytorch/pytorch/issues/64947).
    Uses torch.kthvalue() which is more memory efficient and doesn't have the 2^24 size limitation.
    
    Args:
        x: Input tensor
        q: Quantile to compute, must be between 0 and 1
        dim: Dimension along which to compute the quantile. If None, the tensor is flattened
            before computation and the result is a scalar.
        keepdim: If True, the output tensor has the same dimensions as the input tensor,
            with the reduced dimension having size 1. Only useful when dim is not None.
    
    Returns:
        Tensor containing the q-th quantile values
    
    Raises:
        AssertionError: If q is not between 0 and 1
    """
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
