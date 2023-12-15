import torch

__all__ = [
    'quantile'
]

def quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    # workaround for https://github.com/pytorch/pytorch/issues/64947
    assert 0 <= q <= 1
    k = round(x.numel() * q)
    return x.view(-1).kthvalue(k).values
