import torch

__all__ = [
    'zero_loss',
]

def zero_loss(*args: torch.Tensor):
    """
    make both loss and grad contribution zero
    this can be useful to make DDP work
    """
    zero = []
    for x in args:
        x = x.view(-1)
        zero.append(torch.dot(x, torch.zeros_like(x)))
    zero = torch.stack(zero)
    return torch.dot(zero, torch.zeros_like(zero))
