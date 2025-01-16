import torch

__all__ = [
    'zero_loss',
]

def zero_loss(*args: torch.Tensor | None) -> torch.Tensor:
    """
    make both loss and grad contribution zero
    this can be useful to make DDP work
    """
    zero = []
    for x in args:
        if x is None:
            continue
        x = x.view(-1)
        zero.append(torch.dot(x, torch.zeros_like(x)))
    if len(zero) > 0:
        zero = torch.stack(zero)
        zero = torch.dot(zero, torch.zeros_like(zero))
    else:
        zero = torch.tensor(0.)
    return zero
