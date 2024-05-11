import torch
import torch.nn.functional as nnf

__all__ = [
    'bce_with_binary_label',
    'bce_pos',
    'bce_neg',
]

def bce_with_binary_label(input: torch.Tensor, target: torch.Tensor | None):
    if target is None:
        return bce_neg(input)
    assert not target.is_floating_point()
    bce = -nnf.logsigmoid(input)
    neg_mask = target == 0
    bce[neg_mask] += input[neg_mask]
    return bce

def bce_pos(input: torch.Tensor):
    return -nnf.logsigmoid(input)

def bce_neg(input: torch.Tensor):
    """
    -ln(1 - s(x)) = -ln(1 - 1 / (1 + exp(-x))) = -ln(exp(-x) / (1 + exp(-x))) = x - ln s(x)
    """
    return input - nnf.logsigmoid(input)
