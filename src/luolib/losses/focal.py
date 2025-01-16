import torch
from torch.nn import functional as nnf

from monai.losses.focal_loss import sigmoid_focal_loss as _monai_sigmoid_focal_loss

__all__ = [
    'bce_with_binary_label',
    'bce_pos',
    'bce_neg',
    'sigmoid_focal_loss',
]

def sigmoid_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    """Just convert the target to float by default and make everyone happy
    """
    return _monai_sigmoid_focal_loss(input, target.float(), gamma, alpha)

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
