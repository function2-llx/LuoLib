import torch
from torch import nn

@torch.no_grad()
def grad_norm(m: nn.Module):
    norm = 0.
    for p in m.parameters():
        if p.grad is not None:
            grad = p.grad.flatten()
            norm += torch.dot(grad, grad)
    return norm ** 0.5
