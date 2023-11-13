import torch
from torch import nn

__all__ = [
    'Sequential',
]

class Sequential(nn.Sequential):
    def __init__(self, modules: list[nn.Module]):
        # https://github.com/omni-us/jsonargparse/issues/407
        super().__init__(*modules)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor] | None:
        args = ()
        feature_maps = None
        for module in self:
            feature_maps = module(*args, x)
            args = (feature_maps, )
        return feature_maps
