import os
from typing import Union

from einops import rearrange
from einops.layers.torch import Rearrange
import torch

from .argparse import UMeIParser
from .enums import DataSplit, DataKey

PathLike = Union[str, bytes, os.PathLike]

class ChannelFirst(Rearrange):
    def __init__(self):
        super().__init__('n ... c -> n c ...')

def channel_first(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'n ... c -> n c ...')

class ChannelLast(Rearrange):
    def __init__(self):
        super().__init__('n c ... -> n ... c')

def channel_last(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'n c ... -> n ... c')
