import os
from typing import Union

from einops.layers.torch import Rearrange

from .argparse import UMeIParser
from .logger import MyWandbLogger
from .enums import DataSplit, DataKey

PathLike = Union[str, bytes, os.PathLike]

class ChannelFirst(Rearrange):
    def __init__(self):
        super().__init__('n ... c -> n c ...')

class ChannelLast(Rearrange):
    def __init__(self):
        super().__init__('n c ... -> n ... c')
