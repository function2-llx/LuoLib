import os
from typing import Union

from .argparse import UMeIParser
from .logger import MyWandbLogger
from .enums import DataSplit, DataKey

PathLike = Union[str, bytes, os.PathLike]
