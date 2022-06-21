import os
from typing import Union

from .argparse import UMeIParser
from .logger import MyWandbLogger

PathLike = Union[str, bytes, os.PathLike]
