from dataclasses import field

from umei.args import UMeIArgs

class SwinMAEArgs(UMeIArgs):
    num_input_channels: int = field(default=1)
