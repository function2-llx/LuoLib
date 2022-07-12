from dataclasses import field

import numpy as np

from umei.args import UMeIArgs

class SwinMAEArgs(UMeIArgs):
    num_input_channels: int = field(default=1)
    mask_ratio: float = field(default=0.75)
    mask_size: int = field(default=16)

    @property
    def mask_num(self) -> int:
        return int(np.log(1 - self.mask_ratio) / np.log(1 - self.mask_size ** 3 / np.product(self.sample_shape)))
