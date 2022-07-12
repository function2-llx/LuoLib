from dataclasses import dataclass, field

import numpy as np

from umei.args import UMeIArgs

@dataclass
class SwinMAEArgs(UMeIArgs):
    num_input_channels: int = field(default=1)
    mask_ratio: float = field(default=0.75)
    mask_block_shape: list[int] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        for mask_block_size, vit_patch_size in zip(self.mask_block_shape, self.vit_patch_shape):
            assert mask_block_size % vit_patch_size == 0
