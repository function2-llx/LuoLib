from dataclasses import dataclass, field

from monai.utils import StrEnum
from umei.args import AugArgs, UMeIArgs

class MaskValue(StrEnum):
    PARAM = "param"
    UNIFORM = "uniform"
    DIST = "dist"
    # NORMAL = "normal"

@dataclass
class SnimArgs(UMeIArgs, AugArgs):
    mask_ratio: float = field(default=0.75)
    mask_block_shape: list[int] = field(default=None)
    norm_pix_loss: bool = field(default=False)
    val_size: int = field(default=2)
    visible_factor: float = field(default=1e-3)
    mask_value: MaskValue = field(default='uniform', metadata={'choices': [v.value for v in MaskValue]})
    num_sanity_val_steps: int = field(default=-1)
    modality: str = field(default='ct', metadata={'choices': ['ct', 'mri']})
    uper_channels: int = field(default=512)
    uper_pool_scales: list[int] = field(default=None)
    loss: str = field(default='l2', metadata={'choices': ['l1', 'l2']})

    @property
    def p_block_shape(self):
        return tuple(
            block_size // patch_size
            for block_size, patch_size in zip(self.mask_block_shape, self.vit_patch_shape)
        )

    def __post_init__(self):
        super().__post_init__()
        for mask_block_size, vit_patch_size in zip(self.mask_block_shape, self.vit_patch_shape):
            assert mask_block_size % vit_patch_size == 0

        if self.norm_pix_loss:
            assert self.norm_intensity

    num_input_channels: int = field(default=1)
