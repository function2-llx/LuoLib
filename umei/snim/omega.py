from dataclasses import dataclass, field

from monai.utils import StrEnum
from umei.omega import ExpConfBase
from umei.types import tuple3_t

class MaskValue(StrEnum):
    PARAM = "param"
    UNIFORM = "uniform"
    DIST = "dist"
    NORMAL = "normal"

@dataclass(kw_only=True)
class SnimConf(ExpConfBase):
    mask_ratio: float = 0.8
    mask_block_shape: tuple3_t[int]
    mask_patch_size: tuple3_t[int]
    norm_pix_loss: bool = True
    val_size: int = 2
    visible_factor: float = field(default=1e-3)
    mask_value: MaskValue = MaskValue.DIST
    loss: str = field(default='l2', metadata={'choices': ['l1', 'l2']})
    datasets: list[str]

    @property
    def p_block_shape(self):
        return tuple(
            block_size // patch_size
            for block_size, patch_size in zip(self.mask_block_shape, self.mask_patch_size)
        )

    def __post_init__(self):
        for mask_block_size, vit_patch_size in zip(self.mask_block_shape, self.mask_patch_size):
            assert mask_block_size % vit_patch_size == 0

        if self.norm_pix_loss:
            assert self.norm_intensity

        self.datasets = sorted(set(self.datasets))
