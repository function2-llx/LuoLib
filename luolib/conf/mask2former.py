from dataclasses import dataclass

from .base import ExpConfBase, ModelConf

@dataclass(kw_only=True)
class Mask2FormerConf(ExpConfBase):
    pixel_decoder: ModelConf
    transformer_decoder: ModelConf
    num_fg_classes: int
    num_train_points: int
    eos_coef: float = 0.1
    oversample_ratio: float = 3.
    importance_sample_ratio: float = 0.75
    cost_class: float = 5.
    cost_dice: float = 5.
    cost_bce: float = 2.
