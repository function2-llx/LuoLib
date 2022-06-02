from dataclasses import dataclass, field
from pathlib import Path

from umei.utils import UMeIArgs

@dataclass
class Stoic2021Args(UMeIArgs):
    monitor: str = field(default='auc-severity')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/stoic2021'))
    cls_weight: float = field(default=5, metadata={'help': 'classification weight for positive samples'})
    lr_reduce_factor: float = field(default=0.2)
    # model_ckpts: list[str] = field(default_factory=list)

    @property
    def num_cls_classes(self) -> int:
        return 3

    @property
    def clinical_feature_size(self) -> int:
        return 3

    @property
    def num_input_channels(self) -> int:
        # For CT
        return 2
