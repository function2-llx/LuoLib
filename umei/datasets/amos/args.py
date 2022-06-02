from dataclasses import dataclass, field
from pathlib import Path

from umei.utils import UMeIArgs

@dataclass
class AmosArgs(UMeIArgs):
    monitor: str = field(default=None)
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/amos'))
    conf_root: Path = field(default=Path('conf/amos'))
    crop_num_samples: int = field(default=5)
    use_test_fold: bool = field(default=False)

    @property
    def num_seg_classes(self) -> int:
        return 16

    @property
    def num_input_channels(self) -> int:
        return 1
