from dataclasses import dataclass, field
from pathlib import Path

from umei.args import UMeIArgs

@dataclass
class AmosArgs(UMeIArgs):
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/amos'))
    conf_root: Path = field(default=Path('conf/amos'))
    num_crop_samples: int = field(default=4)
    use_test_fold: bool = field(default=False)
    per_device_eval_batch_size: int = field(default=1)  # unable to batchify the whole image without resize
    sw_batch_size: int = field(default=16)

    @property
    def num_seg_classes(self) -> int:
        return 16

    @property
    def num_input_channels(self) -> int:
        return 1
