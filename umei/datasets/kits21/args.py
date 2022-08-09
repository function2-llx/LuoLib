from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, SegArgs

@dataclass
class KiTS21Args(AugArgs, SegArgs):
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/kits21'))
    conf_root: Path = field(default=Path('conf/kits21'))
    fold_ids: list[int] = field(default_factory=lambda: [0])

    @property
    def num_seg_classes(self) -> int:
        return 4

    @property
    def num_input_channels(self) -> int:
        return 1
