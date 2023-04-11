from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, SegArgs

@dataclass
class BTCVArgs(AugArgs, SegArgs):
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/btcv'))
    conf_root: Path = field(default=Path('conf/btcv'))
    data_ratio: float = field(default=1)
    # val_post: bool = field(default=False, metadata={'help': 'whether to perform post-processing during validation'})

    @property
    def num_seg_classes(self) -> int:
        return 14

    @property
    def num_input_channels(self) -> int:
        return 1
