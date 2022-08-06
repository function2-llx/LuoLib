from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, SegArgs, UMeIArgs

@dataclass
class BTCVArgs(UMeIArgs, AugArgs, SegArgs):
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/btcv'))
    conf_root: Path = field(default=Path('conf/btcv'))
    eval_epochs: int = field(default=10)
    # val_post: bool = field(default=False, metadata={'help': 'whether to perform post-processing during validation'})

    @property
    def num_seg_classes(self) -> int:
        return 14

    @property
    def num_input_channels(self) -> int:
        return 1
