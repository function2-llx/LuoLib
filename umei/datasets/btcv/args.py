from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, CTArgs, SegArgs, UMeIArgs

@dataclass
class BTCVArgs(UMeIArgs, AugArgs, CTArgs, SegArgs):
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/btcv'))
    conf_root: Path = field(default=Path('conf/btcv'))
    per_device_eval_batch_size: int = field(default=1)  # unable to batchify the whole image without resize
    # val_post: bool = field(default=False, metadata={'help': 'whether to perform post-processing during validation'})

    @property
    def num_seg_classes(self) -> int:
        return 14
