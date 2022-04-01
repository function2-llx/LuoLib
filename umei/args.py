from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path

from transformers import TrainingArguments

@dataclass
class UMeIArgs(TrainingArguments):
    exp_name: str = field(default=None)
    log: bool = field(default=True)
    output_dir: Path = field(default=None)
    patience: int = field(default=5)
    cls_loss_factor: float = field(default=1)
    seg_loss_factor: float = field(default=1)
    img_key: str = field(default='img')
    seg_key: str = field(default='seg')
    cls_key: str = field(default='cls')
    conf_root: Path = field(default=Path('conf'))
    output_root: Path = field(default=Path('output'))
    amp: bool = field(default=True)
    dataloader_num_workers: int = field(default=multiprocessing.cpu_count())
    monitor: str = field(default='cls_loss')

    @property
    def precision(self):
        return 16 if self.amp else 32

    @property
    def num_input_channels(self) -> int:
        # For CT
        return 1

    @property
    def num_cls_classes(self) -> int:
        return 2

    @property
    def num_seg_classes(self) -> int:
        pass
