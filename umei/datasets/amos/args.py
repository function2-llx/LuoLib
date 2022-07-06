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
    sw_batch_size: int = field(default=8)
    val_sw_overlap: float = field(default=0.25)
    val_post: bool = field(default=False, metadata={'help': 'whether to perform post-processing during validation'})
    task_id: int = field(default=2, metadata={'choices': [1, 2]})
    a_min: float = field(default=None)
    a_max: float = field(default=None)
    norm_intensity: bool = field(default=False)
    warmup_epochs: int = field(default=50)
    use_monai: bool = field(default=False, metadata={'help': 'run validation for models produced by '
                                                             'official monai implementation'})
    crop: str = field(default='cls', metadata={'choices': ['cls', 'pn']})
    flip_p: float = field(default=0.2)
    rotate_p: float = field(default=0.2)
    scale_p: float = field(default=0.1)
    shift_p: float = field(default=0.1)
    dice_dr: float = field(default=1e-6)
    dice_nr: float = field(default=0)
    dice_include_background: bool = field(default=False)
    squared_dice: bool = field(default=False)

    @property
    def num_seg_classes(self) -> int:
        return 16

    @property
    def num_input_channels(self) -> int:
        return 1
