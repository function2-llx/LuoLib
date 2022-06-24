from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML
from transformers import TrainingArguments

from umei.utils import PathLike, UMeIParser

yaml = YAML()

@dataclass
class UMeIArgs(TrainingArguments):
    exp_name: str = field(default=None)
    num_nodes: int = field(default=1)
    output_dir: Path = field(default=None)
    patience: int = field(default=5)
    sample_size: int = field(default=144)
    sample_slices: int = field(default=160)
    spacing: list[float] = field(default=None)
    vit_patch_size: int = field(default=8)
    vit_hidden_size: int = field(default=768)
    base_feature_size: int = field(default=None, metadata={'help': 'feature size for the first feature map'
                                                                   'assume feature size * 2 each layer'})
    num_seg_heads: int = field(default=1)
    cls_loss_factor: float = field(default=1)
    seg_loss_factor: float = field(default=1)
    img_key: str = field(default='img')
    mask_key: str = field(default='mask')
    seg_key: str = field(default='seg')
    cls_key: str = field(default='cls')
    clinical_key: str = field(default='clinical')
    conf_root: Path = field(default=Path('conf'))
    output_root: Path = field(default=Path('output'))
    amp: bool = field(default=True)
    dataloader_num_workers: int = field(default=multiprocessing.cpu_count() // 2)
    monitor: str = field(default=None)
    monitor_mode: str = field(default=None)
    lr_reduce_factor: float = field(default=0.2)
    num_folds: int = field(default=5)
    use_test_fold: bool = field(default=True)
    num_runs: int = field(default=3)
    encoder: str = field(default=None, metadata={'choices': ['resnet', 'vit', 'swt']})
    decoder: str = field(default=None, metadata={'choices': ['cnn', 'sunetr']})
    model_depth: int = field(default=50)
    pretrain_path: Optional[Path] = field(default=None)
    decoder_pretrain_path: Optional[Path] = field(default=None)
    resnet_shortcut: str = field(default=None, metadata={'choices': ['A', 'B']})
    resnet_conv1_size: int = field(default=7)
    resnet_conv1_stride: int = field(default=2)
    resnet_layer1_stride: int = field(default=1)
    fold_ids: list[int] = field(default=None)
    ddp_find_unused_parameters: bool = field(default=False)
    on_submit: bool = field(default=False)
    log_offline: bool = field(default=False)
    use_monai: bool = field(default=None)

    @property
    def sample_shape(self) -> tuple[int, int, int]:
        return self.sample_size, self.sample_size, self.sample_slices

    @property
    def precision(self):
        return 16 if self.amp else 32

    @property
    def num_input_channels(self) -> int:
        raise NotImplementedError

    @property
    def num_cls_classes(self) -> Optional[int]:
        return None

    @property
    def clinical_feature_size(self) -> int:
        return 0

    # include background
    @property
    def num_seg_classes(self) -> Optional[int]:
        return None

    def __post_init__(self):
        # disable super().__post__init__ or `output_dir` will restore str type specified in the base class
        # as well as a lot of strange things happens
        # super().__post_init__()
        # self.output_dir = Path(self.output_dir)
        if self.fold_ids is None:
            self.fold_ids = list(range(self.num_folds))
        else:
            for i in self.fold_ids:
                assert 0 <= i < self.num_folds

    @classmethod
    def from_yaml_file(cls, yaml_path: PathLike):
        parser = UMeIParser(cls, use_conf=False)
        conf = yaml.load(Path(yaml_path))
        argv = UMeIParser.to_cli_options(conf)
        args, _ = parser.parse_known_args(argv)
        # want to return `Self` type
        return cls(parser.parse_dict(vars(args))[0])
