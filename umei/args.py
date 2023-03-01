from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments

from monai.utils import BlendMode

from umei.utils import PathLike, UMeIParser
from umei.utils.argparse import yaml

@dataclass
class UMeIArgsBase:
    def __post_init__(self):
        pass

@dataclass
class UMeIArgs(UMeIArgsBase, TrainingArguments):
    exp_name: str = field(default=None)
    num_nodes: int = field(default=1)
    output_dir: Path = field(default=None)
    patience: int = field(default=5)
    spatial_dims: int = field(default=3)
    # sample_size: int = field(default=None)
    # sample_slices: int = field(default=None)
    sample_shape: list[int] = field(default=None)
    num_stages: int = field(default=None)
    spacing: list[float] = field(default=None)
    norm_intensity: bool = field(default=False)
    norm_mean: Optional[float] = field(default=None)    # assert T | None is not Union[T, None]
    norm_std: Optional[float] = field(default=None)
    # a_min: Optional[float] = field(default=None)
    # a_max: Optional[float] = field(default=None)
    a_min: Optional[float] = field(default=None)
    a_max: Optional[float] = field(default=None)
    b_min: float = field(default=0)
    b_max: float = field(default=1)
    vit_hidden_size: int = field(default=768)
    swin_window_size: list[int] = field(default=None)
    vit_patch_shape: list[int] = field(default=None)
    num_heads: list[int] = field(default=None)
    vit_depths: list[int] = field(default=None)
    # vit_stem: str = field(default='linear', metadata={'choices': ['linear', 'conv']})
    vit_conv_stem: bool = field(default=False)
    base_feature_size: int = field(default=None, metadata={'help': 'feature size for the first feature map'
                                                                   'assume feature size * 2 each layer'})
    stem_stride: int = field(default=1)
    num_conv_layers: int = field(default=2)
    conv_norm: str = field(default='instance')
    conv_act: str = field(default='leakyrelu')
    layer_channels: list[int] = field(default=None)
    kernel_sizes: list[int] = field(default=None)
    conv_in_channels: int = field(default=None)
    layer_depths: list[int] = field(default=None)
    num_post_upsamplings: int = field(default=0)
    drop_path_rate: float = field(default=0.)
    encode_skip: bool = field(default=False)
    umei_sunetr_decode_use_in: bool = field(default=True)
    umei_impl: bool = field(default=True)
    use_encoder5: bool = field(default=False)
    num_seg_heads: int = field(default=1)
    cls_loss_factor: float = field(default=1)
    seg_loss_factor: float = field(default=1)
    # img_key: str = field(default='img')
    # mask_key: str = field(default='mask')
    # seg_key: str = field(default='seg')
    # cls_key: str = field(default='cls')
    # clinical_key: str = field(default='clinical')
    conf_root: Path = field(default=Path('conf'))
    output_root: Path = field(default=Path('output'))
    amp: bool = field(default=True)
    cache_dataset_workers: int = field(default=8)
    dataloader_num_workers: int = field(default=8)
    # dataloader_pin_memory: bool = field(default=False)
    monitor: str = field(default=None)
    monitor_mode: str = field(default=None)
    lr_reduce_factor: float = field(default=0.2)
    warmup_epochs: int = field(default=50)
    num_runs: int = field(default=3)
    backbone: str = field(default=None, metadata={'choices': ['resnet', 'vit', 'swt', 'swin']})
    decoder: str = field(default=None, metadata={'choices': ['cnn', 'sunetr', 'conv']})
    model_depth: int = field(default=50)
    pretrain_path: Optional[Path] = field(default=None)
    decoder_pretrain_path: Optional[Path] = field(default=None)
    resnet_shortcut: str = field(default=None, metadata={'choices': ['A', 'B']})
    resnet_conv1_size: int = field(default=7)
    resnet_conv1_stride: int = field(default=2)
    resnet_layer1_stride: int = field(default=1)
    ddp_find_unused_parameters: bool = field(default=False)
    on_submit: bool = field(default=False)
    log_offline: bool = field(default=False)
    train_cache_num: int = field(default=0)
    val_cache_num: int = field(default=0)
    val_empty_cuda_cache: bool = field(default=False)
    eval_epochs: int = field(default=1)
    optim: str = field(default='AdamW')
    optimizer_set_to_none: bool = field(default=True)
    num_epoch_batches: int = field(default=250)   # follow nnunet
    num_sanity_val_steps: int = field(default=5)
    self_ensemble: bool = field(default=False)
    ckpt_path: Path = field(default=None, metadata={'help': 'checkpoint path to resume'})
    resume_log: bool = field(default=True)
    no_resume: bool = field(default=False)
    train_batch_size: int = field(default=2, metadata={'help': 'effective train batch size'})
    eval_batch_size: int = field(default=1, metadata={'help': 'effective eval batch size'})
    gradient_checkpointing: bool = field(default=False)

    @TrainingArguments.n_gpu.getter
    def n_gpu(self):
        return torch.cuda.device_count()

    @property
    def interpolate(self) -> str:
        return {2: 'bilinear', 3: 'trilinear'}[self.spatial_dims]

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

    @property
    def vit_stages(self) -> int:
        return len(self.vit_depths)

    def __post_init__(self):
        # disable super().__post__init__ or `output_dir` will restore str type specified in the base class
        # as well as a lot of strange things happens
        # so make sure UMeIArgs is inherited last
        # super().__post_init__()
        # self.output_dir = Path(self.output_dir)
        assert not self.do_train or self.train_batch_size % torch.cuda.device_count() == 0
        self.per_device_train_batch_size = self.train_batch_size // torch.cuda.device_count()
        assert not self.do_eval or self.eval_batch_size % torch.cuda.device_count() == 0
        self.per_device_eval_batch_size = self.eval_batch_size // torch.cuda.device_count()

        if self.vit_patch_shape is not None:
            for size, patch_size in zip(self.sample_shape, self.vit_patch_shape):
                assert size % patch_size == 0
                assert patch_size >= 2

    @classmethod
    def from_yaml_file(cls, yaml_path: PathLike):
        parser = UMeIParser(cls, use_conf=False)
        conf = yaml.load(Path(yaml_path))
        argv = UMeIParser.to_cli_options(conf)
        args, _ = parser.parse_known_args(argv)
        # want to return `Self` type
        return parser.parse_dict(vars(args))[0]

@dataclass
class CVArgs(UMeIArgsBase):
    num_folds: int = field(default=5)
    # use_test_fold: bool = field(default=False)
    fold_ids: list[int] = field(default=None)

    def __post_init__(self):
        if self.fold_ids is None:
            self.fold_ids = list(range(self.num_folds))
        else:
            for i in self.fold_ids:
                assert 0 <= i < self.num_folds
        super().__post_init__()

@dataclass
class AugArgs(UMeIArgsBase):
    gaussian_noise_p: float = field(default=0.1)
    gaussian_noise_std: float = field(default=0.1)
    flip_p: float = field(default=0.5)
    rotate_p: float = field(default=0.5)
    scale_intensity_factor: float = field(default=0.2)
    scale_intensity_p: float = field(default=0.1)
    shift_intensity_offset: float = field(default=0.1)
    shift_intensity_p: float = field(default=0)

@dataclass
class SegArgs(UMeIArgs):
    # crop: str = field(default='cls', metadata={'choices': ['cls', 'pn'], 'help': 'patch cropping strategy'})
    # crop_pos: int = field(default=1)
    # crop_neg: int = field(default=1)
    fg_oversampling_ratio: tuple[int, int] = field(default=(2, 1))
    dice_dr: float = field(default=1e-5)
    dice_nr: float = field(default=1e-5)
    mc_seg: bool = field(default=False)
    include_background: bool = field(default=False)
    squared_dice: bool = field(default=False)
    post_labels: list[int] = field(default_factory=list)
    sw_batch_size: int = field(default=4)
    sw_overlap: float = field(default=0.25)
    sw_blend_mode: BlendMode = field(default=BlendMode.GAUSSIAN, metadata={'choices': list(BlendMode)})
    per_device_eval_batch_size: int = field(default=1)  # unable to batchify the whole image without resize
    spline_seg: bool = field(default=False)
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    do_tta: bool = field(default=False)
    export: bool = field(default=False)
    test_output_dir: Path = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        # assert self.per_device_eval_batch_size == 1
        if self.mc_seg:
            assert self.include_background


