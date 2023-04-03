import builtins
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, TypeVar

import omegaconf
from omegaconf import DictConfig, II, OmegaConf
import torch

from monai.config import PathLike
from monai.utils import BlendMode
from umei.types import tuple2_t

# omegaconf: Unions of containers are not supported
@dataclass(kw_only=True)
class AugConf:
    dummy_dim: int | None
    rotate_range: list  # param3_t[tuple2_t[float]]
    rotate_p: Any   # param3_t[float]
    scale_range: list   # param23_t[tuple2_t[float]]
    scale_p: Any    # param23_t[float]
    gaussian_noise_p: float
    gaussian_noise_std: float
    gaussian_smooth_std_range: tuple2_t[float]
    gaussian_smooth_isotropic_prob: float = 1
    gaussian_smooth_p: float
    scale_intensity_factor: float
    scale_intensity_p: float
    shift_intensity_offset: float
    shift_intensity_p: float
    adjust_contrast_range: tuple2_t[float]
    adjust_contrast_p: float
    simulate_low_res_zoom_range: tuple2_t[float]
    simulate_low_res_p: float
    gamma_range: tuple2_t[float]
    gamma_p: float
    flip_p: float = 0.5

@dataclass(kw_only=True)
class DataConf:
    spatial_dims: int = 3
    spacing: tuple | None = None  # tuple2_t[float] | tuple3_t[float]
    data_ratio: float = 1.
    intensity_min: float | None = None
    intensity_max: float | None = None
    norm_intensity: bool
    norm_mean: float | None = None
    norm_std: float | None = None
    scaled_intensity_min: float = 0.
    scaled_intensity_max: float = 1.

@dataclass(kw_only=True)
class OptimizerConf:
    name: str
    lr: float
    weight_decay: float
    kwargs: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class SchedulerConf:
    name: str
    interval: str
    frequency: int = 1
    kwargs: dict

@dataclass(kw_only=True)
class FitConf(DataConf, AugConf):
    monitor: str | None = None
    monitor_mode: str | None
    max_epochs: int | None
    max_steps: int
    val_check_interval: int | None = None
    train_batch_size: int
    optimizer: OptimizerConf
    scheduler: SchedulerConf
    eta_min: float = 1e-6
    optimizer_set_to_none: bool = True
    precision: int = 16
    ddp_find_unused_parameters: bool = False
    num_nodes: int = 1
    gradient_clip_val: float | None = None
    gradient_clip_algorithm: str | None = None

    @property
    def per_device_train_batch_size(self):
        q, r = divmod(self.train_batch_size, torch.cuda.device_count())
        assert r == 0
        return q

@dataclass(kw_only=True)
class RuntimeConf:
    train_cache_num: int = 100
    val_cache_num: int = 100
    num_cache_workers: int = 8
    dataloader_num_workers: int = 16
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int | None = None
    do_train: bool = False
    do_eval: bool = False
    val_empty_cuda_cache: bool = False
    eval_batch_size: int = torch.cuda.device_count()
    resume_log: bool = True
    log_offline: bool = False
    num_sanity_val_steps: int = 5
    save_every_n_epochs: int = 25
    save_every_n_steps: int
    save_top_k: int = 1
    log_every_n_steps: int = 50

@dataclass(kw_only=True)
class ModelConf:
    name: str
    ckpt_path: Path | None = None
    state_dict_key: str | None = None
    key_prefix: str = ''
    kwargs: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class BackboneOptimConf:
    lr: float = II('.optimizer.lr')
    weight_decay: float = II('.optimizer.weight_decay')
    layer_decay: float = 1.

@dataclass(kw_only=True)
class ExpConfBase(FitConf, RuntimeConf):
    backbone: ModelConf
    backbone_optim: BackboneOptimConf
    num_input_channels: int
    sample_shape: tuple  # tuple2_t[int] | tuple3_t[int]
    conf_root: Path = Path('conf')
    output_root: Path = Path('output')
    output_dir: Path
    exp_name: str
    log_dir: Path
    seed: int = 42
    float32_matmul_precision: str = 'high'
    ckpt_path: Path | None = None
    do_tta: bool = False

@dataclass(kw_only=True)
class CrossValConf:
    num_folds: int = 5
    fold_ids: list[int]

@dataclass(kw_only=True)
class ClsExpConf(ExpConfBase):
    num_cls_classes: int

@dataclass(kw_only=True)
class SegInferConf:
    sw_overlap: float = 0.25
    sw_batch_size: int = 16
    sw_blend_mode: BlendMode = BlendMode.GAUSSIAN
    export_seg_pred: bool = False

@dataclass(kw_only=True)
class SegExpConf(SegInferConf, ExpConfBase):
    monitor: str = 'val/dice/avg'
    monitor_mode: str = 'max'
    max_epochs: int | None = None
    max_steps: int = 250000  # nnunet default

    decoder: ModelConf
    # for multi-label task, not including background
    num_seg_classes: int
    num_seg_heads: int = 3
    spline_seg: bool = False
    self_ensemble: bool = False
    dice_include_background: bool = True
    dice_squared: bool = False
    fg_oversampling_ratio: list[float] = (2, 1)  # random vs force fg
    multi_label: bool
    dice_nr: float = 1e-5
    dice_dr: float = 1e-5

T = TypeVar('T')
def parse_node(conf_path: PathLike, conf_type: type[T] = Any) -> T:
    conf_path = Path(conf_path)
    conf_dir = conf_path.parent

    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else conf_dir / path

    conf = OmegaConf.load(conf_path)
    base_confs = []
    for base in conf.pop('_base', []):
        match type(base):
            case builtins.str:
                base_confs.append(parse_node(resolve(base)))
            case omegaconf.DictConfig:
                base_confs.append({
                    k: parse_node(resolve(v))
                    for k, v in base.items()
                })
            case _:
                raise ValueError

    conf = OmegaConf.unsafe_merge(*base_confs, conf)
    if conf_type != Any:
        conf = OmegaConf.structured(conf_type(**conf))
    return conf

exp_conf_t = TypeVar('exp_conf_t', bound=ExpConfBase)
def parse_exp_conf(conf_type: type[exp_conf_t]) -> exp_conf_t:
    argv = sys.argv[1:]
    conf_path = Path(argv[0])
    conf: ExpConfBase | DictConfig = OmegaConf.structured(conf_type)
    conf.merge_with(parse_node(conf_path))
    conf.merge_with_dotlist(argv[1:])
    if OmegaConf.is_missing(conf, 'output_dir'):
        if OmegaConf.is_missing(conf, 'exp_name'):
            conf.exp_name = conf_path.relative_to(conf.conf_root).with_suffix('')
        conf.output_dir = conf.output_root / conf.exp_name
    elif OmegaConf.is_missing(conf, 'exp_name'):
        conf.exp_name = conf.output_dir.relative_to(conf.output_root).with_suffix('')
    return conf
