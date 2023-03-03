import builtins
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, TypeVar

import omegaconf
from omegaconf import DictConfig, OmegaConf

from monai.utils import BlendMode
from umei.types import tuple2_t, tuple3_t

# omegaconf: Unions of containers are not supported
@dataclass(kw_only=True)
class AugConf:
    dummy_dim: int | None = None
    rotate_range: list
    rotate_p: Any
    scale_range: list
    scale_p: Any
    gaussian_noise_p: float
    gaussian_noise_std: float
    gaussian_smooth_std_range: tuple[float, float]
    gaussian_smooth_isotropic_prob: float = 1
    gaussian_smooth_p: float
    scale_intensity_factor: float
    scale_intensity_p: float
    shift_intensity_offset: float
    shift_intensity_p: float
    adjust_contrast_range: tuple[float, float]
    adjust_contrast_p: float
    simulate_low_res_zoom_range: tuple[float, float]
    simulate_low_res_p: float
    gamma_range: tuple2_t[float]
    gamma_p: float
    flip_p: float = 0.5

@dataclass(kw_only=True)
class DataConf:
    spacing: tuple3_t[float]
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
    kwargs: dict

@dataclass(kw_only=True)
class FitConf(DataConf, AugConf):
    monitor: str
    monitor_mode: str
    num_train_epochs: int = 1000
    num_epoch_batches: int = 250
    train_batch_size: int
    optimizer: OptimizerConf
    scheduler: dict
    eta_min: float = 1e-6
    optimizer_set_to_none: bool = True
    precision: int = 16
    ddp_find_unused_parameters: bool = False
    num_nodes: int = 1

@dataclass(kw_only=True)
class RuntimeConf:
    train_cache_num: int = 100
    val_cache_num: int = 100
    num_cache_workers: int = 8
    dataloader_num_workers: int = 16
    dataloader_pin_memory: bool = True
    do_train: bool = False
    do_eval: bool = False
    val_empty_cuda_cache: bool = False
    eval_batch_size: int = 1
    resume_log: bool = True
    log_offline: bool = False
    num_sanity_val_steps: int = 5

@dataclass(kw_only=True)
class ModelConf:
    name: str
    ckpt_path: Path | None = None
    kwargs: dict

@dataclass(kw_only=True)
class ExpConfBase(FitConf, RuntimeConf):
    num_input_channels: int
    sample_shape: tuple3_t[int]
    backbone: ModelConf
    conf_root: Path = Path('conf-omega')
    output_root: Path = Path('output-omega')
    output_dir: Path
    exp_name: str
    log_dir: Path
    seed: int = 42
    float32_matmul_precision: str = 'high'
    ckpt_path: Path | None = None

@dataclass(kw_only=True)
class SegInferConf:
    sw_overlap: float = 0.25
    sw_batch_size: int = 16
    sw_blend_mode: BlendMode = BlendMode.GAUSSIAN
    do_tta: bool = False
    export: bool = False
    fg_oversampling_ratio: list[float] = (2, 1)  # random vs force fg

@dataclass(kw_only=True)
class SegExpConf(ExpConfBase, SegInferConf):
    monitor: str = 'val/dice/avg'
    monitor_mode: str = 'max'

    decoder: ModelConf
    num_seg_classes: int
    num_seg_heads: int = 3
    spline_seg: bool = False
    self_ensemble: bool = False
    dice_include_background: bool = True
    dice_squared: bool = False
    multi_label: bool
    dice_nr: float = 1e-5
    dice_dr: float = 1e-5

def parse_node(conf_path: Path):
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

    return OmegaConf.unsafe_merge(*base_confs, conf)

T = TypeVar('T', bound=ExpConfBase)
def parse_exp_conf(conf_type: type[T]) -> T:
    argv = sys.argv[1:]
    conf_path = Path(argv[0])
    conf: ExpConfBase | DictConfig = OmegaConf.structured(conf_type)
    conf.merge_with(parse_node(conf_path))
    conf.merge_with_dotlist(argv[1:])
    if OmegaConf.is_missing(conf, 'output_dir'):
        if OmegaConf.is_missing(conf, 'exp_name'):
            conf.exp_name = conf_path.relative_to(conf.conf_root).with_suffix('')
        conf.output_dir = conf.output_root / conf.exp_name
    return conf
