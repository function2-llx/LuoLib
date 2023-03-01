import builtins
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, TypeVar

import omegaconf
from omegaconf import DictConfig, OmegaConf

from umei.types import tuple2_t, tuple3_t

# omegaconf: Unions of containers are not supported
@dataclass(kw_only=True)
class AugConf:
    rotate_range: list
    rotate_p: Any
    scale_range: list
    scale_p: Any
    gaussian_noise_p: float
    gaussian_noise_std: float
    scale_intensity_factor: float
    scale_intensity_p: float
    shift_intensity_offset: float
    shift_intensity_p: float
    gamma_range: tuple2_t[float]
    gamma_p: float

@dataclass(kw_only=True)
class FitConf:
    train_batch_size: int
    sample_shape: tuple3_t[int]
    spacing: tuple3_t[float]
    seed: int = 42

@dataclass(kw_only=True)
class ConfNode:
    _bases: list = field(default_factory=list)

@dataclass(kw_only=True)
class ModelConf(ConfNode):
    name: str
    ckpt_path: Path | None = None
    kwargs: dict

@dataclass(kw_only=True)
class RuntimeConf:
    train_cache_num: int
    val_cache_num: int
    num_cache_workers: int
    dataloader_num_workers: int
    dataloader_pin_memory: bool = True
    do_train: bool
    do_eval: bool


@dataclass(kw_only=True)
class ExpConfBase(ConfNode):
    conf_root: Path = Path('conf')
    output_root: Path = Path('output')
    output_dir: Path

@dataclass
class USegConf(ExpConfBase, AugConf, FitConf, RuntimeConf):
    backbone: ModelConf
    decoder: ModelConf

def parse_node(conf_path: Path):
    conf_dir = conf_path.parent
    conf = OmegaConf.create(vars(ConfNode()))
    conf.merge_with(OmegaConf.load(conf_path))
    base_conf = OmegaConf.create()
    for base in conf._bases:
        match type(base):
            case builtins.str:
                base = Path(base)
                if not base.is_absolute():
                    base = conf_dir / base
                base_conf.merge_with(parse_node(base))
            case omegaconf.DictConfig:
                for k, v in base.items():
                    assert isinstance(v, str)
                    base_conf.merge_with(OmegaConf.create({
                        k: parse_node(conf_dir / v)
                    }))
            case _:
                raise ValueError

    conf.merge_with(base_conf)
    return conf

T = TypeVar('T', bound=ExpConfBase)
def parse_exp_conf(conf_type: type[T]) -> T | DictConfig:
    argv = sys.argv[1:]
    conf_path = Path(argv[0])
    conf: ExpConfBase | DictConfig = OmegaConf.structured(conf_type)
    conf.merge_with(parse_node(conf_path))
    conf.merge_with_dotlist(argv[1:])
    if OmegaConf.is_missing(conf, 'output_dir'):
        suffix = conf_path.relative_to(conf.conf_root).with_suffix('')
        conf.output_dir = conf.output_root / suffix
    return conf
