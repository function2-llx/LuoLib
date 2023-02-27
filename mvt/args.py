from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np

from umei.types import tuple2_t, tuple3_t

@dataclass(kw_only=True)
class MVTArgs:
    seed: int
    model: MVTModelArgs
    datasets: MVTDatasetsArgs
    runtime: RuntimeArgs
    train: MVTTrainArgs

@dataclass(kw_only=True)
class RuntimeArgs:
    num_train_cache: int = 100
    num_val_cache: int = sys.maxsize
    num_cache_workers: int = 8
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True

@dataclass(kw_only=True)
class MVTTrainArgs:
    rotate_range: tuple3_t[Any] = (0, 0, np.pi / 2)
    rotate_prob: float = 1
    scale_range: tuple3_t[Any] = ((-0.3, 0.4), ) * 3
    scale_prob: float = 1
    gaussian_noise_prob: float = 0.1
    gaussian_smooth_prob: float = 0.1
    brightness_range: tuple2_t[float] = (0.75, 1.25)
    brightness_prob: float = 0.15
    contrast_range: tuple2_t[float] = (0.75, 1.5)
    contrast_prob: float = 0.15
    gamma_correction_range: tuple2_t[float] = (0.7, 1.5)
    gamma_correction_prob: float = 0.2
    invert_gamma_correction_prob: float = 0.5
    max_patch_size: int = 224
    max_whole_size: int = 512
    pre_pad_ratio: float = 0.4
    train_batch_size: int = 4

@dataclass(kw_only=True)
class MVTModelArgs:
    num_tokens: int
    embedding_dim: int
    project_decode: bool
    norm_pix_loss: bool
    beta: float = field(default=0.25)
    teacher_model_type: str | None
    rec_loss_type: str

    encoder: Any = field()
    quantizer: Any = field()
    decoder: Any = field()

@dataclass(kw_only=True)
class MVTDatasetsArgs:
    root: Path
    paths: list[Path]
