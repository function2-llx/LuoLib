from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(kw_only=True)
class MVTArgs:
    seed: int
    model: MVTModelArgs
    datasets: MVTDatasetsArgs
    runtime: RuntimeArgs

class RuntimeArgs:
    num_train_cache: int
    num_val_cache: int

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
