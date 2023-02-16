from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(kw_only=True)
class MVTArgs:
    model: MVTModelArgs
    datasets: MVTDatasetsArgs

@dataclass(kw_only=True)
class MVTModelArgs:
    num_tokens: int = field()
    embedding_dim: int = field()
    project_decode: bool = field()
    norm_pix_loss: bool = field(default=True)
    beta: float = field(default=0.25)
    teacher_model_type: str | None = field()
    rec_loss_type: str = field()

    encoder: Any = field()
    quantizer: Any = field()
    decoder: Any = field()

@dataclass(kw_only=True)
class MVTDatasetsArgs:
    root: Path | None
    paths: list[Path]
