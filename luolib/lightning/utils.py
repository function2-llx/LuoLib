from __future__ import annotations

from dataclasses import dataclass

from luolib.optim import OptimizerCallable
from luolib.scheduler import LRSchedulerConfigWithCallable

@dataclass(kw_only=True)
class OptimizationConf:
    prefix: str | list[str] = ''
    optimizer: OptimizerCallable
    lr_scheduler: LRSchedulerConfigWithCallable
