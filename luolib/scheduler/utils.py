from dataclasses import dataclass
from typing import Callable, Literal

from lightning.pytorch.cli import ReduceLROnPlateau
from lightning.pytorch.utilities.types import LRSchedulerConfig as LRSchedulerConfigBase
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler

__all__ = [
    'LRScheduler',
    'LRSchedulerCallable',
    'LRSchedulerConfig',
    'LRSchedulerConfigWithCallable',
]

LRScheduler = TorchLRScheduler | ReduceLROnPlateau | TIMMScheduler
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

@dataclass
class LRSchedulerConfig(LRSchedulerConfigBase):
    scheduler: LRScheduler  # this can be set to `HybridScheduler`, which is not a subclass
    frequency: int = 0  # 0: set to some default value in `build_optimization`
    interval: Literal['step', 'epoch'] = 'step'

@dataclass
class LRSchedulerConfigWithCallable(LRSchedulerConfig):
    scheduler: LRSchedulerCallable
