from timm.scheduler.scheduler import Scheduler as TIMMScheduler

from luolib.optim import HybridOptim
from .utils import LRScheduler

__all__ = [
    'HybridScheduler',
]

class HybridScheduler:
    """
    Wrapper class around ``lr_scheduler``s to return a dummy optimizer to pass PyTorch Lightning
    checks.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687
    """
    def __init__(self, optimizer: HybridOptim, schedulers: list[LRScheduler]):
        self.optimizer = optimizer
        self.schedulers = schedulers

    def state_dict(self) -> list:
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def load_state_dict(self, state_dict: list) -> None:
        for state, scheduler in zip(state_dict, self.schedulers):
            scheduler.load_state_dict(state)

    def step(self, global_step: int, metric=None):
        for scheduler in self.schedulers:
            match scheduler:
                case TIMMScheduler():
                    scheduler.step_update(global_step + 1, metric)
                case _:
                    if metric is None:
                        scheduler.step()  # type: ignore[call-arg]
                    else:
                        scheduler.step(metric)
