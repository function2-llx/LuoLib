import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Optional

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyExperiment

def node_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on rank 0."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs) -> Optional[Any]:
        if node_zero_only.node == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn

node_zero_only.node = getattr(node_zero_only, 'node', int(os.environ.get('NODE_RANK', 0)))

def node_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @node_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment

class MyWandbLogger(WandbLogger):
    # cannot wait: https://github.com/PyTorchLightning/pytorch-lightning/pull/12604/
    @WandbLogger.name.getter
    def name(self) -> Optional[str]:
        return self._experiment.name if self._experiment else self._name

    # FIX: just set both `RANK` instead of `NODE_RANK` (strange discrepancy between pytorch and pl)
    # # https://github.com/PyTorchLightning/pytorch-lightning/issues/11380
    # @WandbLogger.experiment.getter
    # @node_zero_experiment
    # def experiment(self):
    #     return super().experiment
