from __future__ import annotations

from functools import cached_property
import os
from pathlib import Path
import warnings

from lightning import Trainer as TrainerBase
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.loggers import WandbLogger

class Trainer(TrainerBase):
    def __init__(
        self,
        *,
        precision: _PRECISION_INPUT = '16-mixed',
        check_val_every_n_epoch: int | None = None,
        benchmark: bool = True,
        **kwargs,
    ):
        super().__init__(
            precision=precision,
            check_val_every_n_epoch=check_val_every_n_epoch,
            benchmark=benchmark,
            **kwargs,
        )

        from monai.config import USE_COMPILED
        if not USE_COMPILED:
            warnings.warn('MONAI is not using compiled')

    @cached_property
    def log_dir(self) -> Path:
        """
        You must call this on all processes. Failing to do so will cause your program to stall forever.
        """
        if self.is_global_zero:
            logger = self.logger
            assert isinstance(logger, WandbLogger)
            seed = os.getenv('PL_GLOBAL_SEED')
            root_dir = Path(logger.save_dir)
            log_dir = root_dir / f'seed-{seed}' / f'{Path(logger.experiment.dir).parent.name}'
        else:
            log_dir = None
        log_dir = self.strategy.broadcast(log_dir)
        return log_dir

    def play(self, *, ckpt_path: str | None, **kwargs):
        """have to define this function in Trainer or CLI can't find the `ckpt_path` parameter"""
