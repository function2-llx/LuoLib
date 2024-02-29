from functools import cached_property
from pathlib import Path
from typing import Any, Optional
import warnings

from lightning import Trainer as TrainerBase
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.loggers import WandbLogger

from luolib import lightning as lpl

__all__ = [
    'Trainer',
    'PeftTrainer',
]

class Trainer(TrainerBase):
    def __init__(
        self,
        *,
        precision: _PRECISION_INPUT = '16-mixed',
        check_val_every_n_epoch: int | None = None,
        benchmark: bool = True,
        use_distributed_sampler: bool = False,
        **kwargs,
    ):
        super().__init__(
            precision=precision,
            check_val_every_n_epoch=check_val_every_n_epoch,
            benchmark=benchmark,
            use_distributed_sampler=use_distributed_sampler,
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
        logger = self.logger
        if logger is not None and self.is_global_zero:
            assert isinstance(logger, WandbLogger)
            root_dir = Path(logger.save_dir)
            # append something like: run-{yyyymmdd_hhmmss}-{run_id}
            log_dir = root_dir / f'{Path(logger.experiment.dir).parent.name}'
        else:
            log_dir = None
        log_dir = self.strategy.broadcast(log_dir)
        return log_dir

    def play(self, *, ckpt_path: str | None, **kwargs):
        """have to define this function in Trainer or CLI can't find the `ckpt_path` parameter"""


class PeftTrainer(Trainer):
    lightning_module: lpl.LightningModule

    @property
    def peft_model(self):
        return self.lightning_module.peft_model

    def save_checkpoint(
        self, filepath, weights_only: bool = False, storage_options: Optional[Any] = None,
    ) -> None:
        if self.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        checkpoint.pop('state_dict')
        save_dir = Path(filepath)
        if self.is_global_zero:
            self.peft_model.save_pretrained(str(save_dir / 'adapter'))
        self.strategy.save_checkpoint(checkpoint, save_dir / 'state.ckpt', storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")
