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

    def _check_save_checkpoint(self):
        if self.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )

    def dump_checkpoint(self, weights_only: bool) -> dict:
        return self._checkpoint_connector.dump_checkpoint(weights_only)

    def _save_checkpoint_with_strategy(self, checkpoint: dict, filepath, storage_options: ..., local: bool):
        if local:
            # NOTE: this is copied from the `Strategy` class, other implementations are not supported
            self.strategy.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        else:
            self.strategy.save_checkpoint(checkpoint, filepath, storage_options)

    def save_checkpoint(
        self,
        filepath,
        weight_only: bool = False,
        storage_options: Optional[Any] = None,
        local: bool = False,
    ) -> None:
        if local:
            self._check_save_checkpoint()
            checkpoint = self.dump_checkpoint(weight_only)
            self._save_checkpoint_with_strategy(
                checkpoint, filepath, storage_options, False,
            )
        else:
            super().save_checkpoint(filepath, weight_only, storage_options)

class PeftTrainer(Trainer):
    lightning_module: lpl.LightningModule

    def __init__(self, *, save_embedding_layers: bool | str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.save_embedding_layers = save_embedding_layers

    @property
    def peft_model(self):
        return self.lightning_module.peft_model

    def dump_checkpoint(self, weights_only: bool):
        checkpoint = super().dump_checkpoint(weights_only)
        checkpoint.pop('state_dict')
        return checkpoint

    def save_checkpoint(
        self,
        save_dir,
        weights_only: bool = False,
        storage_options: Optional[Any] = None,
        local: bool = False,
    ) -> None:
        self._check_save_checkpoint()
        save_dir = Path(save_dir)
        assert self.save_embedding_layers is not None
        if local or self.is_global_zero:
            self.peft_model.save_pretrained(
                # NOTE: if using save_embedding_layers='auto', it may access the HF hub every time, and your program will
                # crush with no mercy when the Internet becomes unavailable during training due to uncaught exception
                # see: https://github.com/huggingface/peft/blob/v0.8.2/src/peft/utils/save_and_load.py#L146
                str(save_dir / 'adapter'), save_embedding_layers=self.save_embedding_layers,
            )
        checkpoint = self.dump_checkpoint(weights_only)
        self._save_checkpoint_with_strategy(
            checkpoint, save_dir / 'state.ckpt', storage_options, local,
        )
        if not local:
            self.strategy.barrier("Trainer.save_checkpoint")
