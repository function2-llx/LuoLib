from __future__ import annotations

from functools import cache
from typing import final

from lightning import LightningDataModule, LightningModule as LightningModuleBase
from lightning.pytorch.utilities.types import STEP_OUTPUT
from peft import PeftModel
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
import torch
from torch.optim import Optimizer

from luolib import lightning as lpl
from luolib.optim import infer_weight_decay_keys
from luolib.scheduler import HybridScheduler
from luolib.utils.grad import grad_norm

from .utils import OptimizationConf, build_hybrid_optimization

__all__ = [
    'LightningModule',
]

class LightningModule(LightningModuleBase):
    trainer: lpl.Trainer

    def __init__(self, *, log_grad_norm: bool = True, **kwargs):
        # TODO: move log_grad_norm to some callback
        super().__init__(**kwargs)
        self.log_grad_norm = log_grad_norm

    def get_decay_keys(self) -> set[str]:
        return infer_weight_decay_keys(self)

    @property
    def peft_model(self) -> PeftModel:
        return self._peft_model[0]

    @peft_model.setter
    def peft_model(self, value):
        self._peft_model = value

    def __setattr__(self, name: str, value: ...) -> None:
        if name == 'peft_model':
            # let nn.Module not register it as a submodule
            value = (value, )
        super().__setattr__(name, value)

    @cache
    @final
    def _get_decay_keys(self) -> set[str]:
        return self.get_decay_keys()

    @property
    def optimization(self):
        return self._optimization

    @optimization.setter
    def optimization(self, optimization: list[OptimizationConf]):
        self._optimization = optimization

    def configure_optimizers(self):
        optimizer, lr_scheduler_config = build_hybrid_optimization(
            self, self.optimization, self._get_decay_keys(), self.trainer,
        )
        return {
            'optimizer': optimizer,
            # call `vars` due to: https://github.com/Lightning-AI/lightning/issues/18870
            'lr_scheduler': vars(lr_scheduler_config),
        }

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.trainer.log_dir / 'model.txt').write_text(repr(self))

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: ..., batch_idx: int) -> None:
        match outputs:
            case torch.Tensor():
                loss = outputs
            case dict():
                loss = outputs['loss']
            case None:
                return
            case _:
                raise ValueError

        if loss.isfinite() or (save_dir := self.trainer.log_dir / 'bad-loss-ctx').exists():
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(save_dir / 'checkpoint.ckpt')
        torch.save(batch, save_dir / 'batch.pt')

    def lr_scheduler_step(self, scheduler: HybridScheduler, metric=None):
        for inner_scheduler in scheduler.schedulers:
            match inner_scheduler:
                case TIMMScheduler():
                    inner_scheduler.step_update(self.global_step + 1, metric)
                case _:
                    super().lr_scheduler_step(inner_scheduler, metric)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.log_grad_norm:
            self.log('grad_norm', grad_norm(self))

    @property
    def datamodule(self) -> LightningDataModule:
        return self.trainer.datamodule
