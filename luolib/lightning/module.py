from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Optional, Union, final

from lightning import LightningDataModule, LightningModule as LightningModuleBase
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

@dataclass
class TrainingStepContext:
    batch: ... = None

class LightningModule(LightningModuleBase):
    trainer: lpl.Trainer

    def __init__(self, *, log_grad_norm: bool = True, **kwargs):
        # TODO: should I move log_grad_norm to some callback?
        super().__init__(**kwargs)
        self.log_grad_norm = log_grad_norm
        self.training_step_context = TrainingStepContext()

    def get_decay_keys(self) -> set[str]:
        return infer_weight_decay_keys(self)

    @property
    def peft_model(self) -> PeftModel:
        return self._peft_model[0]

    # @peft_model.setter
    # def peft_model(self, value):
    #     self._peft_model = value

    def set_peft_model(self, value: PeftModel):
        self._peft_model = (value, )

    @property
    def batch(self):
        """The current batch"""
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

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
    def optim(self):
        return self._optim

    @optim.setter
    def optim(self, optim: dict[str, OptimizationConf]):
        self._optim = list(optim.values())

    def configure_optimizers(self):
        optimizer, lr_scheduler_config = build_hybrid_optimization(
            self, self.optim, self._get_decay_keys(), self.trainer,
        )
        return {
            'optimizer': optimizer,
            # call `vars` due to: https://github.com/Lightning-AI/lightning/issues/18870
            'lr_scheduler': vars(lr_scheduler_config),
        }

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.trainer.log_dir / 'model.txt').write_text(repr(self))

    def on_train_batch_start(self, batch: ..., batch_idx: int) -> int | None:
        self.training_step_context.batch = batch
        return None  # make PyCharm happy

    def lr_scheduler_step(self, scheduler: HybridScheduler, metric=None):
        for inner_scheduler in scheduler.schedulers:
            match inner_scheduler:
                case TIMMScheduler():
                    inner_scheduler.step_update(self.global_step + 1, metric)
                case _:
                    super().lr_scheduler_step(inner_scheduler, metric)

    def on_after_backward(self):
        if self.trainer.world_size > 1:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f'[rank {self.global_rank}] none grad', name)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if self.log_grad_norm:
            # log gradient before gradient clipping
            self.log('grad_norm', grad_norm(self))
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm,
        )
        if self.log_grad_norm:
            self.log('grad_norm-clipped', grad_norm(self))
        # save bad state for diagnosis
        # TODO: also check:
        #  - loss, but I can't get the step output, over-engineering, 作茧自缚了 :( PL should really officially support "training step context"
        #  - parameters after optimizer step, but this seems to require save the state before optimization
        if (save_dir := self.trainer.log_dir / 'bad-state' / f'rank-{self.global_rank}').exists():
            return
        for param in self.parameters():
            if param.grad is not None and not param.isfinite().all():
                save_dir.mkdir(parents=True)
                self.trainer.save_checkpoint(save_dir, local=True)
                torch.save(self.training_step_context.batch, save_dir / f'batch.pt')
                break

    @property
    def datamodule(self) -> LightningDataModule:
        return self.trainer.datamodule
