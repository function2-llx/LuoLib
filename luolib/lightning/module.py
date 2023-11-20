from __future__ import annotations

from functools import cache
from typing import Generic, TypeVar, final

from lightning import LightningDataModule, LightningModule as LightningModuleBase
from lightning.pytorch.utilities.types import LRSchedulerConfigType, STEP_OUTPUT
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
import torch
from torch.optim import Optimizer

from monai.utils import ensure_tuple

from luolib.optim import (
    HybridOptim, NamedParamGroup, infer_weight_decay_keys, normalize_param_groups,
)
from .utils import OptimizationConf
from luolib.scheduler import HybridScheduler, LRSchedulerConfig
from luolib.utils.grad import grad_norm
from .trainer import Trainer

class LightningModule(LightningModuleBase):
    trainer: Trainer

    def __init__(self, *, log_grad_norm: bool = True, **kwargs):
        # TODO: move log_grad_norm to some callback
        super().__init__(**kwargs)
        self.log_grad_norm = log_grad_norm

    def grad_named_parameters(self):
        # will I ever encounter the abstract case that some parameter is optimized without gradient?
        for pn, p in self.named_parameters():
            if p.requires_grad:
                yield pn, p

    def get_decay_keys(self) -> set[str]:
        return infer_weight_decay_keys(self)

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

    def build_optimization(self, param_groups: list[NamedParamGroup], optimization: OptimizationConf) -> tuple[Optimizer, LRSchedulerConfigType]:
        normalized_param_groups = normalize_param_groups(param_groups, self._get_decay_keys())
        # optimizer_callable, lr_scheduler_config_with_callable = optimization.optimizer
        optimizer = optimization.optimizer(normalized_param_groups)
        lr_scheduler_config = LRSchedulerConfig(**vars(optimization.lr_scheduler))  # no type checks here, thanks
        if lr_scheduler_config.frequency == 0:
            # set default frequency
            if lr_scheduler_config.interval == 'step':
                lr_scheduler_config.frequency = self.trainer.val_check_interval
            else:
                lr_scheduler_config.frequency = self.trainer.check_val_every_n_epoch
        scheduler = optimization.lr_scheduler.scheduler(optimizer)
        lr_scheduler_config.scheduler = scheduler
        # vars(scheduler) due to: https://github.com/Lightning-AI/lightning/issues/18870
        return optimizer, vars(lr_scheduler_config)

    def configure_optimizers(self):
        # https://github.com/Lightning-AI/lightning/issues/3346
        param_groups = [[] for _ in range(len(self.optimization))]
        for pn, p in self.grad_named_parameters():
            for i, optimization in enumerate(self.optimization):
                if any(pn.startswith(prefix) for prefix in ensure_tuple(optimization.prefix)):
                    param_groups[i].append((pn, p))
                    break
            else:
                raise ValueError(f'unable to match optimization for {pn}')

        optimizations = [
            self.build_optimization([{'params': param_group}], optimization)
            for param_group, optimization in zip(param_groups, self.optimization)
        ]
        optimizers, lr_scheduler_configs = zip(*optimizations)
        optimizer = HybridOptim(optimizers)
        # check and combine lr scheduler configs
        ref_config = lr_scheduler_configs[0]
        schedulers = [ref_config['scheduler']]
        # TODO: check monitor
        for config in lr_scheduler_configs[1:]:
            for key in ['interval', 'frequency']:
                assert ref_config[key] == config[key], ("hey, inconsistent scheduler config is not supported. "
                                                        "you don't want some abstract stuff like manual optimization, do you?")
            schedulers.append(config['scheduler'])
        ref_config['scheduler'] = HybridScheduler(optimizer, schedulers)
        # when returning a dict, PL won't check if optimizer is an "Optimizable"
        return {
            'optimizer': optimizer,
            'lr_scheduler': ref_config,
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
