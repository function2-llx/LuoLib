from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightning
from torch import nn
from torch.optim import Optimizer

from monai.utils import ensure_tuple

from luolib.optim import (
    HybridOptim, NamedParamGroup, OptimizerCallable, infer_weight_decay_keys,
    normalize_param_groups,
)
from luolib.scheduler import HybridScheduler, LRSchedulerConfig, LRSchedulerConfigWithCallable
from luolib.types import named_params_t

__all__ = [
    'OptimizationConf',
    'build_hybrid_optimization',
]

@dataclass(kw_only=True)
class OptimizationConf:
    prefix: str | list[str] = ''
    optimizer: OptimizerCallable
    lr_scheduler: LRSchedulerConfigWithCallable

def create_param_groups(
    named_parameters: named_params_t,
    optimizations: Sequence[OptimizationConf],
) -> list[list[tuple[str, nn.Parameter]]]:
    param_groups = [[] for _ in range(len(optimizations))]
    for pn, p in named_parameters:
        if not p.requires_grad:
            # will I ever encounter the abstract case that some parameter is optimized without gradient?
            continue
        for i, optimization in enumerate(optimizations):
            if any(pn.startswith(prefix) for prefix in ensure_tuple(optimization.prefix)):
                param_groups[i].append((pn, p))
                break
        else:
            raise ValueError(f'unable to match optimization for {pn}')
    return param_groups

def instantiate_optimization(
    param_groups: list[NamedParamGroup],
    optimization: OptimizationConf,
    weight_decay_keys: set[str],
    trainer: lightning.Trainer | None = None,
) -> tuple[Optimizer, LRSchedulerConfig]:
    normalized_param_groups = normalize_param_groups(param_groups, weight_decay_keys)
    optimizer = optimization.optimizer(normalized_param_groups)
    lr_scheduler_config = LRSchedulerConfig(**vars(optimization.lr_scheduler))  # no type checks here, thanks
    if lr_scheduler_config.frequency == 0:
        assert trainer is not None
        # set default frequency according to trainer
        if lr_scheduler_config.interval == 'step':
            lr_scheduler_config.frequency = trainer.val_check_interval
        else:
            lr_scheduler_config.frequency = trainer.check_val_every_n_epoch
    scheduler = optimization.lr_scheduler.scheduler(optimizer)
    lr_scheduler_config.scheduler = scheduler
    return optimizer, lr_scheduler_config

def build_hybrid_optimization(
    model: nn.Module,
    optimizations: list[OptimizationConf],
    weight_decay_keys: set[str] | None = None,
    trainer: lightning.Trainer | None = None,
) -> tuple[HybridOptim, LRSchedulerConfig]:
    # idea credit: https://github.com/Lightning-AI/lightning/issues/3346
    optimizers, schedulers = [], []
    param_groups = create_param_groups(model.named_parameters(), optimizations)
    ref_lr_scheduler_config = None
    if weight_decay_keys is None:
        weight_decay_keys = infer_weight_decay_keys(model)
    for param_group, optimization in zip(param_groups, optimizations):
        optimizer, lr_scheduler_config = instantiate_optimization(
            [{'params': param_group}], optimization, weight_decay_keys, trainer,
        )
        optimizers.append(optimizer)
        schedulers.append(lr_scheduler_config.scheduler)
        lr_scheduler_config = lr_scheduler_config
        if ref_lr_scheduler_config is None:
            ref_lr_scheduler_config = lr_scheduler_config
        else:
            # TODO: check monitor
            for key in ['interval', 'frequency']:
                assert getattr(ref_lr_scheduler_config, key) == getattr(ref_lr_scheduler_config, key), (
                    "Hey, inconsistent scheduler config is not supported. "
                    "You don't want some abstract stuff like manual optimization, do you?"
                )

    optimizer = HybridOptim(optimizers)
    ref_lr_scheduler_config.scheduler = HybridScheduler(optimizer, schedulers)
    return optimizer, ref_lr_scheduler_config
