from __future__ import annotations

from collections.abc import Iterable, Sequence
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
from luolib.types import named_param_t

__all__ = [
    'OptimConf',
    'build_hybrid_optim',
]

@dataclass(kw_only=True)
class OptimConf:
    prefix: str | list[str] = ''
    optimizer: OptimizerCallable
    lr_scheduler: LRSchedulerConfigWithCallable

def create_param_groups(
    named_parameters: Iterable[named_param_t],
    optimizations: dict[str, OptimConf],
) -> dict[str, list[named_param_t]]:
    param_groups = {name: [] for name in optimizations}
    for pn, p in named_parameters:
        if not p.requires_grad:
            # will I ever encounter the abstract case that some parameter is optimized without gradient?
            continue
        for name, optimization in optimizations.items():
            # TODO: build a AC automaton (really?!)
            if any(pn.startswith(prefix) for prefix in ensure_tuple(optimization.prefix)):
                param_groups[name].append((pn, p))
                break
        else:
            raise ValueError(f'unable to match optimization for {pn}')
    return param_groups

def instantiate_optim(
    param_groups: list[NamedParamGroup],
    optim: OptimConf,
    weight_decay_keys: set[str],
    trainer: lightning.Trainer | None = None,
) -> tuple[Optimizer, LRSchedulerConfig]:
    normalized_param_groups = normalize_param_groups(param_groups, weight_decay_keys)
    optimizer = optim.optimizer(normalized_param_groups)
    lr_scheduler_config = LRSchedulerConfig(**vars(optim.lr_scheduler))  # no type checks here, thanks
    if lr_scheduler_config.frequency == 0:
        assert trainer is not None
        # set default frequency according to trainer
        if lr_scheduler_config.interval == 'step':
            lr_scheduler_config.frequency = trainer.val_check_interval
        else:
            lr_scheduler_config.frequency = trainer.check_val_every_n_epoch
    scheduler = optim.lr_scheduler.scheduler(optimizer)
    lr_scheduler_config.scheduler = scheduler
    return optimizer, lr_scheduler_config

def build_hybrid_optim(
    model: nn.Module,
    optims: dict[str, OptimConf],
    weight_decay_keys: set[str] | None = None,
    trainer: lightning.Trainer | None = None,
) -> tuple[HybridOptim, LRSchedulerConfig]:
    # idea credit: https://github.com/Lightning-AI/lightning/issues/3346
    optimizers, schedulers = [], []
    param_groups = create_param_groups(model.named_parameters(), optims)
    ref_lr_scheduler_config = None
    if weight_decay_keys is None:
        weight_decay_keys = infer_weight_decay_keys(model)
    for name, optim in optims.items():
        param_group = param_groups[name]
        if len(param_group) == 0:
            print(f'no parameter for optimization group: {name}')
            continue
        optimizer, lr_scheduler_config = instantiate_optim(
            [{'name': name, 'params': param_group}], optim, weight_decay_keys, trainer,
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
