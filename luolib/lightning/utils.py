from __future__ import annotations as _

from collections.abc import Iterable
from dataclasses import dataclass, field

import lightning
from torch import nn
from torch.optim import Optimizer

from luolib.optim import (
    HybridOptim, NamedParamGroup, OptimizerCallable, infer_weight_decay_keys,
    split_param_groups_by_weight_decay,
)
from luolib.scheduler import HybridScheduler, LRSchedulerConfig, LRSchedulerConfigWithCallable
from luolib.types import named_param_t

__all__ = [
    'OptimConf',
    'build_single_optim',
]

@dataclass(kw_only=True)
class _ParamGroup:
    # TODO: support re pattern?
    prefix: list[str]
    kwargs: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class OptimConf:
    param_groups: list[_ParamGroup]
    optimizer: OptimizerCallable
    lr_scheduler: LRSchedulerConfigWithCallable

def _match(name: str, optims: dict[str, OptimConf]) -> tuple[str, int]:
    for key, optim in optims.items():
        for i, para_group in enumerate(optim.param_groups):
            # TODO: build a AC automaton (really?!)
            if any(name.startswith(prefix) for prefix in para_group.prefix):
                return key, i
    raise ValueError(f'unable to match optimization for {name}')

def match_param_groups(
    named_parameters: Iterable[named_param_t],
    optims: dict[str, OptimConf],
) -> dict[str, list[NamedParamGroup]]:
    param_groups_by_optim = {
        name: [{'params': [], **param_group.kwargs} for param_group in optim.param_groups]
        for name, optim in optims.items()
    }
    for pn, p in named_parameters:
        if not p.requires_grad:
            # will I ever encounter the abstract case that some parameter is optimized without gradient?
            continue
        optim_key, param_group_idx = _match(pn, optims)
        param_groups_by_optim[optim_key][param_group_idx]['params'].append((pn, p))
    return param_groups_by_optim

def instantiate_optim(
    param_groups: list[NamedParamGroup],
    optim_key: str,
    optim: OptimConf,
    weight_decay_keys: set[str],
    trainer: lightning.Trainer | None = None,
) -> tuple[Optimizer, LRSchedulerConfig, list[NamedParamGroup]]:
    split_param_groups = split_param_groups_by_weight_decay(optim_key, param_groups, weight_decay_keys)
    # remove name from parameters for optimizer init
    optim_param_groups = [*map(dict, split_param_groups)]
    for param_group in optim_param_groups:
        param_group['params'] = [p for _, p in param_group.pop('params')]
    optimizer = optim.optimizer(optim_param_groups)
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
    return optimizer, lr_scheduler_config, split_param_groups

def build_single_optim(
    model: nn.Module,
    optims: dict[str, OptimConf],
    weight_decay_keys: set[str] | None = None,
    trainer: lightning.Trainer | None = None,
) -> tuple[Optimizer, LRSchedulerConfig, list[NamedParamGroup]]:
    # idea credit: https://github.com/Lightning-AI/lightning/issues/3346
    optimizers, schedulers = [], []
    param_groups_by_optim = match_param_groups(model.named_parameters(), optims)
    ref_lr_scheduler_config = None
    if weight_decay_keys is None:
        weight_decay_keys = infer_weight_decay_keys(model)
    final_param_groups = []
    for optim_key, optim in optims.items():
        param_groups = param_groups_by_optim[optim_key]
        filtered_param_groups = []
        for i, param_group in enumerate(param_groups):
            if len(param_group['params']) == 0:
                print(f'no parameter matched for ({optim_key}, {i}), pattern: {optim.param_groups[i].prefix}')
            else:
                filtered_param_groups.append(param_group)
        if len(filtered_param_groups) == 0:
            print(f'no parameter matched for optim: {optim_key}')
            continue
        optimizer, lr_scheduler_config, split_param_groups = instantiate_optim(
            filtered_param_groups,
            optim_key,
            optim,
            weight_decay_keys,
            trainer,
        )
        final_param_groups.extend(split_param_groups)
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
    assert (num_optimizers := len(optimizers)) > 0
    if num_optimizers == 1:
        return optimizers[0], ref_lr_scheduler_config, final_param_groups
    else:
        optimizer = HybridOptim(optimizers)
        ref_lr_scheduler_config.scheduler = HybridScheduler(optimizer, schedulers)
        return optimizer, ref_lr_scheduler_config, final_param_groups
