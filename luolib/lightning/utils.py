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
    """Single parameter group specification for an optimizer.

    This class specifies a parameter group for an optimizer by matching parameter names with given prefixes.
    Parameters whose names start with any of the specified prefixes will be grouped together
    and optimized with the same optimizer settings.

    Attributes:
        prefix: list of strings used to match parameter names, where a parameter name matches 
            if it starts with any of these prefixes
        kwargs: dict containing optimizer-specific parameters for this group (e.g. lr, 
            momentum, weight_decay)

    Example:
        >>> param_group = _ParamGroup(
        ...     prefix=['encoder.', 'decoder.'],
        ...     kwargs={'lr': 0.001, 'weight_decay': 0.01}
        ... )
    """
    prefix: list[str]
    kwargs: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class OptimConf:
    """Configuration for an optimization group.

    Attributes:
        param_groups: List of parameter group specifications. Each group defines which parameters
            should be optimized together based on name prefixes and their optimizer-specific parameters.
        optimizer: A callable that creates an optimizer instance when given parameter groups. 
            Designed to work with jsonargparse.
        lr_scheduler: Configuration for the learning rate scheduler, including both the scheduler
            factory callable and its settings.

    Example:
        >>> from functools import partial
        >>> from torch.optim import Adam
        >>> from torch.optim.lr_scheduler import StepLR
        >>> optim_conf = OptimConf(
        ...     param_groups=[
        ...         _ParamGroup(prefix=['encoder.'], kwargs={'lr': 0.001}),
        ...         _ParamGroup(prefix=['decoder.'], kwargs={'lr': 0.002})
        ...     ],
        ...     optimizer=partial(Adam, lr=0.001, weight_decay=0.01),
        ...     lr_scheduler=LRSchedulerConfigWithCallable(
        ...         scheduler=partial(StepLR, step_size=10, gamma=0.1),
        ...         interval='epoch',
        ...         frequency=10,
        ...     )
        ... )
    """
    param_groups: list[_ParamGroup]
    optimizer: OptimizerCallable
    lr_scheduler: LRSchedulerConfigWithCallable

def _match(name: str, optims: dict[str, OptimConf]) -> tuple[str, int]:
    """Matches a parameter name to a parameter group specified in some optimization group by prefix matching.

    Iterates through optimization groups and their parameter groups in order, returning the first
    match where the parameter name starts with any of prefixes specified in parameter group.

    Args:
        name: Parameter name to match
        optims: Dictionary mapping optimization group names to their configurations

    Returns:
        A tuple containing:
            - Optimization group name that contains the matching parameter group
            - Index of the matching parameter group within that optimization group

    Raises:
        ValueError: If the parameter name doesn't match any prefix in any parameter group
    """
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
    """Groups model parameters into optimizer parameter groups based on name matching.

    For each parameter, finds the first matching parameter group using :func:`_match`, then
    assigns the parameter to that group along with the group's optimizer-specific settings.

    Args:
        named_parameters: Iterator of (name, parameter) tuples from model.named_parameters()
        optims: Dictionary mapping optimization group names to their configurations

    Returns:
        Dictionary mapping optimization group names to corresponding lists of parameter groups

    Note:
        Parameters with requires_grad=False are skipped and not included in any group.
    """
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
    """Creates optimizer and lr scheduler for an optimization group.

    Args:
        param_groups: List of parameter groups, each containing named parameters and their
            optimization settings
        optim_key: Identifier for this optimization group
        optim: Configuration for this optimization group
        weight_decay_keys: Set of parameter names that should have weight decay applied
        trainer: Optional Lightning Trainer instance used to determine scheduler frequency,
            required if any scheduler has frequency=0 (unspecified)

    Returns:
        A tuple containing:
            - The instantiated optimizer
            - The lr scheduler configuration
            - List of parameter groups split by weight decay settings

    Note:
        If scheduler frequency is not specified (0), it will be automatically determined
        based on trainer settings:
        - For 'step' interval: uses :attr:`trainer.val_check_interval`
        - For epoch interval: uses :attr:`trainer.check_val_every_n_epoch`
    """
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
    """Builds a single optimizer and scheduler setup from multiple optimization configurations.

    This function creates optimizers and schedulers for different optimization groups in a model,
    then combines them into a single optimizer-scheduler pair even if multiple configurations are provided.
    This is inspired by https://github.com/Lightning-AI/lightning/issues/3346, aiming to work with PyTorch Lightning's
    automatic optimization with multiple optimizers.

    Args:
        model: The PyTorch model to optimize
        optims: Dictionary mapping optimization group names to their configurations
        weight_decay_keys: Set of parameter names that should have weight decay applied. If None,
            will be automatically inferred from the model structure with :func:`infer_weight_decay_keys`.
        trainer: Lightning Trainer instance used to determine scheduler frequency settings.
            Required if any scheduler has frequency=0 (unspecified)

    Returns:
        A tuple containing:
            - Combined optimizer (either single optimizer or :class:`HybridOptim`)
            - :class:`LRSchedulerConfig` with the combined scheduler
            - List of all parameter groups, split by weight decay settings

    Raises:
        AssertionError: 
            - If no parameters are matched to any optimization configuration
            - If schedulers have inconsistent interval or frequency settings
        ValueError: If any parameter name doesn't match any optimization configuration

    Note:
        The function combines multiple optimizers/schedulers into :class:`HybridOptim` and
        :class:`HybridScheduler` respectively when multiple optimization configurations are provided.
    """
    # Lists to collect optimizers and schedulers that will be combined later
    optimizers, schedulers = [], []
    
    # Match model parameters to their corresponding parameter groups
    param_groups_by_optim = match_param_groups(model.named_parameters(), optims)
    ref_lr_scheduler_config = None
    
    # Auto-infer weight decay keys if not provided
    if weight_decay_keys is None:
        weight_decay_keys = infer_weight_decay_keys(model)
    
    # Collect all parameter groups after weight decay splitting
    final_param_groups = []
    
    # Process each optimization group
    for optim_key, optim in optims.items():
        param_groups = param_groups_by_optim[optim_key]
        
        # Filter out empty parameter groups (no parameters matched the prefix)
        filtered_param_groups = []
        for i, param_group in enumerate(param_groups):
            if len(param_group['params']) == 0:
                print(f'no parameter matched for ({optim_key}, {i}), pattern: {optim.param_groups[i].prefix}')
            else:
                filtered_param_groups.append(param_group)
                
        # Skip optimization groups with no matched parameters
        if len(filtered_param_groups) == 0:
            print(f'no parameter matched for optimization group: {optim_key}')
            continue
            
        # Create optimizer and scheduler for this group
        optimizer, lr_scheduler_config, split_param_groups = instantiate_optim(
            filtered_param_groups,
            optim_key,
            optim,
            weight_decay_keys,
            trainer,
        )
        
        # Collect results for later combination
        final_param_groups.extend(split_param_groups)
        optimizers.append(optimizer)
        schedulers.append(lr_scheduler_config.scheduler)
        
        # Ensure all schedulers have consistent configurations
        if ref_lr_scheduler_config is None:
            ref_lr_scheduler_config = lr_scheduler_config
        else:
            # TODO: check monitor
            for key in ['interval', 'frequency']:
                assert getattr(ref_lr_scheduler_config, key) == getattr(ref_lr_scheduler_config, key), (
                    "Hey, inconsistent scheduler config is not supported. "
                    "You don't want some abstract stuff like manual optimization, do you?"
                )
    
    # Must have at least one optimizer
    assert (num_optimizers := len(optimizers)) > 0
    
    # Return single optimizer if only one exists, otherwise combine them
    if num_optimizers == 1:
        return optimizers[0], ref_lr_scheduler_config, final_param_groups
    else:
        optimizer = HybridOptim(optimizers)
        ref_lr_scheduler_config.scheduler = HybridScheduler(optimizer, schedulers)
        return optimizer, ref_lr_scheduler_config, final_param_groups
