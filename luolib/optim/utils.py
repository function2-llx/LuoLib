from __future__ import annotations

from typing import Callable, Iterable, TypedDict

from peft.tuners.lora import LoraLayer
from torch import nn
from torch.optim import Optimizer

from luolib.models.param import NoWeightDecayParameter
from luolib.utils import partition_by_predicate

__all__ = [
    'OptimizerCallable',
    'infer_weight_decay_keys',
    'NamedParamGroup',
    'split_param_groups_by_weight_decay',
]

OptimizerCallable = Callable[[Iterable], Optimizer]

def infer_weight_decay_keys(module: nn.Module):
    """
    Force weight decay:
      - weight of linear, conv
      - including *_proj_weight in nn.MultiheadAttention
    Force no weight decay:
      - any bias
      - weight (batch/instance/group/layer) norm
      - weight of embedding
      - explicit NoWeightDecayParameter
      - defined in `no_weight_decay` method of a module
      - weight from LoRA layer
    """
    # modify from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py, `configure_optimizers`
    from torch.nn.modules.conv import _ConvNd
    whitelist_weight_modules = (
        nn.Linear,
        _ConvNd,
    )
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm
    blacklist_weight_modules = (
        nn.LayerNorm,
        _BatchNorm,
        _InstanceNorm,
        nn.GroupNorm,
        nn.Embedding,
    )
    decay = set()
    no_decay = set()
    grad_params = []
    for mn, m in module.named_modules():
        if hasattr(m, 'no_weight_decay'):
            no_decay |= {f'{mn}.{pn}' if mn else pn for pn in m.no_weight_decay()}
        if isinstance(m, LoraLayer):
            # weights from LoRA layers will not be decayed
            assert isinstance(m, nn.Module)
            no_decay |= {pn for pn, p in m.named_parameters(prefix=mn)}
        for pn, p in m.named_parameters(prefix=mn, recurse=False):
            if not p.requires_grad:
                continue
            grad_params.append(pn)
            if isinstance(p, NoWeightDecayParameter):
                no_decay.add(pn)
            elif pn.endswith('.bias'):
                # all biases will not be decayed
                no_decay.add(pn)
            elif pn.endswith('.weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                if pn not in no_decay:
                    # do not add LoRA weights again
                    decay.add(pn)
            elif isinstance(m, nn.MultiheadAttention):
                if pn.endswith('_proj_weight'):
                    # projection weights of MultiheadAttention modules will be weight decayed
                    decay.add(pn)
                elif pn.endswith('_proj_bias'):
                    no_decay.add(pn)
            elif pn not in no_decay:
                assert pn.endswith('.weight') and isinstance(m, blacklist_weight_modules)
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(pn)

    inter_params = decay & no_decay
    assert len(inter_params) == 0, f'parameters {inter_params} made it into both decay/no_decay sets!'
    diff_params = set(grad_params) - (decay | no_decay)
    assert len(diff_params) == 0, f'parameters {diff_params} were not separated into either decay/no_decay set!'
    return decay

class NamedParamGroup(TypedDict, total=False):
    # set total=False to comfort IDE
    name: str
    params: list[tuple[str, nn.Parameter]]
    lr: float
    weight_decay: float
    lr_scale: float  # inserted by timm

def split_param_groups_by_weight_decay(param_groups: list[NamedParamGroup], decay_keys: set[str]) -> list[NamedParamGroup]:
    """
    partition each param group into decay/no decay group
    """
    split_param_groups = []
    for param_group in param_groups:
        name = param_group.pop('name')
        params = param_group.pop('params')

        no_decay_params, decay_params = partition_by_predicate(lambda np: np[0] in decay_keys, params)
        if no_decay_params:
            split_param_groups.append(
                {
                    'name': f'{name} (nd)',
                    'params': no_decay_params,
                    **param_group,
                    'weight_decay': 0,
                }
            )
        if decay_params:
            split_param_groups.append(
                {
                    'name': name,
                    'params': decay_params,
                    **param_group,
                }
            )
    return split_param_groups
