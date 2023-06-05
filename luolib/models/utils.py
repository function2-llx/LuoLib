from pathlib import Path

from mmengine import Registry
import torch
from torch import nn

from luolib.conf import ModelConf

def load_ckpt(model: nn.Module, ckpt_path: Path | None, state_dict_key: str | None, key_prefix: str):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if state_dict_key is None and isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict_key = 'state_dict'
        elif 'model' in ckpt:
            state_dict_key = 'model'
    from timm.models.helpers import clean_state_dict
    state_dict = clean_state_dict(ckpt if state_dict_key is None else ckpt[state_dict_key])
    state_dict = {
        k[len(key_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(key_prefix)
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Loaded {} from checkpoint '{}'".format(state_dict_key, ckpt_path))
    print('missing keys:', missing_keys)
    print('unexpected keys:', unexpected_keys)

# TODO: filter kwargs
def create_model(conf: ModelConf, registry: Registry, **kwargs):
    create_fn = registry.get(conf.name)
    model = create_fn(**conf.kwargs, **kwargs)
    load_ckpt(model, conf.ckpt_path, conf.state_dict_key, conf.key_prefix)

    return model

def get_no_weight_decay_keys(module: nn.Module):
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
    for mn, m in module.named_modules():
        if hasattr(m, 'no_weight_decay'):
            no_decay |= {f'{mn}.{pn}' if mn else pn for pn in m.no_weight_decay()}

        for pn, p in m.named_parameters(prefix=mn, recurse=False):
            if not p.requires_grad:
                continue
            if pn.endswith('.bias'):
                # all biases will not be decayed
                no_decay.add(pn)
            elif pn.endswith('.weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
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
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    return no_decay
