from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from monai.config import PathLike

def load_ckpt(model: nn.Module, ckpt_or_path: dict | PathLike | None, state_dict_key: str | None = None, key_prefix: str = ''):
    if ckpt_or_path is None:
        return
    if isinstance(ckpt_or_path, dict):
        ckpt = ckpt_or_path
    else:
        ckpt: dict = torch.load(ckpt_or_path, map_location='cpu')
    if state_dict_key is None:
        if 'state_dict' in ckpt:
            state_dict_key = 'state_dict'
        elif 'model' in ckpt:
            state_dict_key = 'model'
    from timm.models import clean_state_dict
    state_dict = clean_state_dict(ckpt if state_dict_key is None else ckpt[state_dict_key])
    state_dict = {
        k[len(key_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(key_prefix)
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if not isinstance(ckpt_or_path, dict):
        print(f'Loaded {state_dict_key} from checkpoint {ckpt_or_path}')
    print('missing keys:', missing_keys)
    print('unexpected keys:', unexpected_keys)

def forward_gc(model: nn.Module, enable: bool, gc_func: Callable, *args, **kwargs):
    if enable:
        return gc_func(model, *args, **kwargs)
    return model(*args, **kwargs)
