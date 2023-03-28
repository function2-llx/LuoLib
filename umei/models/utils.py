from pathlib import Path

from mmcv import Registry
import torch
from torch import nn

from umei.conf import ModelConf

def load_ckpt(model: nn.Module, ckpt_path: Path | None, key_prefix: str):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict_key = ''
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict_key = 'state_dict'
        elif 'model' in ckpt:
            state_dict_key = 'model'
    from timm.models.helpers import clean_state_dict
    state_dict = clean_state_dict(ckpt[state_dict_key] if state_dict_key else ckpt)
    state_dict = {
        k[len(key_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(key_prefix)
    }
    model.load_state_dict(state_dict)
    print("Loaded {} from checkpoint '{}'".format(state_dict_key, ckpt_path))

# TODO: filter kwargs
def create_model(conf: ModelConf, registry: Registry, **kwargs):
    create_fn = registry.get(conf.name)
    model = create_fn(**conf.kwargs, **kwargs)
    load_ckpt(model, conf.ckpt_path, conf.key_prefix)

    return model
