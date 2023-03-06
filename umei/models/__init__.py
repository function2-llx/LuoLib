import torch

from umei.omega import ModelConf
from .layers import LayerNormNd

# TODO: refactor with some registry
def create_model(conf: ModelConf, *args, **kwargs):
    match conf.name:
        case 'swin':
            from umei.models.backbones.swin import SwinBackbone as create_fn
        case 'conv':
            from umei.models.decoders.plain_conv_unet import PlainConvUNetDecoder as create_fn
        case 'unet':
            from umei.models.backbones.unet import UNetBackbone as create_fn
        case _:
            raise ValueError(conf.name)

    model = create_fn(*args, **conf.kwargs, **kwargs)
    if conf.ckpt_path is not None:
        ckpt = torch.load(conf.ckpt_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict_key = 'state_dict'
            elif 'model' in ckpt:
                state_dict_key = 'model'
        from timm.models.helpers import clean_state_dict
        state_dict = clean_state_dict(ckpt[state_dict_key] if state_dict_key else ckpt)
        state_dict = {
            k[len(conf.key_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(f'{conf.key_prefix}')
        }
        model.load_state_dict(state_dict)
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, conf.ckpt_path))

    return model
