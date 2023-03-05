from .layers import LayerNormNd

from umei.omega import ModelConf

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

    if conf.ckpt_path is not None:
        raise NotImplementedError

    return create_fn(*args, **conf.kwargs, **kwargs)
