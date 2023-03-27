from mmcv import Registry

backbone_registry = Registry('f2lib/backbone')
decoder_registry = Registry('f2lib/decoder')
loss_registry = Registry('f2lib/loss')

__all__ = [
    'backbone_registry',
    'decoder_registry',
    'loss_registry',
]
