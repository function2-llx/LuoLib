from mmengine import Registry

backbone_registry = Registry('luolib/backbone')
decoder_registry = Registry('luolib/decoder')
loss_registry = Registry('luolib/loss')

__all__ = [
    'backbone_registry',
    'decoder_registry',
    'loss_registry',
]
