from mmengine import Registry

backbone_registry = Registry('luolib/backbone')
decoder_registry = Registry('luolib/decoder')
transformer_decoder_registry = Registry('luolib/transformer_decoder')
loss_registry = Registry('luolib/loss')

__all__ = [
    'backbone_registry',
    'decoder_registry',
    'loss_registry',
    'transformer_decoder_registry',
]
