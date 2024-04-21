from lightning.pytorch.strategies import DDPStrategy, StrategyRegistry

__all__ = []

StrategyRegistry.register(
    'ddp-no_broadcast_buffers',
    DDPStrategy,
    broadcast_buffers=False,
)
