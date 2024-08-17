from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy, StrategyRegistry

__all__ = []

StrategyRegistry.register(
    'ddp-no_broadcast_buffers',
    DDPStrategy,
    broadcast_buffers=False,
)

StrategyRegistry.register(
    'single-cuda',
    SingleDeviceStrategy,
    device=0,
)
