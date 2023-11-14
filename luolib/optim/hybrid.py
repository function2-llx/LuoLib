from collections.abc import Callable, Iterable

import cytoolz
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict

__all__ = [
    'HybridOptim',
]

class HybridOptim(Optimizer):
    """
    Wrapper around multiple optimizers that should be stepped together at a single time. This is
    a hack to avoid PyTorch Lightning calling ``training_step`` once for each optimizer, which
    increases training time and is not always necessary.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687
    """
    def __init__(self, optimizers: Iterable[Optimizer]) -> None:
        # not calling super().__init__() because this one is abstract
        # super().__init__()
        self.optimizers = list(optimizers)
        self.param_groups = list(cytoolz.concat([optimizer.param_groups for optimizer in self.optimizers]))
        self.state = {}
        self.defaults = {}

    # @property
    # def state(self) -> dict[str, torch.Tensor]:
    #     """Return the combined state for each optimizer in ``self.optimizers``."""
    #     return {
    #         f'optim{i}-{key}': value
    #         for i, optimizer in enumerate(self.optimizers)
    #         for key, value in optimizer.state.items()
    #     }

    # @property
    # def param_groups(self):
    #     """Return the combined parameter groups for each optimizer in ``self.optimizers``."""
    #     return list(cytoolz.concat([optimizer.param_groups for optimizer in self.optimizers]))

    def __getstate__(self) -> list[Optimizer]:
        """Return ``self.optimizers`` for pickling purposes."""
        return self.optimizers

    def __setstate__(self, optimizers: list[Optimizer]) -> None:
        self.optimizers = optimizers

    def __repr__(self) -> str:
        """Call and concatenate ``__repr__`` for each optimizer in ``self.optimizers``."""
        repr_str = f'``{self.__class__.__name__}`` containing {len(self.optimizers)} optimizers:\n'

        for optimizer in self.optimizers:
            repr_str += '\n' + optimizer.__repr__()

        return repr_str

    def state_dict(self) -> list[StateDict]:
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict: list[StateDict]) -> None:
        for state, optimizer in zip(state_dict, self.optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] = None) -> torch.Tensor:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss
