from collections.abc import Callable, Iterable, Sequence
import inspect

import cytoolz
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict

__all__ = [
    'HybridOptim',
]

class _HybridSequence:
    def __init__(self, seqs: Sequence[Sequence]):
        self._seqs = seqs
        self._build_map()

    def __len__(self):
        return sum(map(len, self._seqs))

    def __iter__(self):
        yield from cytoolz.concat(self._seqs)

    @property
    def _num_seqs(self):
        return len(self._seqs)

    def __getitem__(self, i: ...):
        # simple sequential search, should not be a problem for most cases
        if not isinstance(i, int):
            raise NotImplementedError
        if i < 0:
            raise IndexError
        for seq in self._seqs:
            if i < len(seq):
                return seq[i]
            i -= len(seq)
        raise IndexError

    def _build_map(self):
        self._map = {}
        for seq_id, seq in enumerate(self._seqs):
            for x in seq:
                self._map[id(x)] = seq_id

    def __setitem__(self, key: ..., value: Sequence):
        if isinstance(key, slice) and key.start is None and key.stop is None:
            # DeepSpeed filter empty parameter groups
            seqs = [[] for _ in range(self._num_seqs)]
            for x in value:
                if (seq_id := self._map.get(id(x))) is None:
                    raise NotImplementedError
                seqs[seq_id].append(x)
            for i in range(self._num_seqs):
                self._seqs[i][:] = seqs[i]
        else:
            raise NotImplementedError
        self._build_map()

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
        assert not any(isinstance(optimizer, HybridOptim) for optimizer in optimizers)
        self._optimizers = list(optimizers)
        # self.defaults = {}

    @property
    def param_groups(self):
        """Return the combined parameter groups for each optimizer in ``self.optimizers``."""
        return _HybridSequence([optimizer.param_groups for optimizer in self._optimizers])

    @property
    def state(self) -> dict[str, torch.Tensor]:
        """Return the combined state for each optimizer in ``self.optimizers``."""
        return {
            f'optim{i}-{key}': value
            for i, optimizer in enumerate(self._optimizers)
            for key, value in optimizer.state.items()
        }

    def __getstate__(self) -> list[Optimizer]:
        """Return ``self.optimizers`` for pickling purposes."""
        return self._optimizers

    def __setstate__(self, optimizers: list[Optimizer]) -> None:
        self._optimizers = optimizers

    def __repr__(self) -> str:
        """Call and concatenate ``__repr__`` for each optimizer in ``self.optimizers``."""
        repr_str = f'``{self.__class__.__name__}`` containing {len(self._optimizers)} optimizers:\n'

        for optimizer in self._optimizers:
            repr_str += '\n' + optimizer.__repr__()

        return repr_str

    def state_dict(self) -> list[StateDict]:
        return [optimizer.state_dict() for optimizer in self._optimizers]

    def load_state_dict(self, state_dict: list[StateDict]) -> None:
        for state, optimizer in zip(state_dict, self._optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self._optimizers:
            if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
                optimizer.zero_grad(set_to_none=set_to_none)
            else:
                optimizer.zero_grad()

    def step(self, closure: Callable[[], torch.Tensor] = None) -> torch.Tensor:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self._optimizers:
            optimizer.step()

        return loss
