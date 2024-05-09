from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from copy import copy
import inspect

import cytoolz
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict

__all__ = [
    'HybridOptim',
]

class _HybridList:
    def __init__(self, lists: Sequence[Sequence]):
        self._lists = list(lists)
        self._build_map()

    def _build_map(self):
        self._map = {}
        for list_id, _list in enumerate(self._lists):
            for x in _list:
                self._map[id(x)] = list_id

    def _detach(self):
        # stop holding reference of param groups
        self._lists = list(map(copy, self._lists))

    def __len__(self):
        return sum(map(len, self._lists))

    def __iter__(self):
        yield from cytoolz.concat(self._lists)

    @property
    def _num_seqs(self):
        return len(self._lists)

    def __getitem__(self, i: ...):
        # simple sequential search, should not be a problem for most cases
        if not isinstance(i, int):
            raise NotImplementedError
        if i < 0:
            raise IndexError
        for _list in self._lists:
            if i < len(_list):
                return _list[i]
            i -= len(_list)
        raise IndexError

    def __setitem__(self, key: ..., value: Sequence):
        if isinstance(key, slice) and key.start is None and key.stop is None:
            # DeepSpeed may set parameter groups
            seqs = [[] for _ in range(self._num_seqs)]
            for x in value:
                if (seq_id := self._map.get(id(x))) is None:
                    # it is assumed that parameter groups to be set always present in the beginning
                    raise NotImplementedError
                seqs[seq_id].append(x)
            for i in range(self._num_seqs):
                # modify the original parameter groups in the optimizer inplace
                self._lists[i][:] = seqs[i]
        else:
            raise NotImplementedError

class _HybridDict:
    """it is assumed that no duplicated keys present in different maps"""
    def __init__(self, dicts: Sequence[Mapping]):
        self._dicts = list(dicts)
        self._map = {}
        for dict_id, _dict in enumerate(self._dicts):
            for k in _dict:
                self._map[k] = dict_id

    def __iter__(self):
        yield from cytoolz.concat(self._dicts)

    def items(self):
        yield from cytoolz.concat(map(Mapping.items, self._dicts))

    @property
    def _num_maps(self):
        return len(self._dicts)

    def _get_dict_id(self, key: Hashable):
        for i, _dict in enumerate(self._dicts):
            if key in _dict:
                return i
        raise KeyError

    def __getitem__(self, key: Hashable):
        dict_id = self._get_dict_id(key)
        return self._dicts[dict_id][key]

    def __setitem__(self, key: Hashable, value: ...):
        dict_id = self._get_dict_id(key)
        self._dicts[dict_id] = value

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
        self._state = _HybridDict([optimizer.state for optimizer in self._optimizers])
        self._param_groups = _HybridList([optimizer.param_groups for optimizer in self._optimizers])
        # NOTE: `LearningRateMonitor` check betas from the `defaults` attribute,
        #  therefore something must be set for this attribute or there will be an `AttributeError`
        self.defaults = {}

    @property
    def param_groups(self):
        return self._param_groups

    @param_groups.setter
    def param_groups(self, param_groups: Sequence[Mapping]):
        # NOTE: some abstract library will call param_groups's setter while holding the reference and make our trick fail,
        #  (yes I'm talking about you DxxpSpxxd ZxRO 1/2), therefore we need to detach the reference in the setter
        old = self._param_groups
        self._param_groups = copy(old)
        old._detach()
        self._param_groups[:] = param_groups

    @property
    def state(self):
        """Return the combined state for each optimizer in ``self.optimizers``."""
        return self._state

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

    def state_dict(self) -> StateDict:
        return {
            f'optim-{i}': optimizer.state_dict()
            for i, optimizer in enumerate(self._optimizers)
        }

    def load_state_dict(self, state_dict: StateDict):
        for i, optimizer in enumerate(self._optimizers):
            optimizer.load_state_dict(state_dict[f'optim-{i}'])

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
