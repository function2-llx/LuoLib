from __future__ import annotations

import os
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    'get_cuda_device',
    'init_mapper',
]

class DeviceMapper:
    def __init__(self):
        # don't import these stuffs globally or multiprocessing context will be implicitly set
        from multiprocessing import Manager
        from multiprocessing.managers import SyncManager

        import torch.cuda
        self.num_devices = torch.cuda.device_count()

        manager: SyncManager = Manager()
        self.pid_to_device_id = manager.dict()
        self.device_ref_count = manager.list([0 for _ in range(self.num_devices)])
        self.lock = manager.Lock()

    @cache
    def get(self):
        import torch.cuda

        pid = os.getpid()
        with self.lock:
            if (device_id := self.pid_to_device_id.get(pid)) is None:
                device_id = min(range(self.num_devices), key=lambda i: self.device_ref_count[i])
                self.pid_to_device_id[pid] = device_id
                self.device_ref_count[device_id] += 1
            torch.cuda.set_device(device_id)
        return torch.device(device_id)

_mapper: DeviceMapper

def init_mapper():
    global _mapper
    _mapper = DeviceMapper()

def get_cuda_device() -> torch.device:
    """get the cuda device allocated for the current worker process"""
    return _mapper.get()
