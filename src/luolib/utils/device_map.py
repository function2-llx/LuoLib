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
    """Manages CUDA device allocation across multiple processes.
    
    This class provides a mechanism to distribute CUDA devices among different processes
    in a multi-processing environment. It ensures balanced device allocation by tracking
    device usage and assigning the least utilized device to new processes.
    
    Attributes:
        num_devices (int): Number of available CUDA devices
        pid_to_device_id (dict[int, int]): Mapping of process IDs to assigned device IDs
        device_ref_count (list[int]): List tracking number of processes using each device
        lock (threading.Lock): Multiprocessing lock for thread-safe device allocation
    """
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
    """Initialize the global DeviceMapper instance.
    
    This function must be called before any calls to :func:`get_cuda_device`.
    It creates a new DeviceMapper instance and assigns it to the global _mapper variable.
    """
    global _mapper
    _mapper = DeviceMapper()

def get_cuda_device() -> torch.device:
    """Get a CUDA device for the current process.
    
    Returns the same device for repeated calls from the same process.
    Device allocation is balanced across all processes using the DeviceMapper.
    
    Returns:
        A :class:`torch.device` object representing the allocated CUDA device
    """
    return _mapper.get()
