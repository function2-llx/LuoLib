from multiprocessing import Manager
from multiprocessing.managers import SyncManager
import os

import torch.cuda

__all__ = [
    'device_map',
]

num_devices = torch.cuda.device_count()
manager: SyncManager = Manager()
pid_to_device_id = manager.dict()
device_ref_count = manager.list([0 for _ in range(num_devices)])
lock = manager.Lock()

def device_map():
    pid = os.getpid()
    if (device_id := pid_to_device_id.get(pid)) is None:
        with lock:
            device_id = min(range(num_devices), key=lambda i: device_ref_count[i])
            pid_to_device_id[pid] = device_id
            device_ref_count[device_id] += 1
        torch.cuda.set_device(device_id)
    return torch.device(device_id)
