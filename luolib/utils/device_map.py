import os

import torch.cuda

__all__ = [
    'device_map',
]

num_devices = torch.cuda.device_count()

class DeviceMap:
    def __init__(self):
        from multiprocessing import Manager
        from multiprocessing.managers import SyncManager
        manager: SyncManager = Manager()
        self.pid_to_device_id = manager.dict()
        self.device_ref_count = manager.list([0 for _ in range(num_devices)])
        self.lock = manager.Lock()

    def get(self):
        pid = os.getpid()
        if (device_id := self.pid_to_device_id.get(pid)) is None:
            with self.lock:
                device_id = min(range(num_devices), key=lambda i: self.device_ref_count[i])
                self.pid_to_device_id[pid] = device_id
                self.device_ref_count[device_id] += 1
            torch.cuda.set_device(device_id)
        return torch.device(device_id)

_device_map: DeviceMap | None = None

def device_map():
    global _device_map
    if _device_map is None:
        _device_map = DeviceMap()
    return _device_map.get()
