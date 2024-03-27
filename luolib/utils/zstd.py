from io import BytesIO
from pathlib import Path

import torch
import zstandard as zstd

__all__ = [
    'save_pt_zst',
    'load_pt_zst',
]

def save_pt_zst(x: ..., path: Path):
    with BytesIO() as buffer, open(path, 'wb') as f:
        torch.save(x, buffer)
        f.write(zstd.compress(buffer.getvalue()))

def load_pt_zst(path: Path):
    with open(path, 'rb') as f:
        data_zst = f.read()
    data = zstd.decompress(data_zst)
    return torch.load(BytesIO(data))
