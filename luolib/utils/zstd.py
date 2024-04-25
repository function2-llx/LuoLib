from io import BytesIO
from pathlib import Path

import torch
import zstandard as zstd

__all__ = [
    'save_pt_zst',
    'load_pt_zst',
]

def save_pt_zst(x: ..., path: Path, atomic: bool = False):
    if atomic:
        tmp_path = path.with_name(f'.{path.name}')
        save_pt_zst(x, tmp_path, atomic=False)
        tmp_path.rename(path)
    else:
        with BytesIO() as buffer, open(path, 'wb') as f:
            torch.save(x, buffer)
            f.write(zstd.compress(buffer.getvalue()))

def load_pt_zst(path: Path):
    with open(path, 'rb') as f:
        data_zst = f.read()
    data = zstd.decompress(data_zst)
    return torch.load(BytesIO(data))
