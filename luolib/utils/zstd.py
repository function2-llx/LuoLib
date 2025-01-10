from io import BytesIO
from pathlib import Path
from luolib.types import PathLike

import torch
import zstandard as zstd

__all__ = [
    'save_pt_zst',
    'load_pt_zst',
]

def save_pt_zst(x: ..., path: PathLike, atomic: bool = False):
    """Saves a PyTorch object to a Zstandard-compressed file.
    
    Args:
        x: PyTorch object to save (tensor, model, etc.)
        path: Path where the compressed file will be saved
        atomic: If True, saves to temporary file first then renames
    """
    if atomic:
        tmp_path = Path(path).with_name(f'.{path.name}')
        save_pt_zst(x, tmp_path, atomic=False)
        tmp_path.rename(path)
    else:
        with BytesIO() as buffer, open(path, 'wb') as f:
            torch.save(x, buffer)
            f.write(zstd.compress(buffer.getvalue()))

def load_pt_zst(path: PathLike, map_location: ... = 'cpu'):
    """Loads a PyTorch object from a Zstandard-compressed file.
    
    Args:
        path: Path to the compressed file
        map_location: Device mapping for loading tensors, see :func:`torch.load` for details

    Returns:
        Decompressed PyTorch object
    """
    with open(path, 'rb') as f:
        data_zst = f.read()
    data = zstd.decompress(data_zst)
    return torch.load(BytesIO(data), map_location)
