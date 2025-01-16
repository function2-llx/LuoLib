from pathlib import Path

import fcntl

from luolib.types import PathLike

__all__ = [
    'file_append',
    'SavedSet',
]

def file_append(filepath: Path, s: str):
    with open(filepath, 'a') as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.write(s + '\n')
        fcntl.lockf(f, fcntl.LOCK_UN)

class SavedSet:
    def __init__(self, save_path: PathLike):
        self.save_path = save_path = Path(save_path)
        if not save_path.exists():
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_path.touch()
        self.set = frozenset(save_path.read_text().splitlines())

    def __contains__(self, item):
        return item in self.set

    def save(self, item: str):
        """be careful that `self.set` is not updated accordingly"""
        if item not in self.set:
            file_append(self.save_path, item)

    def save_list(self, items: list[str]):
        self.save('\n'.join(items))
