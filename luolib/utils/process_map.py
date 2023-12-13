from itertools import starmap
from operator import length_hint

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map as tqdm_process_map

__all__ = [
    'process_map',
]

def process_map(fn, *iterables, **tqdm_kwargs):
    """When max_workers = 0, this function will call normal tqdm"""
    max_workers = tqdm_kwargs.pop('max_workers', None)
    if max_workers is not None and max_workers == 0:
        longest_iterable_len = max(map(length_hint, iterables))
        if 'total' not in tqdm_kwargs:
            tqdm_kwargs['total'] = longest_iterable_len
        return [*starmap(fn, tqdm(zip(*iterables), **tqdm_kwargs))]
    else:
        return tqdm_process_map(fn, *iterables, **tqdm_kwargs, max_workers=max_workers)
