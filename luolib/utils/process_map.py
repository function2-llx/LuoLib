from collections.abc import Callable
from itertools import starmap
from operator import length_hint

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map as tqdm_process_map

from .device_map import init_mapper

__all__ = [
    'process_map',
]

def process_map(fn: Callable, *iterables, new_mapper: bool = True, **tqdm_kwargs):
    """A wrapper for :func:`tqdm.contrib.concurrent.process_map` with additional functionality.

    This function provides parallel processing capabilities with progress bar visualization. When 
    `max_workers` is set to 0, it falls back to sequential processing using normal tqdm.

    Args:
        fn: Function to apply to the elements of the iterables
        *iterables: One or more iterables whose elements will be passed to fn
        new_mapper: Whether to initialize a new device mapper for CUDA device management
        **tqdm_kwargs: Additional keyword arguments passed to tqdm, including:
            - max_workers: Number of worker processes. If 0, uses sequential processing
            - chunksize: Size of chunks sent to worker processes
            - total: Total number of iterations (auto-calculated if not provided)
            - Other tqdm parameters like desc, leave, etc.

    Returns:
        List containing the results of applying fn to the elements of the iterables
    """
    max_workers = tqdm_kwargs.pop('max_workers', None)
    if new_mapper:
        init_mapper()
    if max_workers is not None and max_workers == 0:
        longest_iterable_len = max(map(length_hint, iterables))
        if 'total' not in tqdm_kwargs:
            tqdm_kwargs['total'] = longest_iterable_len
        tqdm_kwargs.pop('chunksize', None)
        return [*starmap(fn, tqdm(zip(*iterables), **tqdm_kwargs))]
    else:
        return tqdm_process_map(fn, *iterables, **tqdm_kwargs, max_workers=max_workers)
