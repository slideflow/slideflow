"""Iterator utils."""

from __future__ import division

import typing
import warnings
import threading
import numpy as np
import random
from collections import deque

def cycle(iterator_fn: typing.Callable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator_fn():
            yield element


def sample_iterators(iterators: typing.List[typing.Iterator],
                     ratios: typing.List[int],
                     infinite: bool = True) -> typing.Iterable[typing.Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """
    if infinite:
        iterators = [cycle(iterator) for iterator in iterators]
    else:
        iterators = [iterator() for iterator in iterators]
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
    while iterators:
        choice = random.choices(range(len(ratios)), ratios, k=1)[0]
        try:
            yield next(iterators[choice])
        except StopIteration:
            if iterators:
                del iterators[choice]
                ratios = np.delete(ratios, choice)
                ratios = ratios / ratios.sum()


def sample_chunk_iterators_threaded(iterators: typing.List[typing.Iterator],
                           ratios: typing.List[int],
                           infinite: bool = True,
                           shard: typing.Optional[typing.Tuple[int, int]] = None,) -> typing.Iterable[typing.Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1) and synchronize random sampling.

    shard:

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """

    if infinite:
        iterators = {str(i):cycle(iterator) for i, iterator in enumerate(iterators)}
    else:
        iterators = {str(i):iterator() for i, iterator in enumerate(iterators)}
    ratios = {str(r):ratios[r] for r in range(len(ratios))}
    chunks = {str(idx):deque() for idx in range(len(ratios))}
    chunk_counts = {str(r):0 for r in range(len(ratios))}
    chunklock = threading.Lock()
    chunk_threads = {}
    finished_iteration = []
    finished_yield = []
    finished_count = 0
    num_total = len(ratios)

    def get_next_chunk(c):
        nonlocal chunks
        nonlocal chunk_counts
        nonlocal iterators
        nonlocal chunk_threads
        nonlocal finished_iteration
        try:
            chunk = next(iterators[c])
            with chunklock:
                chunks[c].extend(chunk)
                chunk_counts[c] += len(chunk)
        except StopIteration:
            finished_iteration += [c]
        if c in chunk_threads:
            with chunklock:
                del chunk_threads[c]

    start_threads = [threading.Thread(target=get_next_chunk, args=(str(r),)) for r in range(len(ratios))]
    for t in start_threads:
        t.start()
    for t in start_threads:
        t.join()

    global_idx = -1
    while finished_count < num_total:
        global_idx += 1
        ratio_keys = [r for r in ratios if r not in finished_yield]
        ratio_vals = [ratios[r] for r in ratio_keys]
        choice = str(random.choices(ratio_keys, ratio_vals, k=1)[0])
        if shard is not None and (global_idx % shard[1] != shard[0]):
            continue
        if choice in chunk_threads:
            chunk_threads[choice].join()
        if chunk_counts[choice]:
            yield chunks[choice].popleft()
            with chunklock:
                chunk_counts[choice] -= 1
            if not chunk_counts[choice] and choice in finished_iteration:
                finished_yield += [choice]
                finished_count += 1
            elif not chunk_counts[choice] and choice not in chunk_threads:
                assert choice in iterators
                t = threading.Thread(target=get_next_chunk, args=(choice,))
                with chunklock:
                    chunk_threads[choice] = t
                t.start()
        elif choice in finished_iteration:
            finished_yield += [choice]
            finished_count += 1
        else:
            #print('Dump:', choice, chunk_counts[choice], choice in chunk_threads, choice in finished_iteration, choice in finished_yield)
            raise IndexError(f"This shouldn't happen!: {choice}")
            #pass

def sample_chunk_iterators(iterators: typing.List[typing.Iterator],
                           ratios: typing.List[int],
                           infinite: bool = True,
                           shard: typing.Optional[typing.Tuple[int, int]] = None,) -> typing.Iterable[typing.Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1) and synchronize random sampling.

    shard:

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """

    if infinite:
        iterators = [cycle(iterator) for iterator in iterators]
    else:
        iterators = [iterator() for iterator in iterators]
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()

    global_idx = -1
    while iterators:
        global_idx += 1
        choice = random.choices(range(len(ratios)), ratios, k=1)[0]
        if shard is not None and (global_idx % shard[1] != shard[0]):
            continue
        try:
            yield next(iterators[choice])[0]
        except StopIteration:
            if iterators:
                del iterators[choice]
                ratios = np.delete(ratios, choice)
                ratios = ratios / ratios.sum()

def shuffle_iterator(iterator: typing.Iterator,
                     queue_size: int) -> typing.Iterable[typing.Any]:
    """Shuffle elements contained in an iterator.

    Params:
    -------
    iterator: iterator
        The iterator.

    queue_size: int
        Length of buffer. Determines how many records are queued to
        sample from.

    Yields:
    -------
    item: Any
        Decoded bytes of the features into its respective data type (for
        an individual record) from an iterator.
    """
    buffer = []
    try:
        for _ in range(queue_size):
            buffer.append(next(iterator))
    except StopIteration:
        warnings.warn("Number of elements in the iterator is less than the "
                      f"queue size (N={queue_size}).")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(iterator)
            yield item
        except StopIteration:
            yield buffer.pop(index)
