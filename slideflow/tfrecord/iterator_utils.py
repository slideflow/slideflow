"""Iterator utils."""

from __future__ import division

import typing
import warnings

import numpy as np


class EmptyIterator(Exception):
    pass


def cycle(iterator):
    """Create a repeating iterator from an iterator."""

    while True:
        has_element = False
        for element in iterator:
            if not has_element:
                has_element = True
            yield element
        if not has_element:  # Handles empty TFRecords
            raise EmptyIterator


class RandomSampler:
    def __init__(self, loaders, ratios, infinite=True, shard=None):

        self.ratios = ratios
        self.loaders = loaders
        self.infinite = infinite
        self.shard = shard

    def __iter__(self):
        if self.infinite:
            iterators = [cycle(loader) for loader in self.loaders]
        else:
            iterators = [iter(loader) for loader in self.loaders]
        self.ratios = np.array(self.ratios)
        self.ratios = self.ratios / self.ratios.sum()
        ratio_indices = np.array(range(len(self.ratios)))
        global_idx = -1
        while iterators:
            global_idx += 1
            choice = np.random.choice(
                ratio_indices[:self.ratios.shape[0]],
                p=self.ratios
            )
            if (self.shard is not None
               and (global_idx % self.shard[1] != self.shard[0])):
                continue
            try:
                yield next(iterators[choice])
            except (StopIteration, EmptyIterator):
                if iterators:
                    del iterators[choice]
                    del self.loaders[choice]
                    self.ratios = np.delete(self.ratios, choice)
                    self.ratios = self.ratios / self.ratios.sum()

    def close(self):
        for loader in self.loaders:
            loader.close()


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
