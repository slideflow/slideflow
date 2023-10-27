"""Load tfrecord files into torch datasets."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.utils.data

from slideflow.tfrecord import iterator_utils, reader

# -----------------------------------------------------------------------------

def _get_worker_id():
    worker_info = torch.utils.data.get_worker_info()
    return (0 if not worker_info else worker_info.id)

# -----------------------------------------------------------------------------

class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str or None
        The path to the index file.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """

    def __init__(
        self,
        data_path: str,
        index_path: Union[str, None] = None,
        description: Union[List[str], Dict[str, str], None] = None,
        shuffle_queue_size: Optional[int] = None,
        transform: Callable[[dict], Any] = None,
        sequence_description: Union[List[str], Dict[str, str], None] = None,
        compression_type: Optional[str] = None,
        autoshard: bool = False,
        clip: Optional[int] = None,
    ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)
        self.compression_type = compression_type
        self.autoshard = autoshard
        self.clip = clip

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.autoshard and worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = iter(reader.tfrecord_loader(
            data_path=self.data_path,
            index=self.index_path,
            description=self.description,
            shard=shard,
            clip=self.clip,
            sequence_description=self.sequence_description,
            compression_type=self.compression_type)
        )
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it

class IndexedTFRecordDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        index: np.ndarray,
        description: Union[List[str], Dict[str, str], None] = None,
        shuffle_queue_size: Optional[int] = None,
        transform: Callable[[dict], Any] = None,
        compression_type: Optional[str] = None,
        autoshard: bool = False,
        clip: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.index = index
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)
        self.compression_type = compression_type
        self.autoshard = autoshard
        self.clip = clip
        self.seed = seed
        self.shuffle = shuffle
        self.reader = reader.IndexedExampleIterator(
            data_path=self.data_path,
            index=self.index,
            description=self.description,
            clip=self.clip,
            compression_type=self.compression_type,
            shuffle=self.shuffle,
            seed=self.seed
        )

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        return self.reader[idx]


class IndexedMultiTFRecordDataset(torch.utils.data.Dataset):

    """Indexable version of MultiTFRecordDataset.

    Note that splits (TFRecord weighting) is not supported.

    """

    def __init__(
        self,
        paths: List[str],
        indices: List[np.ndarray],
        description: Union[List[str], Dict[str, str], None] = None,
        transform: Callable[[dict], Any] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[List[int]] = None,
        compression_type: Optional[str] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,

    ) -> None:
        super().__init__()
        self.paths = paths
        self.indices = indices
        self.description = description
        self.parser = transform
        self.compression_type = compression_type
        self.shard = shard
        self.clip = clip
        self.seed = seed
        self.shuffle = shuffle
        self.readers = []
        self._init_worker = _get_worker_id()

        # Initialize readers
        self._initialize_readers()
        self.reader_lengths = [len(r) for r in self.readers]
        self.num_tiles = sum(self.reader_lengths)

        # Create an array of global indices and shuffle.
        all_idx = np.arange(self.num_tiles)
        self.interleave_index = np.zeros((self.num_tiles, 2), dtype=np.int64)
        np.random.RandomState(seed).shuffle(all_idx)

        # Compute the cumulative sum of array lengths for indexing
        cum_lengths = np.insert(np.cumsum([len(a) for a in self.readers]), 0, 0)

        # Create an array of indices for each subarray
        _reader_idx = [
            all_idx[cum_lengths[i]:cum_lengths[i+1]] for i in range(len(self.readers))
        ]

        # Compute the indices of each subarray
        for i, idx in enumerate(_reader_idx):
            self.interleave_index[idx, 0] = i
            _tfr_idx = np.arange(len(idx))
            # Shuffle the order of each individual TFRecord
            if shuffle:
                np.random.RandomState(None if seed is None else seed+i).shuffle(_tfr_idx)
            self.interleave_index[idx, 1] = _tfr_idx

    def _initialize_readers(self):
         # Prepare readers for each TFRecord
        self.datum_bytes = bytearray(1024 * 1024)
        self.readers = [
            reader.IndexedExampleIterator(
                data_path=(path if isinstance(path, str) else path.decode('utf-8')),
                index=index,
                description=self.description,
                shard=self.shard,
                clip=(None if self.clip is None else self.clip[i]),
                compression_type=self.compression_type,
                seed=(None if not self.seed else self.seed + i),
                datum_bytes = self.datum_bytes
            )
            for i, (path, index) in enumerate(zip(self.paths, self.indices))
        ]

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):
        if idx < 0 or (self.num_tiles is not None and idx >= self.num_tiles):
            raise IndexError

        if _get_worker_id() != self._init_worker:
            self.close()
            self._initialize_readers()

        reader_idx, tile_idx = self.interleave_index[idx]
        item = self.readers[reader_idx][tile_idx]

        if self.parser:
            item = self.parser(item)

        return item

    def close(self):
        for reader in self.readers:
            reader.close()

    def __del__(self):
        self.close()


class MultiTFRecordDataset(torch.utils.data.IterableDataset):
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
        Input index path pattern.

    weights: list of float
        Weights for sampling from each tfrecord. If not provided, will
        perform uniform sampling.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the Dataset should be infinite or not
    """

    def __init__(
        self,
        paths: List[str],
        indices: List[str],
        weights: Optional[Union[List[float], np.ndarray]],
        description: Union[List[str], Dict[str, str], None] = None,
        shuffle_queue_size: Optional[int] = None,
        transform: Callable[[dict], Any] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[List[int]] = None,
        sequence_description: Union[List[str], Dict[str, str], None] = None,
        compression_type: Optional[str] = None,
        infinite: bool = True
    ) -> None:
        super(MultiTFRecordDataset, self).__init__()
        self.paths = paths
        self.indices = indices
        self.weights = weights
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform
        self.compression_type = compression_type
        self.infinite = infinite
        self.shard = shard
        self.clip = clip
        self.loader = None

    def __iter__(self):
        self.loader = reader.multi_tfrecord_loader(
            paths=self.paths,
            indices=self.indices,
            weights=self.weights,
            description=self.description,
            sequence_description=self.sequence_description,
            compression_type=self.compression_type,
            shard=self.shard,
            clip=self.clip,
            infinite=self.infinite
        )
        it = iter(self.loader)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it

    def close(self):
        if self.loader is not None:
            self.loader.close()
