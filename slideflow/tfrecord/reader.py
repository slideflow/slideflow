"""Reader utils."""

from __future__ import absolute_import

import gzip
import io
import os
import struct
import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple, Union


from slideflow.tfrecord import iterator_utils
from slideflow.util import example_pb2, extract_feature_dict, log

# -----------------------------------------------------------------------------

def _read_data(file, length_bytes, crc_bytes, datum_bytes) -> memoryview:
    """Read the next record from the tfrecord file."""
    if file.readinto(length_bytes) != 8:
        raise RuntimeError("Failed to read the record size.")
    if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the start token.")
    length, = struct.unpack("<Q", length_bytes)
    if length > len(datum_bytes):
        try:
            _fill = int(length * 1.5)
            datum_bytes = datum_bytes.zfill(_fill)
        except OverflowError:
            raise OverflowError('Error reading tfrecords; please '
                                'try regenerating index files')
    datum_bytes_view = memoryview(datum_bytes)[:length]
    if file.readinto(datum_bytes_view) != length:
        raise RuntimeError("Failed to read the record.")
    if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the end token.")
    return datum_bytes_view

# -----------------------------------------------------------------------------


class TFRecordIterator:
    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    def __init__(
        self,
        data_path: str,
        index: Optional[np.ndarray] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        random_start: bool = False,
        datum_bytes: Optional[bytearray] = None,
    ) -> None:
        """Create an iterator over the tfrecord dataset.

        Since the tfrecords file stores each example as bytes, we can
        define an iterator over `datum_bytes_view`, which is a memoryview
        object referencing the bytes.

        Params:
        -------
        data_path: str
            TFRecord file path.

        index: optional, default=None
            np.loadtxt(index_path, dtype=np.int64)

        shard: tuple of ints, optional, default=None
            A tuple (index, count) representing worker_id and num_workers
            count. Necessary to evenly split/shard the dataset among many
            workers (i.e. >1).

        random_start: randomize starting location of reading.
            Requires an index file. Only works if shard is None.

        Yields:
        -------
        datum_bytes_view: memoryview
            Object referencing the specified `datum_bytes` contained in the
            file (for a single record).
        """

        if compression_type == "gzip":
            self.file = gzip.open(data_path, 'rb')
        elif compression_type is None:
            self.file = io.open(data_path, 'rb')  # type: ignore
        else:
            raise ValueError("compression_type should be 'gzip' or None")

        self.data_path = data_path
        self.shard = shard
        self.clip = clip
        self.random_start = random_start
        if datum_bytes is not None:
            self.datum_bytes = datum_bytes
        else:
            self.datum_bytes = bytearray(1024 * 1024)
        self.length_bytes = bytearray(8)
        self.crc_bytes = bytearray(4)
        self.index = index
        self.index_is_sequential = None
        if self.index is not None:
            # For the case that there is only a single record in the file
            if len(self.index.shape) == 1:
                self.index = np.expand_dims(self.index, axis=0)

            # Check if the index file contains sequential records
            self.index_is_nonsequential = (
                not np.all(np.cumsum(self.index[:, 1][:-1])
                           + self.index[0, 0] == self.index[:, 0][1:])
            )

            # Only keep the starting bytes for the indices
            self.index = self.index[:, 0]  # type: ignore

            # Ensure the starting bytes are in order
            self.index = np.sort(self.index)

    def _read_sequential_records(self, start_offset=None, end_offset=None):
        """Read sequential records from the given starting byte."""
        if start_offset is not None:
            self.file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(self.data_path)
        while self.file.tell() < end_offset:
            yield self._read_next_data()

    def _read_nonsequential_records(self, start_offset=None, end_offset=None):
        """Read nonsequential records from the given starting byte.

        Only read records with starting bytes reflected in the index file.
        """
        if start_offset not in self.index:
            raise ValueError("Offset not in the tfrecord index.")
        if start_offset is None:
            start_offset = self.index[0]
            index_loc = 0
        else:
            index_loc = np.argwhere(self.index == start_offset)[0][0]

        if end_offset is None:
            end_offset = os.path.getsize(self.data_path)

        while self.index[index_loc] < end_offset:
            if self.file.tell() != self.index[index_loc]:
                self.file.seek(self.index[index_loc])

            yield self._read_next_data()
            index_loc += 1

            # End the loop if we have reached the last index
            if index_loc >= len(self.index):
                break

    def _read_next_data(self) -> memoryview:
        """Read the next record from the tfrecord file."""
        data = _read_data(
            self.file,
            self.length_bytes,
            self.crc_bytes,
            self.datum_bytes
        )
        return self.process(data)

    def read_records(self, start_offset=None, end_offset=None):
        if self.index_is_nonsequential:
            yield from self._read_nonsequential_records(start_offset, end_offset)
        else:
            yield from self._read_sequential_records(start_offset, end_offset)

    def __iter__(self) -> Iterable[memoryview]:
        """Create the iterator."""

        if self.index is None:
            yield from self.read_records()
        else:
            if self.clip:
                if self.clip == len(self.index):
                    clip_offset = None
                else:
                    clip_offset = self.index[self.clip]
                self.index = self.index[:self.clip]
            else:
                clip_offset = None
            if self.shard is None and self.random_start:
                assert self.index is not None
                offset = np.random.choice(self.index)
                yield from self.read_records(offset, clip_offset)
                yield from self.read_records(0, offset)
            elif self.shard is None:
                yield from self.read_records(0, clip_offset)
            else:
                shard_idx, shard_count = self.shard
                all_shard_indices = np.array_split(self.index, shard_count)
                if shard_count >= self.index.shape[0]:  # type: ignore
                    # There are fewer records than shards, so
                    # only the first shard will read
                    if shard_idx == 0:
                        start_byte = all_shard_indices[shard_idx][0]
                        yield from self.read_records(start_byte, clip_offset)
                        return
                    else:
                        return
                elif shard_idx < (shard_count-1):
                    end_byte = all_shard_indices[shard_idx + 1][0]
                else:
                    end_byte = clip_offset
                start_byte = all_shard_indices[shard_idx][0]
                yield from self.read_records(start_byte, end_byte)

    def process(self, record):
        return record

    def close(self):
        self.file.close()


class IndexedTFRecordIterator:
    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    def __init__(
        self,
        data_path: str,
        index: np.ndarray,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        datum_bytes: Optional[bytearray] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if compression_type == "gzip":
            self.file = gzip.open(data_path, 'rb')
        elif compression_type is None:
            self.file = io.open(data_path, 'rb')  # type: ignore
        else:
            raise ValueError("compression_type should be 'gzip' or None")

        self.data_path = data_path
        self.index = index  # type: np.ndarray
        if datum_bytes is not None:
            self.datum_bytes = datum_bytes
        else:
            self.datum_bytes = bytearray(1024 * 1024)
        self.length_bytes = bytearray(8)
        self.crc_bytes = bytearray(4)

        # For the case that there is only a single record in the file.
        if len(self.index.shape) == 1:
            self.index = np.expand_dims(self.index, axis=0)

        # Only keep the starting bytes for the indices.
        self.index = self.index[:, 0]  # tpe: ignore

        # Clip.
        if clip:
            self.index = self.index[:min(clip, len(self.index))]

        # Shard.
        if shard is not None:
            shard_idx, shard_count = shard
            self.index = np.array_split(self.index, shard_count)[shard_idx]

        # Shuffle.
        if shuffle:
            self.index = np.random.RandomState(seed).permutation(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx) -> memoryview:
        return self.get_item_at_index(idx)

    def get_item_at_index(self, idx) -> memoryview:
        start_byte = self.index[idx]
        self.file.seek(start_byte)
        data = _read_data(
            self.file,
            self.length_bytes,
            self.crc_bytes,
            self.datum_bytes
        )
        return self.process(data)

    def process(self, record):
        return record

    def close(self):
        self.file.close()



class ExampleIterator(TFRecordIterator):
    def __init__(
        self,
        data_path: str,
        index: Optional[np.ndarray] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        random_start: bool = False,
        datum_bytes: Optional[bytearray] = None,
        description: Union[List[str], Dict[str, str], None] = None,
    ):
        """
        description: list or dict of str, optional, default=None
            List of keys or dict of (key, value) pairs to extract from each
            record. The keys represent the name of the features and the
            values ("byte", "float", or "int") correspond to the data type.
            If dtypes are provided, then they are verified against the
            inferred type for compatibility purposes. If None (default),
            then all features contained in the file are extracted.
        """
        super().__init__(
            data_path,
            index,
            shard,
            clip,
            compression_type,
            random_start,
            datum_bytes
        )
        self.description = description

    def process(self, record):
        example = example_pb2.Example()
        example.ParseFromString(record)
        return extract_feature_dict(
            example.features,
            self.description,
            self.typename_mapping
        )


class IndexedExampleIterator(IndexedTFRecordIterator):
    def __init__(
        self,
        data_path: str,
        index: np.ndarray,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        datum_bytes: Optional[bytearray] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        description: Union[List[str], Dict[str, str], None] = None,
    ):
        super().__init__(
            data_path,
            index=index,
            shard=shard,
            clip=clip,
            compression_type=compression_type,
            datum_bytes=datum_bytes,
            shuffle=shuffle,
            seed=seed
        )
        self.description = description

    def process(self, record):
        example = example_pb2.Example()
        example.ParseFromString(record)
        return extract_feature_dict(
            example.features,
            self.description,
            self.typename_mapping
        )


class SequenceIterator(TFRecordIterator):
    def __init__(
        self,
        data_path: str,
        index: Optional[np.ndarray] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        random_start: bool = False,
        datum_bytes: Optional[bytearray] = None,
        context_description: Union[List[str], Dict[str, str], None] = None,
        features_description: Union[List[str], Dict[str, str], None] = None,
    ):
        """
        description: list or dict of str, optional, default=None
            List of keys or dict of (key, value) pairs to extract from each
            record. The keys represent the name of the features and the
            values ("byte", "float", or "int") correspond to the data type.
            If dtypes are provided, then they are verified against the
            inferred type for compatibility purposes. If None (default),
            then all features contained in the file are extracted.
        """
        super().__init__(
            data_path,
            index,
            shard,
            clip,
            compression_type,
            random_start,
            datum_bytes
        )
        self.context_description = context_description
        self.features_description = features_description

    def process(self, record):
        example = example_pb2.SequenceExample()
        example.ParseFromString(record)
        context = extract_feature_dict(
            example.context,
            self.context_description,
            self.typename_mapping
        )
        features = extract_feature_dict(
            example.feature_lists,
            self.features_description,
            self.typename_mapping
        )
        yield context, features


def tfrecord_loader(
    data_path: str,
    index: None,
    description: Union[List[str], Dict[str, str], None] = None,
    shard: Optional[Tuple[int, int]] = None,
    clip: Optional[int] = None,
    sequence_description: Union[List[str], Dict[str, str], None] = None,
    compression_type: Optional[str] = None,
    datum_bytes: Optional[bytearray] = None,
) -> Iterable[Union[
        Dict[str, np.ndarray],
        Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
        Index file path. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        or an empty list or dictionary, then all features contained in
        the file are extracted.

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

    Yields:
    -------
    features: dict of {str, value}
        Decoded bytes of the features into its respective data type (for
        an individual record). `value` is either going to be an np.ndarray
        in the instance of an `Example` and a list of np.ndarray in the
        instance of a `SequenceExample`.
    """
    if sequence_description is not None:
        return SequenceIterator(  # type: ignore
            data_path=data_path,
            index=index,
            context_description=description,
            features_description=sequence_description,
            shard=shard,
            clip=clip,
            compression_type=compression_type
        )
    return ExampleIterator(  # type: ignore
        data_path=data_path,
        index=index,
        description=description,
        shard=shard,
        clip=clip,
        compression_type=compression_type,
        datum_bytes=datum_bytes
    )


def multi_tfrecord_loader(
    paths: List[bytes],
    indices,
    splits: Optional[Dict[str, float]],
    description: Union[List[str], Dict[str, str], None] = None,
    sequence_description: Union[List[str], Dict[str, str], None] = None,
    compression_type: Optional[str] = None,
    shard: Optional[Tuple[int, int]] = None,
    clip: List[int] = None,
    infinite: bool = True,
) -> Iterable[Union[Dict[str, np.ndarray],
                    Tuple[Dict[str, np.ndarray],
                    Dict[str, List[np.ndarray]]]]]:
    """Create an iterator by reading and merging multiple tfrecord datasets.

    Params:
    -------
    paths: list of str
        List of tfrecord paths.

    indices: dict mapping tfrecord names to index paths.
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

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
        Whether the returned iterator should be infinite or not

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """

    if indices is None and (shard is not None or clip is not None):
        log.debug("Index files not found for tfrecord; unable to perform "
                  " clipping or sharding (data will be duplicated).")

    datum_bytes = bytearray(1024 * 1024)
    loaders = [
        tfrecord_loader(
            data_path=tfr_path.decode('utf-8'),
            index=indices[i] if indices is not None else None,
            description=description,
            shard=shard,
            clip=(None if not clip else clip[i]),
            sequence_description=sequence_description,
            compression_type=compression_type,
            datum_bytes=datum_bytes)
        for i, tfr_path in enumerate(paths)
    ]
    if splits is not None:
        splits_list = splits
    else:
        splits_list = np.array(  # type: ignore
            [0.5 for t in range(len(paths))]
        )
    return iterator_utils.RandomSampler(
        loaders, splits_list, infinite=infinite, shard=None
    )