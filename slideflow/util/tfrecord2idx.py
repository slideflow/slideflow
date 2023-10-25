from __future__ import print_function

import io
import gzip
import os
import struct
import sys
import numpy as np
import slideflow as sf
from typing import Optional, Dict
from os.path import dirname, join, exists
from slideflow import errors


TYPENAME_MAPPING = {
    "byte": "bytes_list",
    "float": "float_list",
    "int": "int64_list"
}

FEATURE_DESCRIPTION = {
    'image_raw': 'byte',
    'slide': 'byte',
    'loc_x': 'int',
    'loc_y': 'int'
}

# -----------------------------------------------------------------------------

def create_index(
    tfrecord_file: str,
    index_file: Optional[str] = None
) -> str:
    """Create index from the tfrecords file.

    Stores starting location (byte) and length (in bytes) of each
    serialized record.

    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.

    index_file: str
        Path where to store the index file.
    """
    if index_file is None:
        index_file = join(dirname(tfrecord_file),
                          sf.util.path_to_name(tfrecord_file) + '.index')

    infile = open(tfrecord_file, "rb")
    start_bytes_array = []
    loc_array = []
    idx = 0
    datum_bytes = bytearray(1024 * 1024)

    while True:
        cur = infile.tell()
        byte_len = infile.read(8)
        if len(byte_len) == 0:
            break
        infile.read(4)
        proto_len = struct.unpack("q", byte_len)[0]

        if proto_len > len(datum_bytes):
            try:
                _fill = int(proto_len * 1.5)
                datum_bytes = datum_bytes.zfill(_fill)
            except OverflowError:
                raise OverflowError(
                    f'Error reading tfrecord {tfrecord_file}'
                )
        datum_bytes_view = memoryview(datum_bytes)[:proto_len]
        if infile.readinto(datum_bytes_view) != proto_len:
            raise RuntimeError(
                f"Failed to read record {idx} of file {tfrecord_file}"
            )
        infile.read(4)
        start_bytes_array += [[cur, infile.tell() - cur]]

        # Process record bytes, to read location information.
        try:
            record = process_record_from_bytes(datum_bytes_view)
        except errors.TFRecordsError:
            raise errors.TFRecordsError(
                f'Unable to detect TFRecord format: {tfrecord_file}'
            )
        if 'loc_x' in record and 'loc_y' in record:
            loc_array += [[record['loc_x'], record['loc_y']]]
        elif 'loc_x' in record:
            loc_array += [[record['loc_x']]]
        idx += 1

    infile.close()
    if loc_array:
        loc_array = np.array(loc_array)
    return save_index(np.array(start_bytes_array), index_file, locations=loc_array)


def save_index(
    index_array: np.ndarray,
    index_file: str,
    locations: Optional[np.ndarray] = None
) -> str:
    """Save an array as an index file."""
    if sf.util.zip_allowed():
        loc_kw = dict()
        if locations is not None:
            loc_kw['locations'] = locations
        np.savez(
            index_file,
            arr_0=index_array,
            **loc_kw
        )
        return index_file + '.npz'
    else:
        np.save(index_file + '.npy', index_array)
        return index_file + '.npy'


def find_index(tfrecord: str) -> Optional[str]:
    """Find the index file for a TFRecord."""
    name = sf.util.path_to_name(tfrecord)
    if exists(join(dirname(tfrecord), name+'.index')):
        return join(dirname(tfrecord), name+'.index')
    elif exists(join(dirname(tfrecord), name+'.index.npz')):
        return join(dirname(tfrecord), name+'.index.npz')
    elif exists(join(dirname(tfrecord), name+'.index.npy')):
        return join(dirname(tfrecord), name+'.index.npy')
    else:
        return None


def load_index(tfrecord: str) -> Optional[np.ndarray]:
    """Find and load the index associated with a TFRecord."""
    index_path = find_index(tfrecord)
    if index_path is None:
        raise OSError(f"Could not find index path for TFRecord {tfrecord}")
    if os.stat(index_path).st_size == 0:
        return None
    elif index_path.endswith('npz'):
        return np.load(index_path)['arr_0']
    elif index_path.endswith('npy'):
        return np.load(index_path)
    else:
        return np.loadtxt(index_path, dtype=np.int64)


def index_has_locations(index: str) -> bool:
    """Check if an index file has tile location information stored."""
    if index.endswith('npy'):
        return False
    else:
        try:
            return 'locations' in np.load(index).files
        except ValueError as e:
            raise ValueError(
                f"Failed to load TFRecord index. Try regenerating index files "
                f"with Dataset.rebuild_index(). Error received: {e}"
            )


def get_locations_from_index(index: str):
    if index.endswith('npy'):
        raise errors.TFRecordsIndexError(
            f"Index file {index} does not contain location information."
        )
    loaded = np.load(index)
    if 'locations' not in loaded:
        raise errors.TFRecordsIndexError(
            f"Index file {index} does not contain location information."
        )
    return [tuple(l) for l in loaded['locations']]


def get_tfrecord_length(tfrecord: str) -> int:
    """Return the number of records in a TFRecord file.

    Uses an index file if available, otherwise iterates through
    the file to find the total record length.

    Args:
        tfrecord (str): Path to TFRecord.

    Returns:
        int: Number of records.

    """
    index_path = find_index(tfrecord)
    if index_path is None:
        return read_tfrecord_length(tfrecord)
    if os.stat(index_path).st_size == 0:
        return 0
    else:
        index_array = load_index(tfrecord)
        if index_array is None:
            return 0
        else:
            return index_array.shape[0]


def read_tfrecord_length(tfrecord: str) -> int:
    """Returns number of records stored in the given tfrecord file."""
    infile = open(tfrecord, "rb")
    num_records = 0
    while True:
        infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            num_records += 1
        except Exception:
            sf.log.error(f"Failed to parse TFRecord at {tfrecord}")
            infile.close()
            return 0
    infile.close()
    return num_records


def get_tfrecord_by_index(
    tfrecord: str,
    index: int,
    *,
    compression_type: Optional[str] = None,
    index_array: Optional[np.ndarray] = None
) -> Dict:
    """Read a specific record in a TFRecord file.

    Args:
        tfrecord (str): TFRecord file to read.
        index (int): Index of record to read from the file.
        compression_type (str): Type of compression in the TFRecord file.
            Either 'gzip' or None. Defaults to None.

    Returns:
        A dictionary mapping record names (e.g., ``'slide'``, ``'image_raw'``,
        ``'loc_x'``, and ``'loc_y'``) to their values. ``'slide'`` will be a
        string, ``image_raw`` will be bytes, and ``'loc_x'`` and ``'loc_y'``
        will be `int`.

    Raises:
        slideflow.error.EmptyTFRecordsError: If the file is empty.

        slideflow.error.InvalidTFRecordIndex: If the given index cannot be found.
    """

    # Load the TFRecord file.
    if compression_type == "gzip":
        file = gzip.open(tfrecord, 'rb')
    elif compression_type is None:
        file = io.open(tfrecord, 'rb')  # type: ignore
    else:
        raise ValueError("compression_type should be 'gzip' or None")
    if not os.path.getsize(tfrecord):
        raise errors.EmptyTFRecordsError(f"{tfrecord} is empty.")

    # Load the TFRecord index file.
    if index:
        idx = index_array if index_array is not None else load_index(tfrecord)
        if idx is None:
            raise ValueError(f"Could not find tfrecord index for {tfrecord}")
        if index >= idx.shape[0]:
            raise errors.InvalidTFRecordIndex(
                f"Index {index} is invalid for tfrecord {tfrecord} "
                f"(size: {idx.shape[0]})"
            )
        start_offset = idx[index, 0]
        file.seek(start_offset)

    # Read the designated record.
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
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

    # Process record bytes.
    try:
        record = process_record_from_bytes(datum_bytes_view)
    except errors.TFRecordsError:
        raise errors.TFRecordsError(
            f'Unable to detect TFRecord format: {tfrecord}'
        )

    file.close()
    return record


def process_record_from_bytes(bytes_view):
    try:
        record = process_record(bytes_view)
    except KeyError:
        feature_description = {
            k: v for k, v in FEATURE_DESCRIPTION.items()
            if k in ('slide', 'image_raw')
        }
        try:
            record = process_record(bytes_view, description=feature_description)
        except KeyError:
            raise errors.TFRecordsError

    # Final parsing.
    if 'slide' in record:
        record['slide'] = bytes(record['slide']).decode('utf-8')
    if 'image_raw' in record:
        record['image_raw'] = bytes(record['image_raw'])
    if 'loc_x' in record:
        record['loc_x'] = record['loc_x'][0]
    if 'loc_y' in record:
        record['loc_y'] = record['loc_y'][0]
    return record


def process_record(record, description=None):
    if description is None:
        description = FEATURE_DESCRIPTION
    example = sf.util.example_pb2.Example()
    example.ParseFromString(record)
    return sf.util.extract_feature_dict(
        example.features,
        description,
        TYPENAME_MAPPING)


def main():
    if len(sys.argv) < 3:
        print("Usage: tfrecord2idx <tfrecord path> <index path>")
        sys.exit()

    create_index(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
