import os
import io
import struct
import imghdr
from slideflow.util import log, example_pb2, extract_feature_dict
from slideflow import errors


def detect_tfrecord_format(tfr):
    '''Detects tfrecord format.

    Args:
        tfr (str): Path to tfrecord.

    Returns:
        str: Image file type (png/jpeg)

        dict: Feature description dictionary (including or excluding
        location data as supported)
    '''
    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }
    feature_description = {
        'image_raw': 'byte',
        'slide': 'byte',
        'loc_x': 'int',
        'loc_y': 'int'
    }

    def process(record, description):
        example = example_pb2.Example()
        example.ParseFromString(record)
        return extract_feature_dict(example.features, description, typename_mapping)

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
    file = io.open(tfr, 'rb')
    if not os.path.getsize(tfr):
        log.debug(f"Unable to detect format for {tfr}; file empty.")
        return None, None
    file.tell()
    if file.readinto(length_bytes) != 8:
        raise RuntimeError("Failed to read the record size.")
    if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the start token.")
    length, = struct.unpack("<Q", length_bytes)
    if length > len(datum_bytes):
        try:
            datum_bytes = datum_bytes.zfill(int(length * 1.5))
        except OverflowError:
            raise OverflowError('Error reading tfrecords; please try regenerating index files')
    datum_bytes_view = memoryview(datum_bytes)[:length]
    if file.readinto(datum_bytes_view) != length:
        raise RuntimeError("Failed to read the record.")
    if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the end token.")
    try:
        record = process(datum_bytes_view, description=feature_description)
    except KeyError:
        feature_description = {
            k: v for k, v in feature_description.items()
            if k in ('slide', 'image_raw')
        }
        try:
            record = process(datum_bytes_view, description=feature_description)
        except KeyError:
            msg = f'Unable to detect TFRecord format: {tfr}'
            raise errors.TFRecordsError(msg)

    img = bytes(record['image_raw'])
    img_type = imghdr.what('', img)
    return list(feature_description.keys()), img_type
