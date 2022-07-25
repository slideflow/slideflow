"""TFRecord reading/writing utilities for both Tensorflow and PyTorch."""

import copy
import os
import struct
from multiprocessing.dummy import Pool as DPool
from os.path import exists, isdir, isfile, join
from random import shuffle
from typing import Any, Dict, Optional, Tuple, Union

import slideflow as sf
from slideflow import errors
from slideflow.io.io_utils import detect_tfrecord_format, convert_dtype
from slideflow.util import log
from rich.progress import Progress

# --- Backend-specific imports and configuration ------------------------------

if sf.backend() == 'tensorflow':
    from slideflow.io.tensorflow import get_tfrecord_parser  # noqa F401
    from slideflow.io.tensorflow import read_and_return_record  # noqa F401
    from slideflow.io.tensorflow import serialized_record

    import tensorflow as tf
    from tensorflow.data import TFRecordDataset
    from tensorflow.io import TFRecordWriter
    dataloss_errors = [tf.errors.DataLossError, errors.TFRecordsError]

elif sf.backend() == 'torch':
    from slideflow.io.torch import \
        get_tfrecord_parser  # type: ignore  # noqa F401
    from slideflow.io.torch import read_and_return_record, serialized_record
    from slideflow.tfrecord import TFRecordWriter
    from slideflow.tfrecord.torch.dataset import TFRecordDataset
    dataloss_errors = [errors.TFRecordsError]

else:
    raise errors.UnrecognizedBackendError

# -----------------------------------------------------------------------------


def update_manifest_at_dir(
    directory: str,
    force_update: bool = False
) -> Optional[Union[str, Dict]]:
    '''Log number of tiles in each TFRecord file present in the given
    directory and all subdirectories, saving manifest to file within
    the parent directory.
    '''
    manifest_path = join(directory, "manifest.json")
    if not exists(manifest_path):
        manifest = {}
    else:
        manifest = sf.util.load_json(manifest_path)
    prior_manifest = copy.deepcopy(manifest)
    try:
        rel_paths = sf.util.get_relative_tfrecord_paths(directory)
    except FileNotFoundError:
        log.debug(f"Failed to update manifest {directory}; no TFRecords")
        return None

    # Verify all tfrecords in manifest exist
    for rel_tfr in prior_manifest.keys():
        tfr = join(directory, rel_tfr)
        if not exists(tfr):
            log.warning(f"TFRecord {tfr} in manifest was not found; removing")
            del(manifest[rel_tfr])

    def process_tfr(rel_tfr):
        tfr = join(directory, rel_tfr)
        if ((not force_update)
           and (rel_tfr in manifest)
           and ('total' in manifest[rel_tfr])):
            return None
        rel_tfr_manifest = {rel_tfr: {}}
        try:
            total = read_tfrecord_length(tfr)
        except dataloss_errors:
            log.error(f"Corrupt or incomplete TFRecord at {tfr}; removing")
            os.remove(tfr)
            return None
        if not total:
            log.error(f"Corrupt or incomplete TFRecord at {tfr}; removing")
            os.remove(tfr)
            return None
        rel_tfr_manifest[rel_tfr]['total'] = total
        return rel_tfr_manifest

    pool = DPool(8)
    if sf.getLoggingLevel() <= 20:
        pb = Progress(transient=True)
        task = pb.add_task("Verifying tfrecords...", total=len(rel_paths))
        pb.start()
    else:
        pb = None
    for m in pool.imap(process_tfr, rel_paths):
        if pb is not None:
            pb.advance(task)
        if m is None:
            continue
        manifest.update(m)
    if pb is not None:
        pb.stop()
    # Write manifest file
    if (manifest != prior_manifest) or (manifest == {}):
        sf.util.write_json(manifest, manifest_path)
    pool.close()
    return manifest


def get_tfrecord_by_index(
    tfrecord: str,
    index: int,
    decode: bool = True
) -> Any:
    '''Reads and returns an individual record from a tfrecord by index,
    including slide name and processed image data.
    '''
    if type(index) != int:
        try:
            index = int(index)
        except ValueError:
            raise IndexError(f"index must be an integer, not {type(index)} "
                             f"(provided {index}).")

    dataset = TFRecordDataset(tfrecord)
    parser = get_tfrecord_parser(
        tfrecord,
        ('slide', 'image_raw'),
        decode_images=decode
    )
    total = 0
    for i, record in enumerate(dataset):
        total += 1
        if i == index:
            return parser(record)  # type: ignore
        else:
            continue
    log.error(
        f"Unable to find record: index {index} in {sf.util.green(tfrecord)}"
        " ({total} total records)"
    )
    return False, False


def get_tfrecord_by_location(
    tfrecord: str,
    location: Tuple[int, int],
    decode: bool = True
) -> Any:
    '''Reads and returns an individual record from a tfrecord by index,
    including slide name and processed image data.
    '''
    if isinstance(location, list):
        location = tuple(location)
    if (not isinstance(location, tuple)
       or len(location) != 2
       or not isinstance(location[0], int)
       or not isinstance(location[1], int)):
        raise IndexError(f"index must be a tuple of two ints. Got: {location}")

    dataset = TFRecordDataset(tfrecord)
    parser = get_tfrecord_parser(
        tfrecord,
        ('slide', 'image_raw', 'loc_x', 'loc_y'),
        decode_images=decode
    )
    for i, record in enumerate(dataset):
        slide, image_raw, loc_x, loc_y = parser(record)
        if (loc_x, loc_y) == location:
            if decode:
                return slide, image_raw
            else:
                return record

    log.error(
        f"Unable to find record with location {location} in {tfrecord}"
    )
    return False, False


def write_tfrecords_multi(input_directory: str, output_directory: str) -> None:
    '''Scans a folder for subfolders, assumes subfolders are slide names.
    Assembles all image tiles within subfolders and labels using the provided
    annotation_dict, assuming the subfolder is the slide name. Collects all
    image tiles and exports into multiple tfrecord files, one for each slide.
    '''
    log.info("No location data available; writing (0,0) for all locations.")
    slide_dirs = [
        _dir for _dir in os.listdir(input_directory)
        if isdir(join(input_directory, _dir))
    ]
    total_tiles = 0
    for slide_dir in slide_dirs:
        total_tiles += write_tfrecords_single(
            join(input_directory, slide_dir),
            output_directory,
            f'{slide_dir}.tfrecords',
            slide_dir
        )
    log.info(
        f"Wrote {total_tiles} tiles across {len(slide_dirs)} tfrecords "
        f"in [green]{output_directory}"
    )


def write_tfrecords_single(
    input_directory: str,
    output_directory: str,
    filename: str,
    slide: str
) -> int:
    """Scans a folder for image tiles, annotates using the provided slide,
    exports into a single tfrecord file.

    Args:
        input_directory (str): Directory of images.
        output_directory (str): Directory in which to write TFRecord file.
        filename (str): TFRecord filename (without path).
        slide (str): Slide name to assign to records inside TFRecord.

    Returns:
        int: Number of records written.
    """
    if not exists(output_directory):
        os.makedirs(output_directory)
    tfrecord_path = join(output_directory, filename)
    image_labels = {}
    files = [
        f for f in os.listdir(input_directory)
        if ((isfile(join(input_directory, f)))
            and (sf.util.path_to_ext(f) in ("jpg", "jpeg", "png")))
    ]
    for tile in files:
        image_labels.update({
            join(input_directory, tile): bytes(slide, 'utf-8')
        })
    keys = list(image_labels.keys())
    shuffle(keys)
    writer = TFRecordWriter(tfrecord_path)
    for filename in keys:
        label = image_labels[filename]
        image_string = open(filename, 'rb').read()
        record = serialized_record(label, image_string, 0, 0)
        writer.write(record)
    writer.close()
    log.info(f"Wrote {len(keys)} images to {sf.util.green(tfrecord_path)}")
    return len(keys)


def write_tfrecords_merge(
    input_directory: str,
    output_directory: str,
    filename: str
) -> int:
    """Scans a folder for subfolders, assumes subfolders are slide names.
    Assembles all image tiles within subfolders and labels using the provided
    annotation_dict, assuming the subfolder is the slide name. Collects all
    image tiles and exports into a single tfrecord file.

    Args:
        input_directory (str): Directory of images.
        output_directory (str): Directory in which to write TFRecord file.
        filename (str): TFRecord filename (without path).

    Returns:
        int: Number of records written.
    """
    tfrecord_path = join(output_directory, filename)
    if not exists(output_directory):
        os.makedirs(output_directory)
    image_labels = {}
    slide_dirs = [
        _dir for _dir in os.listdir(input_directory)
        if isdir(join(input_directory, _dir))
    ]
    for slide_dir in slide_dirs:
        directory = join(input_directory, slide_dir)
        files = [
            f for f in os.listdir(directory)
            if ((isfile(join(directory, f)))
                and (sf.util.path_to_ext(f) in ("jpg", "jpeg", "png")))
            ]
        for tile in files:
            tgt = join(input_directory, slide_dir, tile)
            image_labels.update({
                tgt: bytes(slide_dir, 'utf-8')
            })
    keys = list(image_labels.keys())
    shuffle(keys)
    writer = TFRecordWriter(tfrecord_path)
    for filename in keys:
        label = image_labels[filename]
        image_string = open(filename, 'rb').read()
        record = serialized_record(label, image_string, 0, 0)
        writer.write(record)
    writer.close()
    log.info(f"Wrote {len(keys)} images to {sf.util.green(tfrecord_path)}")
    return len(keys)


def extract_tiles(tfrecord: str, destination: str) -> None:
    """Reads and saves images from a TFRecord to a destination folder.

    Args:
        tfrecord (str): Path to tfrecord.
        destination (str): Destination path to write loose images.

    Returns:
        None
    """
    if not exists(destination):
        os.makedirs(destination)
    log.info(f"Extracting tiles from tfrecord {sf.util.green(tfrecord)}")
    log.info(f"Saving tiles to directory {sf.util.green(destination)}")

    dataset = TFRecordDataset(tfrecord)
    _, img_type = detect_tfrecord_format(tfrecord)
    parser = get_tfrecord_parser(
        tfrecord,
        ('slide', 'image_raw'),
        to_numpy=True,
        decode_images=False
    )
    for i, record in enumerate(dataset):
        slide, image_raw = parser(record)  # type: ignore
        slidename = slide if type(slide) == str else slide.decode('utf-8')
        dest_folder = join(destination, slidename)
        if not exists(dest_folder):
            os.makedirs(dest_folder)
        tile_filename = f"tile{i}.{img_type}"
        image_string = open(join(dest_folder, tile_filename), 'wb')
        image_string.write(image_raw)
        image_string.close()


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
            log.error(f"Failed to parse TFRecord at {tfrecord}")
            infile.close()
            return 0
    infile.close()
    return num_records


def get_locations_from_tfrecord(filename: str) -> Dict[int, Tuple[int, int]]:
    '''Returns dictionary mapping indices to tile locations (X, Y)'''
    dataset = TFRecordDataset(filename)
    loc_dict = {}
    parser = get_tfrecord_parser(filename, ('loc_x', 'loc_y'), to_numpy=True)
    for i, record in enumerate(dataset):
        loc_x, loc_y = parser(record)  # type: ignore
        loc_dict.update({i: (loc_x, loc_y)})
    return loc_dict
