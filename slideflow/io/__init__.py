"""TFRecord reading/writing utilities for both Tensorflow and PyTorch."""

import copy
import os
import struct
import numpy as np
from multiprocessing.dummy import Pool as DPool
from os.path import exists, isdir, isfile, join
from random import shuffle
from typing import Any, Dict, Optional, Tuple, Union, List

import slideflow as sf
from slideflow import errors
from slideflow.io.io_utils import detect_tfrecord_format, convert_dtype
from slideflow.util import log, tfrecord2idx
from slideflow.util.tfrecord2idx import get_tfrecord_by_index, get_tfrecord_length
from rich.progress import Progress

# --- Backend-specific imports and configuration ------------------------------

if sf.backend() == 'tensorflow':
    from slideflow.io.tensorflow import (
        get_tfrecord_parser, read_and_return_record, serialized_record
    )
    from slideflow.io.tensorflow import auto_decode_image as decode_image
    from tensorflow.data import TFRecordDataset
    from tensorflow.io import TFRecordWriter

elif sf.backend() == 'torch':
    from slideflow.io.torch import (
        get_tfrecord_parser, read_and_return_record, serialized_record,
        decode_image
    )
    from slideflow.tfrecord import TFRecordWriter
    from slideflow.tfrecord.torch.dataset import TFRecordDataset

else:
    raise errors.UnrecognizedBackendError

# -----------------------------------------------------------------------------


def update_manifest_at_dir(
    directory: str,
    force_update: bool = False
) -> Optional[Union[str, Dict]]:
    """Log number of tiles in each TFRecord file present in the given
    directory and all subdirectories, saving manifest to file within
    the parent directory.

    """
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
            total = get_tfrecord_length(tfr)
        except (errors.TFRecordsError, OSError):
            log.error(f"Corrupt or incomplete TFRecord at {tfr}; removing")
            os.remove(tfr)
            return None
        if not total:
            log.error(f"Empty TFRecord at {tfr}; removing")
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
    with sf.util.cleanup_progress(pb):
        for m in pool.imap(process_tfr, rel_paths):
            if pb is not None:
                pb.advance(task)
            if m is None:
                continue
            manifest.update(m)
    # Write manifest file
    if (manifest != prior_manifest) or (manifest == {}):
        sf.util.write_json(manifest, manifest_path)
    pool.close()
    return manifest


def get_tfrecord_by_location(
    tfrecord: str,
    location: Tuple[int, int],
    decode: bool = True,
    *,
    locations_array: Optional[List[Tuple[int, int]]] = None,
    index_array: Optional[np.ndarray] = None
) -> Any:
    '''Reads and returns an individual record from a tfrecord by index,
    including slide name and processed image data.

    Args:
        tfrecord (str): Path to TFRecord file.
        location (tuple(int, int)): ``(x, y)`` tile location.
            Searches the TFRecord for the tile that corresponds to this
            location.
        decode (bool): Decode the associated record, returning Tensors.
            Defaults to True.

    Returns:
        Unprocessed raw TFRecord bytes if ``decode=False``, otherwise a
        tuple containing ``(slide, image)``, where ``image`` is a
        uint8 Tensor.
    '''
    if isinstance(location, list):
        location = tuple(location)
    if (not isinstance(location, tuple)
       or len(location) != 2
       or not isinstance(location[0], (int, np.integer))
       or not isinstance(location[1], (int, np.integer))):
        raise IndexError(f"index must be a tuple of two ints. Got: {location}")

    # Use index files, if available.
    index = tfrecord2idx.find_index(tfrecord)
    if locations_array is not None or (index and tfrecord2idx.index_has_locations(index)):
        if locations_array is None:
            locations = tfrecord2idx.get_locations_from_index(index)
        else:
            locations = locations_array
        try:
            idx = locations.index(location)
        except ValueError:
            log.error(
                f"Unable to find record with location {location} in {tfrecord}"
            )
            return False, False
        record = tfrecord2idx.get_tfrecord_by_index(tfrecord, idx, index_array=index_array)
        slide = record['slide']
        image = sf.io.decode_image(record['image_raw']) if decode else record['image_raw']
        return slide, image

    else:
        parser = get_tfrecord_parser(
            tfrecord,
            ('slide', 'image_raw', 'loc_x', 'loc_y'),
            decode_images=decode
        )
        dataset = TFRecordDataset(tfrecord)
        for i, record in enumerate(dataset):
            slide, image, loc_x, loc_y = parser(record)
            if (loc_x, loc_y) == location:
                if decode:
                    return slide, image
                else:
                    slide = bytes(record['slide']).decode('utf-8')
                    images = bytes(record['image_raw'])
                    return slide, images

        log.error(
            f"Unable to find record with location {location} in {tfrecord}"
        )
        return False, False


def write_tfrecords_multi(input_directory: str, output_directory: str) -> None:
    """Write multiple tfrecords, one for each slide, from a directory of images.

    Scans a folder for subfolders, assumes subfolders are slide names.
    Assembles all image tiles within subfolders, assuming the subfolder is the
    slide name. Collects all image tiles and exports into multiple tfrecord
    files, one for each slide.

    Args:
        input_directory (str): Directory of images.
        output_directory (str): Directory in which to write TFRecord files.

    """
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
    """Extracts images within a TFRecord to a destination folder.

    Args:
        tfrecord (str): Path to tfrecord.
        destination (str): Destination path to write loose images.

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


def get_locations_from_tfrecord(filename: str) -> List[Tuple[int, int]]:
    """Return list of tile locations (X, Y) for all items in the TFRecord."""

    # Use the TFRecord index file, if one exists and it has info stored.
    index = tfrecord2idx.find_index(filename)
    if index and tfrecord2idx.index_has_locations(index):
        return tfrecord2idx.get_locations_from_index(index)

    # Otherwise, read the TFRecord manually.
    out_list = []
    for i in range(sf.io.get_tfrecord_length(filename)):
        record = sf.io.get_tfrecord_by_index(filename, i)
        loc_x = record['loc_x']
        loc_y = record['loc_y']
        out_list.append((loc_x, loc_y))
    return out_list


def tfrecord_has_locations(
    filename: str,
    check_x: int = True,
    check_y: bool = False
) -> bool:
    """Check if a given TFRecord has location information stored."""
    index = tfrecord2idx.find_index(filename)
    if index and tfrecord2idx.index_has_locations(index):
        if check_y:
            return np.load(index)['locations'].shape[1] == 2
        return True
    record = sf.io.get_tfrecord_by_index(filename, 0)
    return (((not check_x) or 'loc_x' in record ) and ((not check_y) or 'loc_y' in record ))
