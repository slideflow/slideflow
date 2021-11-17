"""TFRecord reading/writing utilities for both Tensorflow and PyTorch."""

import os
import copy
import struct
import slideflow as sf

from tqdm import tqdm
from multiprocessing.dummy import Pool as DPool
from random import shuffle
from slideflow.util import log
from os.path import join, exists, isdir, isfile

# Backend-specific imports and configuration
if os.environ['SF_BACKEND'] == 'tensorflow':
    import tensorflow as tf
    from slideflow.io.tensorflow import get_tfrecord_parser, detect_tfrecord_format, serialized_record, \
                                        read_and_return_record, TFRecordsError
    from tensorflow.data import TFRecordDataset
    from tensorflow.io import TFRecordWriter
    dataloss_errors = (tf.errors.DataLossError, TFRecordsError)

elif os.environ['SF_BACKEND'] == 'torch':
    from slideflow.io.torch import get_tfrecord_parser, detect_tfrecord_format, serialized_record, \
                                   read_and_return_record, TFRecordsError
    from slideflow.tfrecord.torch.dataset import TFRecordDataset
    from slideflow.tfrecord import TFRecordWriter
    dataloss_errors = (TFRecordsError,)

else:
    raise ValueError(f"Unknown backend {os.environ['SF_BACKEND']}")

def update_manifest_at_dir(directory, force_update=False):
    '''Log number of tiles in each TFRecord file present in the given directory and all subdirectories,
    saving manifest to file within the parent directory.'''

    manifest_path = join(directory, "manifest.json")
    manifest = {} if not exists(manifest_path) else sf.util.load_json(manifest_path)
    prior_manifest = copy.deepcopy(manifest)
    try:
        relative_tfrecord_paths = sf.util.get_relative_tfrecord_paths(directory)
    except FileNotFoundError:
        log.debug(f"Unable to update manifest at {directory}; TFRecords not found")
        return

    # Verify all tfrecords in manifest exist
    for rel_tfr in prior_manifest.keys():
        tfr = join(directory, rel_tfr)
        if not exists(tfr):
            log.warning(f"TFRecord in manifest was not found at {tfr}; removing")
            del(manifest[rel_tfr])

    def process_tfr(rel_tfr):
        tfr = join(directory, rel_tfr)

        if (not force_update) and (rel_tfr in manifest) and ('total' in manifest[rel_tfr]):
            return None

        rel_tfr_manifest = {rel_tfr: {}}
        try:
            total = read_tfrecord_length(tfr)
        except dataloss_errors:
            return 'delete'
        if total is None:
            return 'delete'
        rel_tfr_manifest[rel_tfr]['total'] = total
        return rel_tfr_manifest

    pool = DPool(8)
    if log.getEffectiveLevel() <= 20:
        pb = tqdm(desc='Verifying tfrecords...', total=len(relative_tfrecord_paths), leave=False)
    else:
        pb = None
    for m in pool.imap(process_tfr, relative_tfrecord_paths):
        if pb is not None:
            pb.update()
        if m is None:
            continue
        if m == 'delete':
            log.error(f"Corrupt or incomplete TFRecord at {tfr}; removing")
            os.remove(tfr)
            continue
        manifest.update(m)

    # Write manifest file
    if (manifest != prior_manifest) or (manifest == {}):
        sf.util.write_json(manifest, manifest_path)

    return manifest

def get_tfrecord_by_index(tfrecord, index, decode=True):
    '''Reads and returns an individual record from a tfrecord by index, including slide name and processed image data.'''

    if type(index) != int:
        try:
            index = int(index)
        except:
            raise IndexError(f"index must be an integer, not {type(index)} (provided {index}).")

    dataset = TFRecordDataset(tfrecord)
    parser = get_tfrecord_parser(tfrecord, ('slide', 'image_raw'), decode_images=decode)

    total = 0
    for i, record in enumerate(dataset):
        total += 1
        if i == index:
            return parser(record)
        else: continue

    log.error(f"Unable to find record at index {index} in {sf.util.green(tfrecord)} ({total} total records)")
    return False, False

def write_tfrecords_multi(input_directory, output_directory):
    '''Scans a folder for subfolders, assumes subfolders are slide names. Assembles all image tiles within
    subfolders and labels using the provided annotation_dict, assuming the subfolder is the slide name.
    Collects all image tiles and exports into multiple tfrecord files, one for each slide.'''
    log.info("No location data available; writing (0,0) for all tile locations.")
    slide_dirs = [_dir for _dir in os.listdir(input_directory) if isdir(join(input_directory, _dir))]
    total_tiles = 0
    for slide_dir in slide_dirs:
        total_tiles += write_tfrecords_single(join(input_directory, slide_dir),
                                              output_directory,
                                              f'{slide_dir}.tfrecords',
                                              slide_dir)
    msg_num_tiles = sf.util.bold(total_tiles)
    msg_num_tfr = sf.util.bold(len(slide_dirs))
    log.info(f"Wrote {msg_num_tiles} tiles across {msg_num_tfr} tfrecords in {sf.util.green(output_directory)}")

def write_tfrecords_single(input_directory, output_directory, filename, slide):
    '''Scans a folder for image tiles, annotates using the provided slide, exports
    into a single tfrecord file.'''
    if not exists(output_directory):
        os.makedirs(output_directory)
    tfrecord_path = join(output_directory, filename)
    image_labels = {}
    files = [f for f in os.listdir(input_directory) if (isfile(join(input_directory, f))) and
            (sf.util.path_to_ext(f) in ("jpg", "png"))]
    for tile in files:
        image_labels.update({join(input_directory, tile): bytes(slide, 'utf-8')})
    keys = list(image_labels.keys())
    shuffle(keys)
    writer = TFRecordWriter(tfrecord_path)
    for filename in keys:
        label = image_labels[filename]
        image_string = open(filename, 'rb').read()
        record = serialized_record(label, image_string, 0, 0)
        writer.write(record)
    writer.close()
    log.info(f"Wrote {len(keys)} image tiles to {sf.util.green(tfrecord_path)}")
    return len(keys)

def write_tfrecords_merge(input_directory, output_directory, filename):
    '''Scans a folder for subfolders, assumes subfolders are slide names. Assembles all image tiles within
    subfolders and labels using the provided annotation_dict, assuming the subfolder is the slide name.
    Collects all image tiles and exports into a single tfrecord file.'''
    tfrecord_path = join(output_directory, filename)
    if not exists(output_directory):
        os.makedirs(output_directory)
    image_labels = {}
    slide_dirs = [_dir for _dir in os.listdir(input_directory) if isdir(join(input_directory, _dir))]
    for slide_dir in slide_dirs:
        directory = join(input_directory, slide_dir)
        files = [f for f in os.listdir(directory) if (isfile(join(directory, f))) and
                (sf.util.path_to_ext(f) in ("jpg", "png"))]

        for tile in files:
            image_labels.update({join(input_directory, slide_dir, tile): bytes(slide_dir, 'utf-8')})
    keys = list(image_labels.keys())
    shuffle(keys)
    writer = TFRecordWriter(tfrecord_path)
    for filename in keys:
        label = image_labels[filename]
        image_string = open(filename, 'rb').read()
        record = serialized_record(label, image_string, 0, 0)
        writer.write(record)
    writer.close()
    log.info(f"Wrote {len(keys)} image tiles to {sf.util.green(tfrecord_path)}")
    return len(keys)

def extract_tiles(tfrecord, destination):
    '''Reads and saves images from a TFRecord to a destination folder.'''

    if not exists(destination):
        os.makedirs(destination)
    log.info(f"Extracting tiles from tfrecord {sf.util.green(tfrecord)}")
    log.info(f"Saving tiles to directory {sf.util.green(destination)}")

    dataset = TFRecordDataset(tfrecord)
    _, img_type = detect_tfrecord_format(tfrecord)
    parser = get_tfrecord_parser(tfrecord, ('slide', 'image_raw'), to_numpy=True, decode_images=False)
    for i, record in enumerate(dataset):
        slide, image_raw = parser(record)
        slidename = slide if type(slide) == str else slide.decode('utf-8')
        dest_folder = join(destination, slidename)
        if not exists(dest_folder):
            os.makedirs(dest_folder)
        tile_filename = f"tile{i}.{img_type}"
        image_string = open(join(dest_folder, tile_filename), 'wb')
        image_string.write(image_raw)
        image_string.close()

def read_tfrecord_length(tfrecord):
    """Returns number of records stored in the given tfrecord file."""
    infile = open(tfrecord, "rb")
    num_records = 0
    while True:
        current = infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            num_records += 1
        except:
            log.error(f"Failed to parse TFRecord at {tfrecord}")
            infile.close()
            return None
    infile.close()
    return num_records

def get_locations_from_tfrecord(filename):
    '''Returns dictionary mapping indices to tile locations (X, Y)'''

    dataset = TFRecordDataset(filename)
    loc_dict = {}
    parser = get_tfrecord_parser(filename, ('loc_x', 'loc_y'), to_numpy=True)
    for i, record in enumerate(dataset):
        loc_x, loc_y = parser(record)
        loc_dict.update({ i: (loc_x, loc_y)    })
    return loc_dict