import imghdr
import os
import copy
import shutil
import logging
import slideflow as sf

from os import listdir
from os.path import isfile, isdir, join, exists
from random import shuffle, randint
from slideflow.util import log
from glob import glob
from functools import partial

logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'
FEATURE_TYPES = (tf.int64, tf.string, tf.string)

FEATURE_DESCRIPTION_LEGACY =  {'slide':    tf.io.FixedLenFeature([], tf.string),
                               'image_raw':tf.io.FixedLenFeature([], tf.string)}

FEATURE_DESCRIPTION = {'slide':        tf.io.FixedLenFeature([], tf.string),
                       'image_raw':    tf.io.FixedLenFeature([], tf.string),
                       'loc_x':        tf.io.FixedLenFeature([], tf.int64),
                       'loc_y':        tf.io.FixedLenFeature([], tf.int64)}

class TFRecordsError(Exception):
    pass

def _float_feature(value):
    """Returns a bytes_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _get_images_by_dir(directory):
    files = [f for f in listdir(directory) if (isfile(join(directory, f))) and
                (sf.util.path_to_ext(f) in ("jpg", "png"))]
    return files

def _read_and_return_record(record, feature_description, assign_slide=None):
    features = tf.io.parse_single_example(record, feature_description)
    read_features = {f:v.numpy() for f,v in features.items()}
    if assign_slide:
        read_features['slide'] = assign_slide
    tf_example = tfrecord_example(**read_features)
    return tf_example.SerializeToString()

def _print_record(filename):
    dataset = tf.data.TFRecordDataset(filename)
    parser = get_tfrecord_parser(filename, ('slide', 'loc_x', 'loc_y'), to_numpy=True, error_if_invalid=False)

    for i, record in enumerate(dataset):
        slide, loc_x, loc_y = parser(record)
        print(f"{sf.util.purple(filename)}: Record {i}: Slide: {sf.util.green(str(slide))} Loc: {(loc_x, loc_y)}")

def _decode_image(img_string, img_type, size=None, standardize=False, normalizer=None, augment=False):
    tf_decoders = {
        'png': tf.image.decode_png,
        'jpeg': tf.image.decode_jpeg,
        'jpg': tf.image.decode_jpeg
    }
    decoder = tf_decoders[img_type.lower()]
    image = decoder(img_string, channels=3)

    if normalizer:
        image = tf.py_function(normalizer.tf_to_rgb, [image], tf.int32)
        if size: image.set_shape([size, size, 3])
    if augment is True or (isinstance(augment, str) and 'j' in augment):
        # Augment with random compession
        image = tf.cond(tf.random.uniform(shape=[],
                                          minval=0,
                                          maxval=1,
                                          dtype=tf.float32) < 0.5,
                        true_fn=lambda: tf.image.adjust_jpeg_quality(image, tf.random.uniform(shape=[],
                                                                                              minval=50,
                                                                                              maxval=100,
                                                                                              dtype=tf.int32)),
                        false_fn=lambda: image)
    if augment is True or (isinstance(augment, str) and 'r' in augment):
        # Rotate randomly 0, 90, 180, 270 degrees
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        # Random flip and rotation
    if augment is True or (isinstance(augment, str) and 'x' in augment):
        image = tf.image.random_flip_left_right(image)
    if augment is True or (isinstance(augment, str) and 'y' in augment):
        image = tf.image.random_flip_up_down(image)
    if standardize:
        image = tf.image.per_image_standardization(image)
    if size:
        image.set_shape([size, size, 3])
    return image

def get_tfrecords_from_model_manifest(path_to_model):
    log.warning("Deprecation Warning: sf.io.tfrecords.get_tfrecord_from_model_manifest() will be removed " + \
                "in a future version. Please use sf.util.get_slides_from_model_manifest()")
    return sf.util.get_slides_from_model_manifest(path_to_model)

def detect_tfrecord_format(path):
    """Loads a tfrecord at the specified path, and detects the feature description and image type.

    Returns:
        dict: Feature description dictionary.
        str:  Stored image type, either 'png' or 'jpg'.
    """

    try:
        record = next(iter(tf.data.TFRecordDataset(path)))
    except StopIteration:
        log.warning(f"TFRecord {path} is empty.")
        return None, None
    try:
        features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
        for feature in FEATURE_DESCRIPTION:
            if feature not in features:
                raise tf.errors.InvalidArgumentError
        feature_description = FEATURE_DESCRIPTION
    except tf.errors.InvalidArgumentError:
        try:
            features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION_LEGACY)
            feature_description = FEATURE_DESCRIPTION_LEGACY
        except tf.errors.InvalidArgumentError:
            raise TFRecordsError(f"Unrecognized TFRecord format: {path}")
    image_type = imghdr.what('', features['image_raw'].numpy())
    return feature_description, image_type

def get_tfrecord_parser(tfrecord_path, features_to_return=None, to_numpy=False, decode_images=True,
                        standardize=False, img_size=None, normalizer=None, augment=False, error_if_invalid=True):

    """Returns a tfrecord parsing function based on the specified parameters.

    Args:
        tfrecord_path (str): Path to tfrecord to parse.
        features_to_return (list or dict, optional): Designates format for how features should be returned from parser.
            If a list of feature names is provided, the parsing function will return tfrecord features as a list
            in the order provided. If a dictionary of labels (keys) mapping to feature names (values) is provided,
            features will be returned from the parser as a dictionary matching the same format. If None, will
            return all features as a list.
        to_numpy (bool, optional): Convert records from tensors to numpy arrays. Defaults to False.
        decode_images (bool, optional): Decode raw image strings into image arrays. Defaults to True.
        standardize (bool, optional): Standardize images into the range (0,1). Defaults to False.
        img_size (int): Width of images in pixels. Will call tf.set_shape(...) if provided. Defaults to False.
        normalizer (:class:`slideflow.util.StainNormalizer`): Stain normalizer to use on images. Defaults to None.
        augment (str): Image augmentations to perform. String containing characters designating augmentations.
            'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
            at random quality levels. Passing either 'xyrj' or True will use all augmentations.
        error_if_invalid (bool, optional): Raise an error if a tfrecord cannot be read. Defaults to True.
    """

    feature_description, img_type = detect_tfrecord_format(tfrecord_path)
    if feature_description is None:
        log.warning(f"Unable to read tfrecord at {tfrecord_path} - is it empty?")
        return None
    if features_to_return is None:
        features_to_return = list(feature_description.keys())

    def parser(record):
        features = tf.io.parse_single_example(record, feature_description)

        def process_feature(f):
            if f not in features and error_if_invalid:
                raise TFRecordsError(f"Unknown feature {f} (available features: {', '.join(features)})")
            elif f not in features:
                return None
            elif f == 'image_raw' and decode_images:
                return _decode_image(features['image_raw'], img_type, img_size, standardize, normalizer, augment)
            elif to_numpy:
                return features[f].numpy()
            else:
                return features[f]

        if type(features_to_return) == dict:
            return {label: process_feature(f) for label, f in features_to_return.items()}
        else:
            return [process_feature(f) for f in features_to_return]

    return parser

def get_locations_from_tfrecord(filename):
    '''Returns dictionary mapping indices to tile locations (X, Y)'''

    dataset = tf.data.TFRecordDataset(filename)
    loc_dict = {}
    parser = get_tfrecord_parser(filename, ('loc_x', 'loc_y'), to_numpy=True)
    for i, record in enumerate(dataset):
        loc_x, loc_y = parser(record)
        loc_dict.update({ i: (loc_x, loc_y)    })
    return loc_dict

def interleave(tfrecords, img_size, batch_size, label_parser=None, model_type='categorical', balance=None,
                finite=False, annotations=None, max_tiles=0, min_tiles=0, augment=True, standardize=True,
                normalizer=None, manifest=None, slides=None, num_shards=None, shard_idx=None, num_parallel_reads=4):

    """Generates an interleaved dataset from a collection of tfrecord files, sampling from tfrecord files randomly
    according to balancing if provided. Requires manifest for balancing. Assumes TFRecord files are named by slide.

    Args:
        tfrecords (list(str)): List of paths to TFRecord files.
        img_size (int): Image width in pixels.
        batch_size (int): Batch size.
        label_parser (func, optional): Base function to use for parsing labels. Function must accept an image (tensor)
            and slide name (str), and return an image (tensor) and label. If None is provided, all labels will be None.
        model_type (str, optional): Model type. 'categorical' enables category-level balancing. Defaults to 'categorical'.
        balance (str, optional): Batch-level balancing. Options: BALANCE_BY_CATEGORY, BALANCE_BY_PATIENT, and NO_BALANCE.
            If finite option is used, will drop tiles in order to maintain proportions across the interleaved dataset.
        finite (bool, optional): Create a finite dataset iterating through tiles only once. WARNING: If finite option is
            used with balancing, some tiles will be skipped. Defaults to False (infinite dataset).
        annotations (dict, optional): Dict mapping slide names to outcome labels, used for balancing. Defaults to None.
        max_tiles (int, optional): Maximum number of tiles to use per slide. Defaults to 0 (use all tiles).
        min_tiles (int, optional): Minimum number of tiles that each slide must have to be included. Defaults to 0.
        augment (str, optional): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
        standardize (bool, optional): Standardize images to (0,1). Defaults to True.
        normalizer (:class:`slideflow.util.StainNormalizer`, optional): Normalizer to use on images. Defaults to None.
        manfest (dict, optional): Dataset manifest containing number of tiles per tfrecord.
        slides (list(str), optional): Only interleaves tfrecords with these slide names. If None, uses all tfrecords.
        num_shards (int, optional): Shard the tfrecord datasets, used for multiprocessing datasets. Defaults to None.
        shard_idx (int, optional): Index of the tfrecord shard to use. Defaults to None.
        num_parallel_reads (int, optional): Number of parallel reads for each TFRecordDataset. Defaults to 4.
    """

    log.debug(f'Interleaving {len(tfrecords)} tfrecords: finite={finite}, max_tiles={max_tiles}, min={min_tiles}')
    with tf.device('cpu'):
        datasets = []
        datasets_categories = []
        num_tiles = []
        global_num_tiles = 0
        categories = {}
        categories_prob = {}
        categories_tile_fraction = {}
        prob_weights = None
        base_parser = None
        detected_format = None
        num_tfrecords_empty = 0
        num_tfrecords_less_than_min = 0

        if label_parser is None:
            label_parser = default_label_parser

        if slides is None:
            slides = [sf.util.path_to_name(t) for t in tfrecords]

        if tfrecords == []:
            raise TFRecordsError('No TFRecords found.')

        if manifest:
            pb = sf.util.ProgressBar(len(tfrecords), counter_text='files', leadtext='Interleaving tfrecords... ')
            for filename in tfrecords:
                slide_name = sf.util.path_to_name(filename)

                if slide_name not in slides:
                    continue

                # Determine total number of tiles available in TFRecord
                try:
                    tiles = manifest[filename]['total']
                except KeyError:
                    log.error(f'Manifest not finished, unable to find {sf.util.green(filename)}')
                    raise TFRecordsError(f'Manifest not finished, unable to find {filename}')

                # Ensure TFRecord has minimum number of tiles; otherwise, skip
                if not min_tiles and tiles == 0:
                    num_tfrecords_empty += 1
                    continue
                elif tiles < min_tiles:
                    num_tfrecords_less_than_min += 1
                    continue

                # Get the base TFRecord parser, based on the first tfrecord
                if detected_format is None:
                    detected_format = detect_tfrecord_format(filename)
                elif detected_format != detect_tfrecord_format(filename):
                    raise TFRecordsError('Inconsistent TFRecord internal formatting; all must be formatted the same.')
                if base_parser is None:
                    base_parser = get_tfrecord_parser(filename,
                                                      ('image_raw', 'slide'),
                                                      standardize=standardize,
                                                      img_size=img_size,
                                                      normalizer=normalizer,
                                                      augment=augment)

                # Assign category by outcome if this is a categorical model,
                #    Merging category names if there are multiple outcomes
                #   (balancing across all combinations of outcome categories equally)
                # Otherwise, consider all slides from the same category (effectively skipping balancing).
                #   Appropriate for linear models.
                if model_type == 'categorical' and annotations is not None:
                    category = annotations[slide_name]
                    category = [category] if not isinstance(category, list) else category
                    category = '-'.join(map(str, category))
                elif model_type == 'categorical' and balance == BALANCE_BY_CATEGORY:
                    raise TFRecordsError('No annotations provided; unable to perform category-level balancing')
                else:
                    category = 1

                tf_dts = tf.data.TFRecordDataset(filename, num_parallel_reads=num_parallel_reads)
                if num_shards:
                    tf_dts = tf_dts.shard(num_shards, index=shard_idx)

                datasets += [tf_dts]
                datasets_categories += [category]

                # Cap number of tiles to take from TFRecord at maximum specified
                if max_tiles and tiles > max_tiles:
                    log.debug(f'Only taking maximum of {max_tiles} (of {tiles}) tiles from {sf.util.green(filename)}')
                    tiles = max_tiles

                if category not in categories.keys():
                    categories.update({category: {'num_slides': 1,
                                                'num_tiles': tiles}})
                else:
                    categories[category]['num_slides'] += 1
                    categories[category]['num_tiles'] += tiles
                num_tiles += [tiles]
                pb.increase_bar_value()
            pb.end()

            if num_tfrecords_empty:
                log.info(f'Skipped {num_tfrecords_empty} empty tfrecords')
            if num_tfrecords_less_than_min:
                log.info(f'Skipped {num_tfrecords_less_than_min} tfrecords with less than {min_tiles} tiles')

            for category in categories:
                lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
                lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
                categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']
                categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']

            # Balancing
            if not balance or balance == NO_BALANCE:
                log.debug(f'Not balancing input')
                prob_weights = [i/sum(num_tiles) for i in num_tiles]
            if balance == BALANCE_BY_PATIENT:
                log.debug(f'Balancing input across slides')
                prob_weights = [1.0] * len(datasets)
                if finite:
                    # Only take as many tiles as the number of tiles in the smallest dataset
                    minimum_tiles = min(num_tiles)
                    for i in range(len(datasets)):
                        num_tiles[i] = minimum_tiles
            if balance == BALANCE_BY_CATEGORY:
                log.debug(f'Balancing input across categories')
                prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
                if finite:
                    # Only take as many tiles as the number of tiles in the smallest category
                    for i in range(len(datasets)):
                        num_tiles[i] = int(num_tiles[i] * categories_tile_fraction[datasets_categories[i]])
                        fraction = categories_tile_fraction[datasets_categories[i]]
                        log.debug(f'Tile fraction (dataset {i+1}/{len(datasets)}): {fraction}, taking {num_tiles[i]}')
                    log.debug(f'Global num tiles: {global_num_tiles}')

            # Take the calculcated number of tiles from each dataset and calculate global number of tiles
            for i in range(len(datasets)):
                if max_tiles or (balance in (BALANCE_BY_PATIENT, BALANCE_BY_CATEGORY)):
                    to_take = num_tiles[i]
                    if num_shards:
                        to_take = to_take // num_shards
                    datasets[i] = datasets[i].take(to_take)
                if not finite:
                    datasets[i] = datasets[i].repeat()
            global_num_tiles = sum(num_tiles)

        else:
            manifest_msg = 'No manifest detected! Unable to perform balancing or any tile-level selection operations'
            if (balance and balance != NO_BALANCE) or max_tiles or min_tiles:
                log.error(manifest_msg)
            else:
                log.warning(manifest_msg)
            pb = sf.util.ProgressBar(len(tfrecords), counter_text='files', leadtext='Interleaving tfrecords... ')
            for filename in tfrecords:
                slide_name = sf.util.path_to_name(filename)

                if slide_name not in slides:
                    continue

                if base_parser is None:
                    base_parser = get_tfrecord_parser(filename,
                                                      ('image_raw', 'slide'),
                                                      standardize=standardize,
                                                      img_size=img_size,
                                                      normalizer=normalizer,
                                                      augment=augment)

                datasets += [tf.data.TFRecordDataset(filename, num_parallel_reads=num_parallel_reads)]
                pb.increase_bar_value()
            pb.end()

        # Interleave and batch datasets
        try:
            sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
            dataset = _get_parsed_datasets(sampled_dataset,
                                          label_parser=label_parser,
                                          base_parser=base_parser,
                                          include_slidenames=False)
            if batch_size:
                dataset = dataset.batch(batch_size, drop_remainder=False)
            #dataset = dataset.prefetch(tf.data.AUTOTUNE)
        except IndexError:
            raise TFRecordsError('No TFRecords found after filter criteria; please verify TFRecords exist')

        dataset_with_slidenames = _get_parsed_datasets(sampled_dataset,
                                                        label_parser=label_parser,
                                                        base_parser=base_parser,
                                                        include_slidenames=True)
        if batch_size:
            dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=False)
        #dataset_with_slidenames = dataset_with_slidenames.prefetch(tf.data.AUTOTUNE)

        return dataset, dataset_with_slidenames, global_num_tiles

def _get_parsed_datasets(tfrecord_dataset, label_parser, base_parser, include_slidenames=False):
    def final_parser(record, incl_slidenames):
        image, slide = base_parser(record)
        l_image, label = label_parser(image, slide)
        if incl_slidenames:
            return l_image, label, slide
        else:
            return l_image, label

    if include_slidenames:
        training_parser_with_slidenames = partial(final_parser, incl_slidenames=True)

        dataset_with_slidenames = tfrecord_dataset.map(training_parser_with_slidenames, num_parallel_calls=32)
        return dataset_with_slidenames
    else:
        training_parser = partial(final_parser, incl_slidenames=False)
        dataset = tfrecord_dataset.map(training_parser, num_parallel_calls=32)
        return dataset

def default_label_parser(image, slide):
    return image, None

def tfrecord_example(slide, image_raw, loc_x=0, loc_y=0):
    '''Returns a Tensorflow Data example for TFRecord storage.'''
    feature = {
        'slide':     _bytes_feature(slide),
        'image_raw':_bytes_feature(image_raw),
        'loc_x': _int64_feature(loc_x),
        'loc_y': _int64_feature(loc_y)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def multi_image_example(slide, image_dict):
    '''Returns a Tensorflow Data example for TFRecord storage with multiple images.'''
    feature = {
        'slide':    _bytes_feature(slide)
    }
    for image_label in image_dict:
        feature.update({
            image_label: _bytes_feature(image_dict[image_label])
        })
    return tf.train.Example(features=tf.train.Features(feature=feature))

def merge_split_tfrecords(source, destination):
    '''Merges TFRecords with the same name in subfolders within the given source folder,
    as may be the case when using split TFRecords for tile-level validation.'''
    tfrecords = {}
    subdirs = [d for d in listdir(source) if isdir(join(source, d))]
    for subdir in subdirs:
        tfrs = [tfr for tfr in listdir(join(source, subdir)) if isfile(join(source, subdir, tfr)) and tfr[-9:] == 'tfrecords']
        for tfr in tfrs:
            name = sf.util.path_to_name(tfr)
            if name not in tfrecords:
                tfrecords.update({name: [join(source, subdir, tfr)] })
            else:
                tfrecords[name] += [join(source, subdir, tfr)]
    for tfrecord_name in tfrecords:
        writer = tf.io.TFRecordWriter(join(destination, f'{tfrecord_name}.tfrecords'))
        datasets = []
        feature_description, img_type = detect_tfrecord_format(tfrecords.values()[0])
        for tfrecord in tfrecords[tfrecord_name]:
            n_feature_description, n_img_type = detect_tfrecord_format(tfrecord)
            if n_feature_description != feature_description or n_img_type != img_type:
                raise TFRecordsError("Mismatching tfrecord format found, unable to merge")
            dataset = tf.data.TFRecordDataset(tfrecord)
            dataset = dataset.shuffle(1000)
            dataset_iter = iter(dataset)
            datasets += [dataset_iter]
        while len(datasets):
            index = randint(0, len(datasets)-1)
            try:
                record = next(datasets[index])
            except StopIteration:
                del(datasets[index])
                continue
            writer.write(_read_and_return_record(record, feature_description, None))

def join_tfrecord(input_folder, output_file, assign_slide=None):
    '''Randomly samples from tfrecords in the input folder with shuffling,
    and combines into a single tfrecord file.'''
    writer = tf.io.TFRecordWriter(output_file)
    tfrecord_files = glob(join(input_folder, "*.tfrecords"))
    datasets = []
    if assign_slide: assign_slide = assign_slide.encode('utf-8')
    feature_description, img_type = detect_tfrecord_format(tfrecord_files[0])
    for tfrecord in tfrecord_files:
        n_feature_description, n_img_type = detect_tfrecord_format(tfrecord)
        if n_feature_description != feature_description or n_img_type != img_type:
            raise TFRecordsError("Mismatching tfrecord format found, unable to merge")
        dataset = tf.data.TFRecordDataset(tfrecord)
        dataset = dataset.shuffle(1000)
        dataset_iter = iter(dataset)
        datasets += [dataset_iter]
    while len(datasets):
        index = randint(0, len(datasets)-1)
        try:
            record = next(datasets[index])
        except StopIteration:
            del(datasets[index])
            continue
        writer.write(_read_and_return_record(record, feature_description, assign_slide))

def split_tfrecord(tfrecord_file, output_folder):
    '''Splits records from a single tfrecord file into individual tfrecord files by slide.'''
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    feature_description, _ = detect_tfrecord_format(tfrecord_file)
    parser = get_tfrecord_parser(tfrecord_file, ['slide'], to_numpy=True)
    writers = {}
    for record in dataset:
        slide = parser(record)
        shortname = sf.util._shortname(slide.decode('utf-8'))

        if shortname not in writers.keys():
            tfrecord_path = join(output_folder, f"{shortname}.tfrecords")
            writer = tf.io.TFRecordWriter(tfrecord_path)
            writers.update({shortname: writer})
        else:
            writer = writers[shortname]
        writer.write(_read_and_return_record(record, feature_description))

    for slide in writers.keys():
        writers[slide].close()

def print_tfrecord(target):
    '''Prints the slide names (and locations, if present) for records in the given tfrecord file'''
    if isfile(target):
        _print_record(target)
    else:
        tfrecord_files = glob(join(target, "*.tfrecords"))
        for tfr in tfrecord_files:
            _print_record(tfr)

def write_tfrecords_merge(input_directory, output_directory, filename):
    '''Scans a folder for subfolders, assumes subfolders are slide names. Assembles all image tiles within
    subfolders and labels using the provided annotation_dict, assuming the subfolder is the slide name.
    Collects all image tiles and exports into a single tfrecord file.'''
    tfrecord_path = join(output_directory, filename)
    if not exists(output_directory):
        os.makedirs(output_directory)
    image_labels = {}
    slide_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
    for slide_dir in slide_dirs:
        files = _get_images_by_dir(join(input_directory, slide_dir))
        for tile in files:
            image_labels.update({join(input_directory, slide_dir, tile): bytes(slide_dir, 'utf-8')})
    keys = list(image_labels.keys())
    shuffle(keys)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for filename in keys:
            label = image_labels[filename]
            image_string = open(filename, 'rb').read()
            tf_example = tfrecord_example(label, image_string)
            writer.write(tf_example.SerializeToString())
    log.info(f"Wrote {len(keys)} image tiles to {sf.util.green(tfrecord_path)}")
    return len(keys)

def write_tfrecords_multi(input_directory, output_directory):
    '''Scans a folder for subfolders, assumes subfolders are slide names. Assembles all image tiles within
    subfolders and labels using the provided annotation_dict, assuming the subfolder is the slide name.
    Collects all image tiles and exports into multiple tfrecord files, one for each slide.'''
    slide_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
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
    files = _get_images_by_dir(input_directory)
    for tile in files:
        image_labels.update({join(input_directory, tile): bytes(slide, 'utf-8')})
    keys = list(image_labels.keys())
    shuffle(keys)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for filename in keys:
            label = image_labels[filename]
            image_string = open(filename, 'rb').read()
            tf_example = tfrecord_example(label, image_string)
            writer.write(tf_example.SerializeToString())
    log.info(f"Wrote {len(keys)} image tiles to {sf.util.green(tfrecord_path)}")
    return len(keys)

def checkpoint_to_tf_model(models_dir, model_name):
    '''Converts a checkpoint file into a saved model.'''

    checkpoint = join(models_dir, model_name, "cp.ckpt")
    tf_model = join(models_dir, model_name, "untrained_model")
    updated_tf_model = join(models_dir, model_name, "checkpoint_model")
    model = tf.keras.models.load_model(tf_model)
    model.load_weights(checkpoint)
    try:
        model.save(updated_tf_model)
    except KeyError:
        # Not sure why this happens, something to do with the optimizer?
        pass

def update_tfrecord_dir(directory, old_feature_description=FEATURE_DESCRIPTION, slide='slide', assign_slide=None,
                        image_raw='image_raw'):

    '''Updates tfrecords in a directory from an old format to a new format.'''

    if not exists(directory):
        log.error(f"Directory {directory} does not exist; unable to update tfrecords.")
    else:
        tfrecord_files = glob(join(directory, "*.tfrecords"))
        for tfr in tfrecord_files:
            update_tfrecord(tfr, old_feature_description, slide, assign_slide, image_raw)
        return len(tfrecord_files)

def update_tfrecord(tfrecord_file, assign_slide=None, image_raw='image_raw'):

    '''Updates a single tfrecord from an old format to a new format.'''

    shutil.move(tfrecord_file, tfrecord_file+".old")
    dataset = tf.data.TFRecordDataset(tfrecord_file+".old")
    writer = tf.io.TFRecordWriter(tfrecord_file)
    feature_description, _ = detect_tfrecord_format(tfrecord_file+'.old')
    for record in dataset:
        slidename = bytes(assign_slide, 'utf-8') if assign_slide else None
        writer.write(_read_and_return_record(record, feature_description, assign_slide=slidename))
    writer.close()
    os.remove(tfrecord_file+'.old')

def transform_tfrecord(origin,target, assign_slide=None, hue_shift=None, resize=None, silent=False):

    '''Transforms images in a single tfrecord. Can perform hue shifting, resizing, or re-assigning slide label.'''

    print_func = None if silent else print
    log.info(f"Transforming tiles in tfrecord {sf.util.green(origin)}")
    log.info(f"Saving to new tfrecord at {sf.util.green(target)}")
    if assign_slide:
        log.info(f"Assigning slide name {sf.util.bold(assign_slide)}")
    if hue_shift:
        log.info(f"Shifting hue by {sf.util.bold(str(hue_shift))}")
    if resize:
        log.info(f"Resizing records to ({resize}, {resize})")

    dataset = tf.data.TFRecordDataset(origin)
    writer = tf.io.TFRecordWriter(target)
    parser = get_tfrecord_parser(origin, ('slide', 'image_raw', 'loc_x', 'loc_y'), error_if_invalid=False, to_numpy=True)

    def process_image(image_string):
        if hue_shift:
            decoded_image = tf.image.decode_png(image_string, channels=3)
            adjusted_image = tf.image.adjust_hue(decoded_image, hue_shift)
            encoded_image = tf.io.encode_jpeg(adjusted_image, quality=80)
            return encoded_image.numpy()
        elif resize:
            decoded_image = tf.image.decode_png(image_string, channels=3)
            resized_image = tf.image.resize(decoded_image, (resize, resize),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            encoded_image = tf.io.encode_jpeg(resized_image, quality=80)
            return encoded_image.numpy()
        else:
            return image_string

    for record in dataset:
        slide, image_raw, loc_x, loc_y = parser(record)
        slidename = slide if not assign_slide else bytes(assign_slide, 'utf-8')
        image_processed_data = process_image(image_raw)
        tf_example = tfrecord_example(slidename, image_processed_data, loc_x, loc_y)
        writer.write(tf_example.SerializeToString())
    writer.close()

def shuffle_tfrecord(target):

    '''Shuffles records in a TFRecord, saving the original to a .old file.'''

    old_tfrecord = target+".old"
    shutil.move(target, old_tfrecord)

    dataset = tf.data.TFRecordDataset(old_tfrecord)
    writer = tf.io.TFRecordWriter(target)

    extracted_tfrecord = []
    for record in dataset:
        extracted_tfrecord += [record.numpy()]

    shuffle(extracted_tfrecord)

    for record in extracted_tfrecord:
        writer.write(record)

    writer.close()

def shuffle_tfrecords_by_dir(directory):

    '''For each TFRecord in a directory, shuffles records in the TFRecord, saving the original to a .old file.'''

    records = [tfr for tfr in listdir(directory) if tfr[-10:] == ".tfrecords"]
    for record in records:
        log.info(f'Working on {record}')
        shuffle_tfrecord(join(directory, record))

def get_tfrecord_by_index(tfrecord, index, decode=True):

    '''Reads and returns an individual record from a tfrecord by index, including slide name and processed image data.'''

    if type(index) != int:
        try:
            index = int(index)
        except:
            raise IndexError(f"index must be an integer, not {type(index)} (provided {index}).")

    dataset = tf.data.TFRecordDataset(tfrecord)
    parser = get_tfrecord_parser(tfrecord, ('slide', 'image_raw'), decode_images=decode)

    total = 0
    for i, record in enumerate(dataset):
        total += 1
        if i == index:
            return parser(record)
        else: continue

    log.error(f"Unable to find record at index {index} in {sf.util.green(tfrecord)} ({total} total records)")
    return False, False

def extract_tiles(tfrecord, destination, description=FEATURE_DESCRIPTION, feature_label='image_raw'):

    '''Reads and saves images from a TFRecord to a destination folder.'''

    if not exists(destination):
        os.makedirs(destination)
    log.info(f"Extracting tiles from tfrecord {sf.util.green(tfrecord)}")
    log.info(f"Saving tiles to directory {sf.util.green(destination)}")

    dataset = tf.data.TFRecordDataset(tfrecord)
    _, img_type = detect_tfrecord_format(tfrecord)
    parser = get_tfrecord_parser(tfrecord, ('slide', 'image_raw'), to_numpy=True, decode_images=False)
    for i, record in enumerate(dataset):
        slide, image_raw = parser(record)
        slidename = slide.decode('utf-8')
        dest_folder = join(destination, slidename)
        if not exists(dest_folder):
            os.makedirs(dest_folder)
        tile_filename = f"tile{i}.{img_type}"
        image_string = open(join(dest_folder, tile_filename), 'wb')
        image_string.write(image_raw)
        image_string.close()

def update_manifest_at_dir(directory, force_update=False):

    '''Log number of tiles in each TFRecord file present in the given directory and all subdirectories,
    saving manifest to file within the parent directory.'''

    manifest_path = join(directory, "manifest.json")
    manifest = {} if not exists(manifest_path) else sf.util.load_json(manifest_path)
    prior_manifest = copy.deepcopy(manifest)
    try:
        relative_tfrecord_paths = sf.util.get_relative_tfrecord_paths(directory)
    except FileNotFoundError:
        log.warning(f"Unable to find TFRecords in the directory {directory}")
        return

    # Verify all tfrecords in manifest exist
    for rel_tfr in prior_manifest.keys():
        tfr = join(directory, rel_tfr)
        if not exists(tfr):
            log.warning(f"TFRecord in manifest was not found at {tfr}; removing")
            del(manifest[rel_tfr])

    for rel_tfr in relative_tfrecord_paths:
        tfr = join(directory, rel_tfr)

        if (not force_update) and (rel_tfr in manifest) and ('total' in manifest[rel_tfr]):
            continue

        manifest.update({rel_tfr: {}})
        try:
            raw_dataset = tf.data.TFRecordDataset(tfr)
        except Exception as e:
            log.error(f"Unable to open TFRecords file with Tensorflow: {str(e)}")
            return
        if log.getEffectiveLevel() <= 20: print(f"\r\033[K + Verifying tiles in {sf.util.green(rel_tfr)}...", end="")
        total = 0
        try:
            #TODO: consider updating this to use sf.io.tfrecords.get_tfrecord_parser()
            for raw_record in raw_dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')
                if slide not in manifest[rel_tfr]:
                    manifest[rel_tfr][slide] = 1
                else:
                    manifest[rel_tfr][slide] += 1
                total += 1
        except tf.errors.DataLossError:
            print('\r\033[K', end="")
            log.error(f"Corrupt or incomplete TFRecord at {tfr}")
            log.info(f"Deleting and removing corrupt TFRecord from manifest...")
            del(raw_dataset)
            os.remove(tfr)
            del(manifest[rel_tfr])
            continue
        manifest[rel_tfr]['total'] = total
        print('\r\033[K', end="")

    # Write manifest file
    if (manifest != prior_manifest) or (manifest == {}):
        sf.util.write_json(manifest, manifest_path)

    return manifest