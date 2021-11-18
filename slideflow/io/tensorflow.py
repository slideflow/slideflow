import imghdr
import os
import shutil
import logging
import slideflow as sf
import numpy as np

from tqdm import tqdm
from os import listdir
from os.path import isfile, isdir, join, exists
from random import shuffle, randint
from slideflow.util import log
from glob import glob

logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

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

def read_and_return_record(record, parser, assign_slide=None):
    features = parser(record)
    if assign_slide:
        features['slide'] = assign_slide
    tf_example = tfrecord_example(**features)
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
        image = tf.cond(tf.random.uniform(shape=[], # pylint: disable=unexpected-keyword-arg
                                          minval=0,
                                          maxval=1,
                                          dtype=tf.float32) < 0.5,
                        true_fn=lambda: tf.image.adjust_jpeg_quality(image, tf.random.uniform(shape=[], # pylint: disable=unexpected-keyword-arg
                                                                                              minval=50,
                                                                                              maxval=100,
                                                                                              dtype=tf.int32)),
                        false_fn=lambda: image)
    if augment is True or (isinstance(augment, str) and 'r' in augment):
        # Rotate randomly 0, 90, 180, 270 degrees
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)) # pylint: disable=unexpected-keyword-arg
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
    log.warning("Deprecation Warning: sf.io.tensorflow.get_tfrecord_from_model_manifest() will be removed " + \
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
        log.debug(f"TFRecord {path} is empty.")
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
        normalizer (:class:`slideflow.slide.StainNormalizer`): Stain normalizer to use on images. Defaults to None.
        augment (str): Image augmentations to perform. String containing characters designating augmentations.
            'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
            at random quality levels. Passing either 'xyrj' or True will use all augmentations.
        error_if_invalid (bool, optional): Raise an error if a tfrecord cannot be read. Defaults to True.
    """

    feature_description, img_type = detect_tfrecord_format(tfrecord_path)
    if feature_description is None:
        log.debug(f"Unable to read tfrecord at {tfrecord_path} - is it empty?")
        return None
    if features_to_return is None:
        features_to_return = {k:k for k in feature_description.keys()}

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

def parser_from_labels(labels):
    '''Returns a label parsing function used for parsing slides into single or multi-outcome labels.'''

    outcome_labels = np.array(list(labels.values()))
    slides = list(labels.keys())
    if len(outcome_labels.shape) == 1:
        outcome_labels = np.expand_dims(outcome_labels, axis=1)
    with tf.device('/cpu'):
        annotations_tables = []
        for oi in range(outcome_labels.shape[1]):
            annotations_tables += [tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(slides, outcome_labels[:,oi]), -1
            )]

    def label_parser(image, slide):
        if outcome_labels.shape[1] > 1:
            label = [annotations_tables[oi].lookup(slide) for oi in range(outcome_labels.shape[1])]
        else:
            label = annotations_tables[0].lookup(slide)
        return image, label

    return label_parser

def interleave(tfrecords, img_size, batch_size, prob_weights=None, clip=None, labels=None, incl_slidenames=False,
               incl_loc=False, infinite=True, augment=True, standardize=True, normalizer=None, num_shards=None,
               shard_idx=None, num_parallel_reads=4):

    """Generates an interleaved dataset from a collection of tfrecord files, sampling from tfrecord files randomly
    according to balancing if provided. Requires manifest for balancing. Assumes TFRecord files are named by slide.

    Args:
        tfrecords (list(str)): List of paths to TFRecord files.
        img_size (int): Image width in pixels.
        batch_size (int): Batch size.
        prob_weights (dict, optional): Dict mapping tfrecords to probability of including in batch. Defaults to None.
        clip (dict, optional): Dict mapping tfrecords to number of tiles to take per tfrecord. Defaults to None.
        labels (dict or str, optional): Dict or function. If dict, must map slide names to outcome labels.
                If function, function must accept an image (tensor) and slide name (str), and return a dict
                {'image_raw': image (tensor)} and label (int or float). If not provided,  all labels will be None.
        incl_slidenames (bool, optional): Include slidenames as third returned variable. Defaults to False.
        incl_loc (bool, optional): Include loc_x and loc_y as additional returned variables. Defaults to False.
        infinite (bool, optional): Create an finite dataset. WARNING: If infinite is False && balancing is used,
            some tiles will be skipped. Defaults to True.
        augment (str, optional): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
        standardize (bool, optional): Standardize images to (0,1). Defaults to True.
        normalizer (:class:`slideflow.slide.StainNormalizer`, optional): Normalizer to use on images. Defaults to None.
        num_shards (int, optional): Shard the tfrecord datasets, used for multiprocessing datasets. Defaults to None.
        shard_idx (int, optional): Index of the tfrecord shard to use. Defaults to None.
        num_parallel_reads (int, optional): Number of parallel reads for each TFRecordDataset. Defaults to 4.
    """

    if not len(tfrecords):
        raise ValueError("Interleaving failed: no tfrecords found.")
    log.debug(f'Interleaving {len(tfrecords)} tfrecords: infinite={infinite}')
    if num_shards:
        log.debug(f'num_shards={num_shards}, shard_idx={shard_idx}')

    if isinstance(labels, dict):
        label_parser = parser_from_labels(labels)
    elif callable(labels) or labels is None:
        label_parser = labels
    else:
        raise ValueError(f"Unrecognized type for labels: {type(labels)} (must be dict or function)")

    with tf.device('cpu'):
        # -------- Get the base TFRecord parser, based on the first tfrecord ------
        features_to_return = ('image_raw', 'slide') if not incl_loc else ('image_raw', 'slide', 'loc_x', 'loc_y')
        base_parser = None
        for i in range(len(tfrecords)):
            if base_parser is not None: continue
            if i > 0:
                log.debug(f"Unable to get parser from tfrecord, will try another (n={i})...")
            base_parser = get_tfrecord_parser(tfrecords[i],
                                             features_to_return,
                                             standardize=standardize,
                                             img_size=img_size,
                                             normalizer=normalizer,
                                             augment=augment)

        datasets = []
        weights = [] if prob_weights else None
        for tfr in tqdm(tfrecords, desc='Interleaving...', leave=False):
            tf_dts = tf.data.TFRecordDataset(tfr, num_parallel_reads=num_parallel_reads)
            if num_shards:
                tf_dts = tf_dts.shard(num_shards, index=shard_idx)
            if clip:
                tf_dts = tf_dts.take(clip[tfr] // (num_shards if num_shards else 1))
            if infinite:
                tf_dts = tf_dts.repeat()
            datasets += [tf_dts]
            if prob_weights:
                weights += [prob_weights[tfr]]

        #  -------  Interleave and batch datasets ---------------------------------
        sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
        dataset = _get_parsed_datasets(sampled_dataset,
                                       base_parser=base_parser,
                                       label_parser=label_parser,
                                       include_slidenames=incl_slidenames,
                                       include_loc=incl_loc)
        if batch_size:
            dataset = dataset.batch(batch_size, drop_remainder=False)
        #dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

def _get_parsed_datasets(tfrecord_dataset, base_parser, label_parser=None, include_slidenames=False, include_loc=False):

    def final_parser(record):

        if include_loc:
            image, slide, loc_x, loc_y = base_parser(record)
        else:
            image, slide = base_parser(record)
        image, label = label_parser(image, slide) if label_parser else (image, None)

        to_return = [image, label]
        if include_slidenames: to_return += [slide]
        if include_loc: to_return += [loc_x, loc_y]
        return tuple(to_return)

    return tfrecord_dataset.map(final_parser, num_parallel_calls=32)

def tfrecord_example(slide, image_raw, loc_x=0, loc_y=0):
    '''Returns a Tensorflow Data example for TFRecord storage.'''
    feature = {
        'slide':     _bytes_feature(slide),
        'image_raw':_bytes_feature(image_raw),
        'loc_x': _int64_feature(loc_x),
        'loc_y': _int64_feature(loc_y)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def serialized_record(slide, image_raw, loc_x=0, loc_y=0):
    '''Returns a serialized example for TFRecord storage, ready to be written
    by a TFRecordWriter.'''

    return tfrecord_example(slide, image_raw, loc_x, loc_y).SerializeToString()

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
        parser = get_tfrecord_parser(tfrecords.values()[0], decode_images=False, to_numpy=True)
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
            writer.write(read_and_return_record(record, parser, None))

def join_tfrecord(input_folder, output_file, assign_slide=None):
    '''Randomly samples from tfrecords in the input folder with shuffling,
    and combines into a single tfrecord file.'''
    writer = tf.io.TFRecordWriter(output_file)
    tfrecord_files = glob(join(input_folder, "*.tfrecords"))
    datasets = []
    if assign_slide: assign_slide = assign_slide.encode('utf-8')
    feature_description, img_type = detect_tfrecord_format(tfrecord_files[0])
    parser = get_tfrecord_parser(tfrecord_files[0], decode_images=False, to_numpy=True)
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
        writer.write(read_and_return_record(record, parser, assign_slide))

def split_tfrecord(tfrecord_file, output_folder):
    '''Splits records from a single tfrecord file into individual tfrecord files by slide.'''
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    feature_description, _ = detect_tfrecord_format(tfrecord_file)
    parser = get_tfrecord_parser(tfrecord_file, ['slide'], to_numpy=True)
    full_parser = get_tfrecord_parser(tfrecord_file, decode_images=False, to_numpy=True)
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
        writer.write(read_and_return_record(record, full_parser))

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
            update_tfrecord(tfr, assign_slide, image_raw)
        return len(tfrecord_files)

def update_tfrecord(tfrecord_file, assign_slide=None):
    '''Updates a single tfrecord from an old format to a new format.'''

    shutil.move(tfrecord_file, tfrecord_file+".old")
    dataset = tf.data.TFRecordDataset(tfrecord_file+".old")
    writer = tf.io.TFRecordWriter(tfrecord_file)
    parser = get_tfrecord_parser(tfrecord_file+'.old', decode_images=False, to_numpy=True)
    for record in dataset:
        slidename = bytes(assign_slide, 'utf-8') if assign_slide else None
        writer.write(read_and_return_record(record, parser, assign_slide=slidename))
    writer.close()
    os.remove(tfrecord_file+'.old')

def transform_tfrecord(origin, target, assign_slide=None, hue_shift=None, resize=None, silent=False):
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