import os
import shutil
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from functools import partial
from glob import glob
from os import listdir
from os.path import exists, isfile, join
from random import randint, shuffle
from rich.progress import track, Progress
from rich import print as richprint
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Tuple, Union)

import slideflow as sf
from slideflow import errors
from slideflow.io import gaussian
from slideflow.io.io_utils import detect_tfrecord_format
from slideflow.util import Labels
from slideflow.util import log

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer
    from tensorflow.core.example.feature_pb2 import Example, Feature


FEATURE_DESCRIPTION = {
    'slide': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'loc_x': tf.io.FixedLenFeature([], tf.int64),
    'loc_y': tf.io.FixedLenFeature([], tf.int64)
}


def _bytes_feature(value: bytes) -> "Feature":
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> "Feature":
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _apply_otsu(wsi):
    wsi.qc('otsu')
    return wsi


def read_and_return_record(
    record: bytes,
    parser: Callable,
    assign_slide: Optional[bytes] = None
) -> "Example":
    """Process raw TFRecord bytes into a format that can be written with
    ``tf.io.TFRecordWriter``.

    Args:
        record (bytes): Raw TFRecord bytes (unparsed)
        parser (Callable): TFRecord parser, as returned by
            :func:`sf.io.get_tfrecord_parser()`
        assign_slide (str, optional): Slide name to override the record with.
            Defaults to None.

    Returns:
        Dictionary mapping record key to a tuple containing (bytes, dtype).
    """
    features = parser(record)
    if assign_slide:
        features['slide'] = assign_slide
    tf_example = tfrecord_example(**features)
    return tf_example.SerializeToString()


def _print_record(filename: str) -> None:
    dataset = tf.data.TFRecordDataset(filename)
    parser = get_tfrecord_parser(
        filename,
        ('slide', 'loc_x', 'loc_y'),
        to_numpy=True,
        error_if_invalid=False
    )
    if parser is None:
        raise errors.TFRecordsError(f"Unable to read TFRecord {filename}")
    for i, record in enumerate(dataset):
        slide, loc_x, loc_y = parser(record)
        line = f"[magenta]{filename}[/]: Record {i}: Slide: "
        line += f"[green]{str(slide)}[/] Loc: {(loc_x, loc_y)}"
        richprint(line)


@tf.function
def preprocess_uint8(
    img: tf.Tensor,
    normalizer: Optional["StainNormalizer"] = None,
    standardize: bool = True,
    resize_px: Optional[int] = None,
    resize_method: str = 'lanczos3',
    resize_aa: bool = True,
    as_dict: bool = True
) -> Dict[str, tf.Tensor]:
    """Process batch of tensorflow images, resizing, normalizing,
    and standardizing.

    Args:
        img (tf.Tensor): Batch of tensorflow images (uint8).
        normalizer (sf.norm.StainNormalizer, optional): Normalizer.
            Defaults to None.
        standardize (bool, optional): Standardize images. Defaults to True.
        resize_px (Optional[int], optional): Resize images. Defaults to None.
        resize_method (str, optional): Resize method. Defaults to 'lanczos3'.
        resize_aa (bool, optional): Apply antialiasing during resizing.
            Defaults to True.

    Returns:
        Dict[str, tf.Tensor]: Processed image.
    """
    if resize_px is not None:
        img = tf.image.resize(
            img,
            (resize_px, resize_px),
            method=resize_method,
            antialias=resize_aa
        )
        img = tf.cast(img, tf.uint8)
    if normalizer is not None:
        img = normalizer.tf_to_tf(img)  # type: ignore
    if standardize:
        img = tf.image.per_image_standardization(img)
    if as_dict:
        return {'tile_image': img}
    else:
        return img


@tf.function
def process_image(
    record: Union[tf.Tensor, Dict[str, tf.Tensor]],
    *args: Any,
    standardize: bool = False,
    augment: Union[bool, str] = False,
    size: Optional[int] = None
) -> Tuple[Union[Dict, tf.Tensor], ...]:
    """Applies augmentations and/or standardization to an image Tensor."""

    if isinstance(record, dict):
        image = record['tile_image']
    else:
        image = record
    if size is not None:
        image.set_shape([size, size, 3])
    if augment is True or (isinstance(augment, str) and 'j' in augment):
        # Augment with random compession
        image = tf.cond(tf.random.uniform(
                            shape=[],  # pylint: disable=unexpected-keyword-arg
                            minval=0,
                            maxval=1,
                            dtype=tf.float32
                        ) < 0.5,
                        true_fn=lambda: tf.image.adjust_jpeg_quality(
                            image, tf.random.uniform(
                                shape=[],  # pylint: disable=unexpected-keyword-arg
                                minval=50,
                                maxval=100,
                                dtype=tf.int32
                            )
                        ),
                        false_fn=lambda: image)
    if augment is True or (isinstance(augment, str) and 'r' in augment):
        # Rotate randomly 0, 90, 180, 270 degrees
        image = tf.image.rot90(
            image,
            tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        )  # pylint: disable=unexpected-keyword-arg
        # Random flip and rotation
    if augment is True or (isinstance(augment, str) and 'x' in augment):
        image = tf.image.random_flip_left_right(image)
    if augment is True or (isinstance(augment, str) and 'y' in augment):
        image = tf.image.random_flip_up_down(image)
    if augment is True or (isinstance(augment, str) and 'b' in augment):
        # Augment with random gaussian blur (p=0.1)
        uniform_kwargs = {
            'shape': [],
            'minval': 0,
            'maxval': 1,
            'dtype': tf.float32
        }
        image = tf.cond(
            tf.random.uniform(**uniform_kwargs) < 0.1,
            true_fn=lambda: tf.cond(
                tf.random.uniform(**uniform_kwargs) < 0.5,
                true_fn=lambda: tf.cond(
                    tf.random.uniform(**uniform_kwargs) < 0.5,
                    true_fn=lambda: tf.cond(
                        tf.random.uniform(**uniform_kwargs) < 0.5,
                        true_fn=lambda: gaussian.auto_gaussian(image, sigma=2.0),
                        false_fn=lambda: gaussian.auto_gaussian(image, sigma=1.5),
                    ),
                    false_fn=lambda: gaussian.auto_gaussian(image, sigma=1.0),
                ),
                false_fn=lambda: gaussian.auto_gaussian(image, sigma=0.5),
            ),
            false_fn=lambda: image
        )
    if augment is True or (isinstance(augment, str) and 'i' in augment):
        ...
    if standardize:
        image = tf.image.per_image_standardization(image)

    if isinstance(record, dict):
        to_return = {k: v for k, v in record.items() if k != 'tile_image'}
        to_return['tile_image'] = image
        return tuple([to_return] + list(args))
    else:
        return tuple([image] + list(args))


@tf.function
def decode_image(
    img_string: bytes,
    img_type: str,
    crop_left: Optional[int] = None,
    crop_width: Optional[int] = None,
    resize_target: Optional[int] = None,
    resize_method: str = 'lanczos3',
    resize_aa: bool = True,
    size: Optional[int] = None
) -> tf.Tensor:
    """Decodes an image.

    Args:
        img_string (bytes): Image bytes (JPG/PNG).
        img_type (str): Type of image data; 'jpg', 'jpeg', or 'png'.
        crop_left (int, optional): Crop image starting at this top-left
            coordinate. Defaults to None.
        crop_width (int, optional): Crop image to this width.
            Defaults to None.
        resize_target (int, optional): Resize image, post-crop, to this target
            size in pixels. Defaults to None.
        resize_method (str, optional): Resizing method, if applicable.
            Defaults to 'lanczos3'.
        resize_aa (bool, optional): If resizing, use antialiasing.
            Defaults to True.
        size (int, optional): Set the image size/width (pixels).
            Defaults to None.

    Returns:
        tf.Tensor: Processed image (uint8).
    """
    tf_decoders = {
        'png': tf.image.decode_png,
        'jpeg': tf.image.decode_jpeg,
        'jpg': tf.image.decode_jpeg
    }
    decoder = tf_decoders[img_type.lower()]
    image = decoder(img_string, channels=3)
    if crop_left is not None:
        image = tf.image.crop_to_bounding_box(
            image, crop_left, crop_left, crop_width, crop_width
        )
    if resize_target is not None:
        image = tf.image.resize(image, (resize_target, resize_target), method=resize_method, antialias=resize_aa)
        image.set_shape([resize_target, resize_target, 3])
    elif size:
        image.set_shape([size, size, 3])
    return image


def get_tfrecord_parser(
    tfrecord_path: str,
    features_to_return: Optional[Iterable[str]] = None,
    to_numpy: bool = False,
    decode_images: bool = True,
    img_size: Optional[int] = None,
    error_if_invalid: bool = True,
    **decode_kwargs: Any
) -> Optional[Callable]:

    """Returns a tfrecord parsing function based on the specified parameters.

    Args:
        tfrecord_path (str): Path to tfrecord to parse.
        features_to_return (list or dict, optional): Designates format for how
            features should be returned from parser. If a list of feature names
            is provided, the parsing function will return tfrecord features as
            a list in the order provided. If a dictionary of labels (keys)
            mapping to feature names (values) is provided, features will be
            returned from the parser as a dictionary matching the same format.
            If None, will return all features as a list.
        to_numpy (bool, optional): Convert records from tensors->numpy arrays.
            Defaults to False.
        decode_images (bool, optional): Decode image strings into arrays.
            Defaults to True.
        standardize (bool, optional): Standardize images into the range (0,1).
            Defaults to False.
        img_size (int): Width of images in pixels. Will call tf.set_shape(...)
            if provided. Defaults to False.
        normalizer (:class:`slideflow.norm.StainNormalizer`): Stain normalizer
            to use on images. Defaults to None.
        augment (str): Image augmentations to perform. String containing
            characters designating augmentations. 'x' indicates random
            x-flipping, 'y' y-flipping, 'r' rotating, 'j' JPEG
            compression/decompression at random quality levels, and 'b'
            random gaussian blur. Passing either 'xyrjb' or True will use all
            augmentations.
        error_if_invalid (bool, optional): Raise an error if a tfrecord cannot
            be read. Defaults to True.
    """
    features, img_type = detect_tfrecord_format(tfrecord_path)
    if features is None:
        log.debug(f"Unable to read tfrecord at {tfrecord_path} - is it empty?")
        return None
    if features_to_return is None:
        features_to_return = {k: k for k in features}
    feature_description = {
        k: v for k, v in FEATURE_DESCRIPTION.items()
        if k in features
    }

    def parser(record):
        features = tf.io.parse_single_example(record, feature_description)

        def process_feature(f):
            if f not in features and f in ('loc_x', 'loc_y'):
                return None
            elif f not in features and error_if_invalid:
                raise errors.TFRecordsError(f"Unknown TFRecord feature {f}")
            elif f not in features:
                return None
            elif f == 'image_raw' and decode_images:
                return decode_image(
                    features['image_raw'],
                    img_type,
                    size=img_size,
                    **decode_kwargs
                )
            elif to_numpy:
                return features[f].numpy()
            else:
                return features[f]

        if type(features_to_return) == dict:
            return {
                label: process_feature(f)
                for label, f in features_to_return.items()
            }
        else:
            return [process_feature(f) for f in features_to_return]

    return parser


def parser_from_labels(labels: Labels) -> Callable:
    '''Returns a label parsing function used for parsing slides into single
    or multi-outcome labels.
    '''
    outcome_labels = np.array(list(labels.values()))
    slides = list(labels.keys())
    if len(outcome_labels.shape) == 1:
        outcome_labels = np.expand_dims(outcome_labels, axis=1)
    with tf.device('/cpu'):
        annotations_tables = []
        for oi in range(outcome_labels.shape[1]):
            annotations_tables += [tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    slides,
                    outcome_labels[:, oi]
                ), -1
            )]

    def label_parser(image, slide):
        if outcome_labels.shape[1] > 1:
            label = [
                annotations_tables[oi].lookup(slide)
                for oi in range(outcome_labels.shape[1])
            ]
        else:
            label = annotations_tables[0].lookup(slide)
        return image, label

    return label_parser


def interleave(
    paths: List[str],
    *,
    img_size: int,
    batch_size: Optional[int],
    prob_weights: Optional[Dict[str, float]] = None,
    clip: Optional[Dict[str, int]] = None,
    labels: Optional[Labels] = None,
    incl_slidenames: bool = False,
    incl_loc: Optional[str] = None,
    infinite: bool = True,
    augment: bool = False,
    standardize: bool = True,
    normalizer: Optional["StainNormalizer"] = None,
    num_shards: Optional[int] = None,
    shard_idx: Optional[int] = None,
    num_parallel_reads: int = 4,
    deterministic: bool = False,
    drop_last: bool = False,
    from_wsi: bool = False,
    tile_um: Optional[int] = None,
    rois: Optional[List[str]] = None,
    roi_method: str = 'auto',
    pool: Optional["mp.pool.Pool"] = None,
    tfrecord_parser: Optional[Callable] = None,
    **decode_kwargs: Any
) -> Iterable:

    """Generates an interleaved dataset from a collection of tfrecord files,
    sampling from tfrecord files randomly according to balancing if provided.
    Requires manifest for balancing. Assumes TFRecord files are named by slide.

    Args:
        paths (list(str)): List of paths to TFRecord files or whole-slide
            images.
        img_size (int): Image width in pixels.
        batch_size (int): Batch size.
        prob_weights (dict, optional): Dict mapping tfrecords to probability of
            including in batch. Defaults to None.
        clip (dict, optional): Dict mapping tfrecords to number of tiles to
            take per tfrecord. Defaults to None.
        labels (dict or str, optional): Dict or function. If dict, must map
            slide names to outcome labels. If function, function must accept an
            image (tensor) and slide name (str), and return a dict
            {'image_raw': image (tensor)} and label (int or float). If not
            provided,  all labels will be None.
        incl_slidenames (bool, optional): Include slidenames as third returned
            variable. Defaults to False.
        incl_loc (str, optional): 'coord', 'grid', or None. Return (x,y)
                origin coordinates ('coord') for each tile along with tile
                images, or the (x,y) grid coordinates for each tile.
                Defaults to 'coord'.
        infinite (bool, optional): Create an finite dataset. WARNING: If
            infinite is False && balancing is used, some tiles will be skipped.
            Defaults to True.
        augment (str): Image augmentations to perform. String containing
            characters designating augmentations. 'x' indicates random
            x-flipping, 'y' y-flipping, 'r' rotating, 'j' JPEG
            compression/decompression at random quality levels, and 'b'
            random gaussian blur. Passing either 'xyrjb' or True will use all
            augmentations.
        standardize (bool, optional): Standardize images to (0,1).
            Defaults to True.
        normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
            Normalizer to use on images. Defaults to None.
        num_shards (int, optional): Shard the tfrecord datasets, used for
            multiprocessing datasets. Defaults to None.
        shard_idx (int, optional): Index of the tfrecord shard to use.
            Defaults to None.
        num_parallel_reads (int, optional): Number of parallel reads for each
            TFRecordDataset. Defaults to 4.
        deterministic (bool, optional): When num_parallel_calls is specified,
            if this boolean is specified, it controls the order in which the
            transformation produces elements. If set to False, the
            transformation is allowed to yield elements out of order to trade
            determinism for performance. Defaults to False.
        drop_last (bool, optional): Drop the last non-full batch.
            Defaults to False.
    """
    if not len(paths):
        raise errors.TFRecordsNotFoundError
    _path_type = "slides" if from_wsi else "tfrecords"
    log.debug(
        f'Interleaving {len(paths)} {_path_type}: infinite={infinite}, '
        f'num_parallel_reads={num_parallel_reads}')
    if from_wsi and not tile_um:
        raise ValueError("`tile_um` required for interleave() "
                         "if `from_wsi=True`")

    if num_shards:
        log.debug(f'num_shards={num_shards}, shard_idx={shard_idx}')
    if isinstance(labels, dict):
        label_parser = parser_from_labels(labels)
    elif callable(labels) or labels is None:
        label_parser = labels  # type: ignore
    else:
        raise ValueError(
            f"Unrecognized type for labels: {type(labels)} (must be dict"
            " or function)")

    datasets = []
    weights = [] if prob_weights else None  # type: Optional[List]

    pb = Progress(transient=True)
    if from_wsi:
        read_task = pb.add_task('Reading slides...', total=len(paths), visible=False)
        otsu_task = pb.add_task("Otsu thresholding...", total=len(paths), visible=False)
    interleave_task = pb.add_task('Interleaving...', total=len(paths))
    pb.start()
    with tf.device('cpu'), sf.util.cleanup_progress(pb):
        features_to_return = ['image_raw', 'slide']
        if incl_loc:
            features_to_return += ['loc_x', 'loc_y']

        if from_wsi:
            assert tile_um is not None
            pb.update(read_task, visible=True)
            pb.update(otsu_task, visible=True)

            def base_parser(record):
                return tuple([record[f] for f in features_to_return])

            # Load slides and apply Otsu's thresholding
            if pool is None and sf.slide_backend() == 'cucim':
                pool = mp.Pool(
                    8 if os.cpu_count is None else os.cpu_count(),
                    initializer=sf.util.set_ignore_sigint
                )
            elif pool is None:
                pool = mp.dummy.Pool(16 if os.cpu_count is None else os.cpu_count())
            wsi_list = []
            to_remove = []
            otsu_list = []
            for path in paths:
                try:
                    wsi = sf.WSI(
                        path,
                        img_size,
                        tile_um,
                        rois=rois,
                        roi_method=roi_method,
                        verbose=False
                    )
                    wsi_list += [wsi]
                    pb.advance(read_task)
                except errors.SlideLoadError as e:
                    log.error(f"Error reading slide {path}: {e}")
                    to_remove += [path]
            for path in to_remove:
                paths.remove(path)
            for task in (read_task, otsu_task, interleave_task):
                pb.update(task, total=len(paths))
            for wsi in pool.imap(_apply_otsu, wsi_list):
                otsu_list += [wsi]
                pb.advance(otsu_task)
            est_num_tiles = sum([wsi.estimated_num_tiles for wsi in otsu_list])
        elif tfrecord_parser is None:
            base_parser = None  # type: ignore
            for i in range(len(paths)):
                if base_parser is not None:
                    continue
                if i > 0:
                    log.debug(f"Failed to get parser, trying again (n={i})...")
                base_parser = get_tfrecord_parser(
                    paths[i],
                    features_to_return,
                    img_size=img_size,
                    **decode_kwargs)
        else:
            base_parser = tfrecord_parser

        for t, tfr in enumerate(paths):
            if from_wsi:
                tf_dts = otsu_list[t].tensorflow(
                    pool=pool,
                    lazy_iter=True,
                    incl_slidenames=True,
                    grayspace_fraction=1,
                    incl_loc=incl_loc,
                )
                tfr = sf.util.path_to_name(tfr)
            else:
                tf_dts = tf.data.TFRecordDataset(
                    tfr,
                    num_parallel_reads=num_parallel_reads
                )
            if num_shards:
                tf_dts = tf_dts.shard(num_shards, index=shard_idx)
            if clip:
                tf_dts = tf_dts.take(
                    clip[tfr] // (num_shards if num_shards else 1)
                )
            if infinite:
                tf_dts = tf_dts.repeat()
            datasets += [tf_dts]
            if prob_weights:
                weights += [prob_weights[tfr]]  # type: ignore
            pb.advance(interleave_task)

        # ------- Interleave and parse datasets -------------------------------
        sampled_dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            weights=weights
        )
        dataset = _get_parsed_datasets(
            sampled_dataset,
            base_parser=base_parser,  # type: ignore
            label_parser=label_parser,
            include_slidenames=incl_slidenames,
            include_loc=incl_loc,
            deterministic=deterministic
        )
        # ------- Apply normalization -----------------------------------------
        if normalizer:
            if normalizer.vectorized:
                log.debug("Using vectorized normalization")
                norm_batch_size = 32 if not batch_size else batch_size
                dataset = dataset.batch(norm_batch_size, drop_remainder=drop_last)
            else:
                log.debug("Using per-image normalization")
            dataset = dataset.map(
                partial(normalizer.tf_to_tf, augment=(isinstance(augment, str)
                                                      and 'n' in augment)),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=deterministic,
            )
            if normalizer.vectorized:
                dataset = dataset.unbatch()
            if normalizer.method == 'macenko':
                # Drop the images that causes an error, e.g. if eigen
                # decomposition is unsuccessful.
                dataset = dataset.apply(tf.data.experimental.ignore_errors())

        # ------- Standardize and augment images ------------------------------
        dataset = dataset.map(
            partial(
                process_image,
                standardize=standardize,
                augment=augment,
                size=img_size
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic
        )
        # ------- Batch and prefetch ------------------------------------------
        if batch_size:
            dataset = dataset.batch(batch_size, drop_remainder=drop_last)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if from_wsi:
            dataset.est_num_tiles = est_num_tiles
        return dataset


def _get_parsed_datasets(
    tfrecord_dataset: tf.data.Dataset,
    base_parser: Callable,
    label_parser: Optional[Callable] = None,
    include_slidenames: bool = False,
    include_loc: Optional[str] = None,
    deterministic: bool = False
) -> tf.data.Dataset:
    """Return a parsed dataset.

    Args:
        tfrecord_dataset (tf.data.Dataset): Dataset to be parsed; should be
            a raw TFRecord reading dataset, yielding bytes.
        base_parser (Callable): Base TFRecord parser which parses bytes into
            features.
        label_parser (Optional[Callable], optional): Function to parse input
            (image, slide) into (image, label). Defaults to None.
        include_slidenames (bool, optional): Yield slide names as a third
            returned value. Defaults to False.
        include_loc (Optional[str], optional): Yield location X and Y coords
            as two additional values. If include_slidenames is true, these will
            follow slide names. Defaults to None.
        deterministic (bool, optional): Read from TFRecords in order, at the
            expense of performance. Defaults to False.

    Returns:
        tf.data.Dataset: Parsed dataset.
    """

    def final_parser(record):
        if include_loc:
            image, slide, loc_x, loc_y = base_parser(record)
        else:
            image, slide = base_parser(record)
        image, label = label_parser(image, slide) if label_parser else (image, None)

        to_return = [image, label]
        if include_slidenames:
            to_return += [slide]
        if include_loc:
            to_return += [loc_x, loc_y]
        return tuple(to_return)

    return tfrecord_dataset.map(
        final_parser,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic
    )


def tfrecord_example(
    slide: bytes,
    image_raw: bytes,
    loc_x: Optional[int] = 0,
    loc_y: Optional[int] = 0
) -> "Example":
    '''Returns a Tensorflow Data example for TFRecord storage.'''
    feature = {
        'slide': _bytes_feature(slide),
        'image_raw': _bytes_feature(image_raw),
    }
    if loc_x is not None:
        feature.update({'loc_x': _int64_feature(loc_x)})
    if loc_y is not None:
        feature.update({'loc_y': _int64_feature(loc_y)})
    return tf.train.Example(features=tf.train.Features(feature=feature))


def serialized_record(
    slide: bytes,
    image_raw: bytes,
    loc_x: int = 0,
    loc_y: int = 0
) -> bytes:
    '''Returns a serialized example for TFRecord storage, ready to be written
    by a TFRecordWriter.'''
    return tfrecord_example(slide, image_raw, loc_x, loc_y).SerializeToString()


def multi_image_example(slide: bytes, image_dict: Dict) -> "Example":
    '''Returns a Tensorflow Data example for storage with multiple images.'''
    feature = {
        'slide': _bytes_feature(slide)
    }
    for image_label in image_dict:
        feature.update({
            image_label: _bytes_feature(image_dict[image_label])
        })
    return tf.train.Example(features=tf.train.Features(feature=feature))


def join_tfrecord(
    input_folder: str,
    output_file: str,
    assign_slide: str = None
) -> None:
    '''Randomly samples from tfrecords in the input folder with shuffling,
    and combines into a single tfrecord file.'''
    writer = tf.io.TFRecordWriter(output_file)
    tfrecord_files = glob(join(input_folder, "*.tfrecords"))
    datasets = []
    if assign_slide:
        slide = assign_slide.encode('utf-8')
    features, img_type = detect_tfrecord_format(tfrecord_files[0])
    parser = get_tfrecord_parser(
        tfrecord_files[0],
        decode_images=False,
        to_numpy=True
    )
    for tfrecord in tfrecord_files:
        n_feat, n_img_type = detect_tfrecord_format(tfrecord)
        if n_feat != features or n_img_type != img_type:
            raise errors.TFRecordsError(
                "Mismatching tfrecord format found, unable to merge"
            )
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
        writer.write(
            read_and_return_record(record, parser, slide)  # type: ignore
        )


def split_tfrecord(tfrecord_file: str, output_folder: str) -> None:
    '''Splits records from a single tfrecord file into individual tfrecord
    files by slide.
    '''
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    parser = get_tfrecord_parser(tfrecord_file, ['slide'], to_numpy=True)
    full_parser = get_tfrecord_parser(
        tfrecord_file,
        decode_images=False,
        to_numpy=True
    )
    writers = {}  # type: ignore
    for record in dataset:
        slide = parser(record)  # type: ignore
        shortname = sf.util._shortname(slide.decode('utf-8'))
        if shortname not in writers.keys():
            tfrecord_path = join(output_folder, f"{shortname}.tfrecords")
            writer = tf.io.TFRecordWriter(tfrecord_path)
            writers.update({shortname: writer})
        else:
            writer = writers[shortname]
        writer.write(
            read_and_return_record(record, full_parser)  # type: ignore
        )
    for slide in writers.keys():
        writers[slide].close()


def print_tfrecord(target: str) -> None:
    '''Prints the slide names (and locations, if present) for records
    in the given tfrecord file.
    '''
    if isfile(target):
        _print_record(target)
    else:
        tfrecord_files = glob(join(target, "*.tfrecords"))
        for tfr in tfrecord_files:
            _print_record(tfr)


def checkpoint_to_tf_model(models_dir: str, model_name: str) -> None:
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
        log.debug("KeyError encountered in checkpoint_to_tf_model")
        pass


def transform_tfrecord(
    origin: str,
    target: str,
    assign_slide: Optional[str] = None,
    hue_shift: Optional[float] = None,
    resize: Optional[float] = None,
) -> None:
    '''Transforms images in a single tfrecord. Can perform hue shifting,
    resizing, or re-assigning slide label.
    '''
    log.info(f"Transforming tiles in tfrecord [green]{origin}")
    log.info(f"Saving to new tfrecord at [green]{target}")
    if assign_slide:
        log.info(f"Assigning slide name [bold]{assign_slide}")
    if hue_shift:
        log.info(f"Shifting hue by [bold]{hue_shift}")
    if resize:
        log.info(f"Resizing records to ({resize}, {resize})")
    dataset = tf.data.TFRecordDataset(origin)
    writer = tf.io.TFRecordWriter(target)
    parser = get_tfrecord_parser(
        origin,
        ('slide', 'image_raw', 'loc_x', 'loc_y'),
        decode_images=(hue_shift is not None or resize is not None),
        error_if_invalid=False,
        to_numpy=True
    )

    def process_image(image_string):
        if hue_shift:
            decoded_image = tf.image.decode_png(image_string, channels=3)
            adjusted_image = tf.image.adjust_hue(decoded_image, hue_shift)
            encoded_image = tf.io.encode_jpeg(adjusted_image, quality=80)
            return encoded_image.numpy()
        elif resize:
            decoded_image = tf.image.decode_png(image_string, channels=3)
            resized_image = tf.image.resize(
                decoded_image,
                (resize, resize),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            encoded_image = tf.io.encode_jpeg(resized_image, quality=80)
            return encoded_image.numpy()
        else:
            return image_string

    for record in dataset:
        slide, image_raw, loc_x, loc_y = parser(record)  # type: ignore
        if assign_slide and isinstance(assign_slide, str):
            slidename = bytes(assign_slide, 'utf-8')
        elif assign_slide:
            slidename = bytes(assign_slide(slide), 'utf-8')
        else:
            slidename = slide
        image_processed_data = process_image(image_raw)
        tf_example = tfrecord_example(
            slidename,
            image_processed_data,
            loc_x,
            loc_y
        )
        writer.write(tf_example.SerializeToString())
    writer.close()


def shuffle_tfrecord(target: str) -> None:
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


def shuffle_tfrecords_by_dir(directory: str) -> None:
    '''For each TFRecord in a directory, shuffles records in the TFRecord,
    saving the original to a .old file.
    '''
    records = [tfr for tfr in listdir(directory) if tfr[-10:] == ".tfrecords"]
    for record in records:
        log.info(f'Working on {record}')
        shuffle_tfrecord(join(directory, record))
