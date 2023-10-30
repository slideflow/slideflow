"""Data utilities for Torch datasets."""

import pandas as pd
import numpy as np

from slideflow import errors
from slideflow.util import tfrecord2idx, to_onehot
from slideflow.io.io_utils import detect_tfrecord_format
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable,
                    Optional, Tuple, Union)

from .augment import compose_augmentations
from .img_utils import decode_image

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -------------------------------------------------------------------------


FEATURE_DESCRIPTION = {
    'image_raw': 'byte',
    'slide': 'byte',
    'loc_x': 'int',
    'loc_y': 'int'
}

# -------------------------------------------------------------------------

def process_labels(
    labels: Optional[Dict[str, Any]] = None,
    onehot: bool = False
) -> Tuple[Optional[Union[Dict[str, Any], pd.DataFrame]],
           Optional[np.ndarray],
           Optional[np.ndarray],
           int]:
    """Analyze labels to determine unique labels, label probabilities, and
    number of outcomes.

    Args:
        labels (dict): Dict mapping slide names to labels.
        onehot (bool, optional): Onehot encode outcomes. Defaults to False.

    Returns:
        labels (dict): Dict mapping slide names to labels.
        unique_labels (np.ndarray): Unique labels.
        label_prob (np.ndarray): Label probabilities.
        num_outcomes (int): Number of outcomes.

    """
    # Weakly supervised labels from slides.
    if labels is not None and not isinstance(labels, (str, pd.DataFrame)):
        if onehot:
            _all_labels_raw = np.array(list(labels.values()))
            _unique_raw = np.unique(_all_labels_raw)
            max_label = np.max(_unique_raw)
            labels = {
                k: to_onehot(v, max_label+1)  # type: ignore
                for k, v in labels.items()
            }
            num_outcomes = 1
        else:
            first_label = list(labels.values())[0]
            if not isinstance(first_label, list):
                num_outcomes = 1
            else:
                num_outcomes = len(first_label)

        _all_labels = np.array(list(labels.values()))
        unique_labels = np.unique(_all_labels, axis=0)
        _lbls = np.array([
            np.sum(_all_labels == i)
            for i in unique_labels
        ])
        label_prob = _lbls / len(_all_labels)

    # Strongly supervised tile labels from a dataframe.
    elif isinstance(labels, (pd.DataFrame, str)):
        if isinstance(labels, str):
            df = pd.read_parquet(labels)
        else:
            df = labels
        if 'label' not in df.columns:
            raise ValueError('Could not find column "label" in the '
                             f'tile labels dataframe at {labels}.')
        labels = df
        unique_labels = None
        label_prob = None
        num_outcomes = 1
    else:
        unique_labels = None
        label_prob = None  # type: ignore
        num_outcomes = 1
    return labels, unique_labels, label_prob, num_outcomes

# -------------------------------------------------------------------------

def load_index(tfr):
    if isinstance(tfr, bytes):
        tfr = tfr.decode('utf-8')
    try:
        index = tfrecord2idx.load_index(tfr)
    except OSError:
        raise errors.TFRecordsError(
            f"Could not find index path for TFRecord {tfr}"
        )
    return index


def read_and_return_record(
    record: bytes,
    parser: Callable,
    assign_slide: Optional[str] = None
) -> Dict:
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
    parsed = parser(record)
    if assign_slide:
        parsed['slide'] = assign_slide
    parsed['slide'] = parsed['slide'].encode('utf-8')
    return {k: (v, FEATURE_DESCRIPTION[k]) for k, v in parsed.items()}


def serialized_record(
    slide: bytes,
    image_raw: bytes,
    loc_x: int = 0,
    loc_y: int = 0
):
    """Returns a serialized example for TFRecord storage, ready to be written
    by a TFRecordWriter."""

    example = {
        'image_raw': (image_raw, FEATURE_DESCRIPTION['image_raw']),
        'slide': (slide, FEATURE_DESCRIPTION['slide']),
        'loc_x': (loc_x, FEATURE_DESCRIPTION['loc_x']),
        'loc_y': (loc_y, FEATURE_DESCRIPTION['loc_y']),
    }
    return example


def get_tfrecord_parser(
    tfrecord_path: str,
    features_to_return: Iterable[str] = None,
    decode_images: bool = True,
    standardize: bool = False,
    normalizer: Optional["StainNormalizer"] = None,
    augment: bool = False,
    **kwargs
) -> Callable:

    """Gets tfrecord parser using dareblopy reader. Torch implementation;
    different than sf.io.tensorflow

    Args:
        tfrecord_path (str): Path to tfrecord to parse.
        features_to_return (list or dict, optional): Designates format for how
            features should be returned from parser. If a list of feature names
            is provided, the parsing function will return tfrecord features as
            a list in the order provided. If a dictionary of labels (keys)
            mapping to feature names (values) is provided, features will be
            returned from the parser as a dictionary matching the same format.
            If None, will return all features as a list.
        decode_images (bool, optional): Decode raw image strings into image
            arrays. Defaults to True.
        standardize (bool, optional): Standardize images into the range (0,1).
            Defaults to False.
        normalizer (:class:`slideflow.norm.StainNormalizer`): Stain normalizer
            to use on images. Defaults to None.
        augment (str or bool): Image augmentations to perform. Augmentations include:

            * ``'x'``: Random horizontal flip
            * ``'y'``: Random vertical flip
            * ``'r'``: Random 90-degree rotation
            * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
            * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)

            Combine letters to define augmentations, such as ``'xyrjn'``.
            A value of True will use ``'xyrjb'``.
            Note: this function does not support stain augmentation.

    Returns:
        A tuple containing

            func: Parsing function

            dict: Detected feature description for the tfrecord
    """

    features, img_type = detect_tfrecord_format(tfrecord_path)
    if features is None or img_type is None:
        raise errors.TFRecordsError(f"Unable to read TFRecord {tfrecord_path}")
    if features_to_return is None:
        features_to_return = {k: k for k in features}
    elif not all(f in features for f in features_to_return):
        detected = ",".join(features)
        _ftrs = list(features_to_return.keys())  # type: ignore
        raise errors.TFRecordsError(
            f'Not all features {",".join(_ftrs)} '
            f'were found in the tfrecord {detected}'
        )

    # Build the transformations / augmentations.
    transform = compose_augmentations(
        augment=augment,
        standardize=standardize,
        normalizer=normalizer,
        whc=True
    )

    def parser(record):
        """Each item in args is an array with one item, as the dareblopy reader
        returns items in batches and we have set our batch_size = 1 for
        interleaving.
        """
        features = {}
        if ('slide' in features_to_return):
            slide = bytes(record['slide']).decode('utf-8')
            features['slide'] = slide
        if ('image_raw' in features_to_return):
            img = bytes(record['image_raw'])
            if decode_images:
                features['image_raw'] = decode_image(
                    img,
                    img_type=img_type,
                    transform=transform
                )
            else:
                features['image_raw'] = img
        if ('loc_x' in features_to_return):
            features['loc_x'] = record['loc_x'][0]
        if ('loc_y' in features_to_return):
            features['loc_y'] = record['loc_y'][0]
        if type(features_to_return) == dict:
            return {
                label: features[f]
                for label, f in features_to_return.items()
            }
        else:
            return [features[f] for f in features_to_return]
    return parser
