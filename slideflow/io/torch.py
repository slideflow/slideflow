import multiprocessing as mp
import os
import random
import threading
import numpy as np
import pandas as pd
import torchvision
import torch
import math
from torchvision import transforms
from os import listdir
from os.path import dirname, exists, isfile, join
from queue import Queue
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Tuple, Union)


import slideflow as sf
from slideflow import errors
from slideflow.io import convert_dtype
from slideflow.io.io_utils import detect_tfrecord_format
from slideflow.tfrecord.torch.dataset import MultiTFRecordDataset
from slideflow.tfrecord.iterator_utils import RandomSampler
from slideflow.util import Labels, log, to_onehot, tfrecord2idx

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer
    from torchvision.transforms import InterpolationMode

FEATURE_DESCRIPTION = {
    'image_raw': 'byte',
    'slide': 'byte',
    'loc_x': 'int',
    'loc_y': 'int'
}


class InterleaveIterator(torch.utils.data.IterableDataset):
    """Pytorch Iterable Dataset that interleaves tfrecords with the
    interleave() function below. Serves as a bridge between the python
    generator returned by interleave() and the pytorch DataLoader class.
    """

    def __init__(
        self,
        tfrecords: List[str],
        img_size: int,
        labels: Optional[Labels] = None,
        incl_slidenames: bool = False,
        incl_loc: bool = False,
        rank: int = 0,
        num_replicas: int = 1,
        augment: Union[str, bool] = False,
        standardize: bool = True,
        num_tiles: Optional[int] = None,
        infinite: bool = True,
        prob_weights: Optional[Dict[str, float]] = None,
        normalizer: Optional["StainNormalizer"] = None,
        clip: Optional[List[int]] = None,
        chunk_size: int = 1,
        use_labels: bool = True,
        model_type: str = 'categorical',
        onehot: bool = False,
        indices: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        max_size: int = 0,
        from_wsi: bool = False,
        tile_um: Optional[int] = None,
        rois: Optional[List[str]] = None,
        roi_method: str = 'auto',
        pool: Optional[Any] = None,
        transform: Optional[Any] = None,
        **interleave_kwargs
    ) -> None:
        """Pytorch IterableDataset that interleaves tfrecords with
        :func:`slideflow.io.torch.interleave`.

        Args:
            tfrecords (list(str)): Path to tfrecord files to interleave.
            img_size (int): Image width in pixels.
            labels (dict, optional): Dict mapping slide names to labels.
                Defaults to None.
            incl_slidenames (bool, optional): Include slide names when iterated
                (returns image, label, slide). Defaults to False.
            incl_loc (bool, optional): Include location info. Returns samples
                in the form (returns ..., loc_x, loc_y). Defaults to False.
            rank (int, optional): Which GPU replica this dataset is used for.
                Assists with synchronization across GPUs. Defaults to 0.
            num_replicas (int, optional): Total number of GPU replicas.
                Defaults to 1.
            augment (str of bool, optional): Image augmentations to perform.
                If string, 'x' performs horizontal flipping, 'y' performs
                vertical flipping, 'r' performs rotation, 'j' performs random
                JPEG compression (e.g. 'xyr', 'xyrj', 'xy'). If bool, True
                performs all and False performs None. Defaults to True.
            standardize (bool, optional): Standardize images to mean 0 and
                variance of 1. Defaults to True.
            num_tiles (int, optional): Dict mapping tfrecord names to number
                of total tiles. Defaults to None.
            infinite (bool, optional): Inifitely loop through dataset.
                Defaults to True.
            prob_weights (list(float), optional): Probability weights for
                interleaving tfrecords. Defaults to None.
            normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
                Normalizer. Defaults to None.
            clip (list(int), optional): Array of maximum tiles to take for each
                tfrecord. Defaults to None.
            chunk_size (int, optional): Chunk size for image decoding.
                Defaults to 1.
            use_labels (bool, optional): Enable use of labels (disabled for
                non-conditional GANs). Defaults to True.
            model_type (str, optional): Used to generate random labels
                (for StyleGAN2). Not required. Defaults to 'categorical'.
            onehot (bool, optional): Onehot encode outcomes. Defaults to False.
            indices (numpy.ndarray, optional): Indices in form of array,
                with np.loadtxt(index_path, dtype=np.int64) for each tfrecord.
                Defaults to None.
            max_size (bool, optional): Unused argument present for legacy
                compatibility; will be removed.
        """
        self.tfrecords = np.array(tfrecords).astype(np.string_)
        if prob_weights is not None:
            self.prob_weights = np.array(prob_weights)
        else:
            self.prob_weights = None  # type: ignore
        self.clip = clip
        self.indices = indices
        self.img_size = img_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.augment = augment
        self.standardize = standardize
        self.infinite = infinite
        self.max_size = max_size
        self.use_labels = use_labels
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.onehot = onehot
        self.incl_slidenames = incl_slidenames
        self.incl_loc = incl_loc
        self.num_tiles = num_tiles
        self.model_type = model_type
        self.device = device
        self.from_wsi = from_wsi
        self.tile_um = tile_um
        self.rois = rois
        self.roi_method = roi_method
        self.pool = pool
        self.transform = transform
        self.interleave_kwargs = interleave_kwargs

        # Values for random label generation, for GAN
        if labels is not None:
            if self.onehot:
                _all_labels_raw = np.array(list(labels.values()))
                _unique_raw = np.unique(_all_labels_raw)
                max_label = np.max(_unique_raw)
                labels = {
                    k: to_onehot(v, max_label+1)  # type: ignore
                    for k, v in labels.items()
                }
                self.num_outcomes = 1
            else:
                first_label = list(labels.values())[0]
                if not isinstance(first_label, list):
                    self.num_outcomes = 1
                else:
                    self.num_outcomes = len(first_label)

            _all_labels = np.array(list(labels.values()))
            self.unique_labels = np.unique(_all_labels, axis=0)
            _lbls = np.array([
                np.sum(_all_labels == i)
                for i in self.unique_labels
            ])
            self.label_prob = _lbls / len(_all_labels)
        else:
            self.unique_labels = None
            self.label_prob = None  # type: ignore
            self.num_outcomes = 1
        self.labels = labels

    @property
    def name(self) -> str:
        return 'slideflow-interleave-iterator'

    @property
    def resolution(self) -> int:
        return self.img_size

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (3, self.resolution, self.resolution)

    @property
    def num_channels(self) -> int:
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def label_shape(self) -> Union[int, Tuple[int, ...]]:
        """For use with StyleGAN2"""
        if self.use_labels and self.unique_labels is not None:
            return self.unique_labels[0].shape
        else:
            return 0

    @property
    def label_dim(self) -> int:
        if self.use_labels:
            assert len(self.label_shape) == 1  # type: ignore
            return self.label_shape[0]  # type: ignore
        else:
            return 0

    @property
    def has_labels(self) -> bool:
        return (self.use_labels
                and any(x != 0 for x in self.label_shape))  # type: ignore

    def __len__(self) -> Optional[int]:
        return self.num_tiles

    def _parser(
        self,
        image: torch.Tensor,
        slide: str,
        loc_x: Optional[int] = None,
        loc_y: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Parse a standardize PyTorch image (WHC) and slide/location
        information, to a CWH image formatted for model input."""
        if self.labels is not None:
            label = self.labels[slide]
        else:
            label = 0

        image = whc_to_cwh(image)
        to_return = [image]  # type: List[Any]

        # Support for multiple outcome labels
        if self.num_outcomes > 1:
            to_return += [{
                f'out-{i}': torch.tensor(l)
                for i, l in enumerate(label)  # type: ignore
            }]
        else:
            to_return += [torch.tensor(label)]

        if self.incl_slidenames:
            to_return += [slide]
        if self.incl_loc:
            to_return += [loc_x, loc_y]
        return to_return

    def __repr__(self) -> str:
        n_records = self.tfrecords.shape[0]
        msg = f"<InterleaveIterator object (num_records={n_records}, num_tiles"
        msg += f"={self.num_tiles}, infinite={self.infinite}, rank=("
        msg += f"{self.rank} of {self.num_replicas}), augment={self.augment}, "
        msg += f"standardize={self.standardize})>"
        return msg

    def __del__(self):
        self.close()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if not worker_info else worker_info.id
        num_workers = 1 if not worker_info else worker_info.num_workers

        queue_retriever = interleave(
            self.tfrecords,
            incl_loc=self.incl_loc,
            standardize=self.standardize,
            augment=self.augment,
            prob_weights=self.prob_weights,
            clip=self.clip,
            infinite=self.infinite,
            normalizer=self.normalizer,
            num_replicas=self.num_replicas * num_workers,
            rank=self.rank + worker_id,
            chunk_size=self.chunk_size,
            indices=self.indices,
            device=self.device,
            tile_px=self.img_size,
            from_wsi=self.from_wsi,
            tile_um=self.tile_um,
            rois=self.rois,
            roi_method=self.roi_method,
            pool=self.pool,
            transform=self.transform,
            **self.interleave_kwargs
        )
        self.close = queue_retriever.close
        try:
            for record in queue_retriever:
                yield self._parser(*record)
        # Closes open files if iterator terminated early
        except GeneratorExit as e:
            log.debug("Generator exit triggered")
            queue_retriever.close()
            del(queue_retriever)
            raise e

    def close(self) -> None:
        pass

    def get_details(self, idx):
        raise NotImplementedError


class StyleGAN2Interleaver(InterleaveIterator):
    """Iterator to enable compatibility with StyleGAN2."""

    def __init__(
        self,
        resolution=None,
        xflip=None,
        normalizer=None,
        normalizer_source=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if normalizer:
            self.normalizer = sf.norm.autoselect(
                normalizer,
                source=normalizer_source,
                device='cpu',
                backend='torch'
            )

    def get_label(self, idx: Any) -> Any:
        """Returns a random label. Used for compatibility with StyleGAN2."""
        if self.use_labels and self.model_type == 'categorical':
            return random.choices(
                self.unique_labels,
                weights=self.label_prob, # type: ignore
                k=1
            )[0]
        elif self.use_labels:
            return [np.random.rand()]
        else:
            return np.zeros((1,))


class LocLabelInterleaver(StyleGAN2Interleaver):
    """Pytorch Iterable Dataset that interleaves tfrecords with the
    as the `InterleaveIterator`, but applies tile-specific labels.
    """

    def __init__(
        self,
        loc_labels: str,
        resolution: Any = None,
        xflip: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes an InterleaveIterator modified to use tile-level labels.

        Args:
            loc_labels (str): Location of parquet-format pandas DataFrame
                containing tile-level labels. Labels are indexed by the slide
                name and X/Y location, with the format {slide}-{loc_x}-{loc_y}.
                Labels are determined by the `label` columns.
        """
        super().__init__(*args, **kwargs)

        self.df = pd.read_parquet(loc_labels)
        if 'label' not in self.df.columns:
            raise ValueError('Could not find column "label" in the '
                             'loc_labels dataset.')

        self.incl_loc = True
        first_row  = next(self.df.itertuples())
        self._label_shape = first_row.label.shape

    @property
    def label_shape(self) -> Union[int, Tuple[int, ...]]:
        """For use with StyleGAN2"""
        return self._label_shape

    def _parser(  # type: ignore
        self,
        image: torch.Tensor,
        slide: str,
        loc_x: int,
        loc_y: int
    ) -> List[torch.Tensor]:
        """Parses an image. Labels determined from the tile-level DataFrame.

        Args:
            image (torch.Tensor): Image.
            slide (str): Slide name.
            loc_x (int): Tile X-coordinate location on the corresponding slide.
            loc_y (int): Tile Y-coordinate location on the corresponding slide.

        Returns:
            List[torch.Tensor]: image, label, and slide
            (slide included if if self.incl_slidenames is True)
        """

        label_key = f'{slide}-{loc_x}-{loc_y}'
        label = torch.tensor(self.df.iloc[self.df.index.get_loc(label_key)].values)[0]

        image = whc_to_cwh(image)
        to_return = [image, label]  # type: List[Any]

        if self.incl_slidenames:
            to_return += [slide]
        return to_return

    def get_label(self, idx: Any) -> Any:
        """Returns a random label. Used for compatibility with StyleGAN2."""
        return np.random.rand(*self.label_shape)


def _get_images_by_dir(directory: str) -> List[str]:
    files = [
        f for f in listdir(directory)
        if ((isfile(join(directory, f)))
            and (sf.util.path_to_ext(f) in ("jpg", "jpeg", "png")))
    ]
    return files


def _apply_otsu(wsi):
    wsi.qc('otsu')
    return wsi


def multi_slide_loader(
    slides: List["sf.WSI"],
    splits: Optional[Dict[str, float]] = None,
    shard: Optional[Tuple[int, int]] = None,
    infinite: bool = True,
    **kwargs
) -> Iterable[Union[Dict[str, np.ndarray],
                    Tuple[Dict[str, np.ndarray],
                    Dict[str, List[np.ndarray]]]]]:
    """Create an iterator by reading and merging multiple tfrecord datasets.

    Args:
        paths (list of str): List of tfrecord paths.
        indices (dict): dict mapping tfrecord names to index paths.
            Input index path pattern.
        splits (dict):  Dictionary of (key, value) pairs, where the key is used
            to construct the data and index path(s) and the value determines
            the contribution of each split to the batch.
        infinite (bool, optional): Whether the returned iterator should be
            infinite or not. Defaults to True.

    Returns:

        it (iterator): A repeating iterator that generates batches of data.
    """
    if splits is not None:
        splits_list = splits
    else:
        splits_list = np.array(  # type: ignore
            [0.5 for t in range(len(slides))]
        )
    loaders = [slide.torch(lazy_iter=True,
                           shard=shard,
                           infinite=infinite,
                           **kwargs)
               for slide in slides]
    return RandomSampler(loaders, splits_list, shard=None)


def is_cwh(img: torch.Tensor) -> torch.Tensor:
    return (len(img.shape) == 3 and img.shape[0] == 3
            or (len(img.shape) == 4 and img.shape[1] == 3))


def is_whc(img: torch.Tensor) -> torch.Tensor:
    return img.shape[-1] == 3


def cwh_to_whc(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) == 3:
        return img.permute(1, 2, 0)  # CWH -> WHC
    elif len(img.shape) == 4:
        return img.permute(0, 2, 3, 1)  # BCWH -> BWHC
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")


def whc_to_cwh(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) == 3:
        return img.permute(2, 0, 1)  # WHC => CWH
    elif len(img.shape) == 4:
        return img.permute(0, 3, 1, 2)  # BWHC -> BCWH
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")


def auto_gaussian(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Perform Gaussian blur on an image with a given sigma, automatically
    calculating the appropriate Gaussian kernel size.

    Args:
        image (torch.Tensor): Image or batch of images.
        sigma (float): Sigma.

    Returns:
        torch.Tensor: Image(s) with Gaussian blur applied.
    """
    opt_kernel = int((sigma * 4) + 1)
    if opt_kernel % 2 == 0:
        opt_kernel += 1
    return torchvision.transforms.GaussianBlur(opt_kernel, sigma=sigma)(image)


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


def preprocess_uint8(
    img: torch.Tensor,
    normalizer: Optional["StainNormalizer"] = None,
    standardize: bool = True,
    resize_px: Optional[int] = None,
    resize_method: Optional["InterpolationMode"] = None,
    resize_aa: bool = True,
) -> torch.Tensor:
    """Process batch of tensorflow images, resizing, normalizing,
    and standardizing.

    Args:
        img (tf.Tensor): Batch of tensorflow images (uint8).
        normalizer (sf.norm.StainNormalizer, optional): Normalizer.
            Defaults to None.
        standardize (bool, optional): Standardize images. Defaults to True.
        resize_px (Optional[int], optional): Resize images. Defaults to None.
        resize_method (str, optional): Interpolation mode for resizing. Must
            be a valid torchvision.transforms.InterpolationMode. Defaults to
            BICUBIC.
        resize_aa (bool, optional): Apply antialiasing during resizing.
            Defaults to True.

    Returns:
        Dict[str, tf.Tensor]: Processed image.
    """
    if resize_px is not None:
        if resize_method is None:
            resize_method = torchvision.transforms.InterpolationMode.BICUBIC
        img = transforms.functional.resize(
            img,
            size=resize_px,
            interpolation=resize_method,
            antialias=resize_aa
        )
    if normalizer is not None:
        img = normalizer.torch_to_torch(img)  # type: ignore
    if standardize:
        img = convert_dtype(img, torch.float32)
    return img


def _decode_image(
    image: Union[bytes, str, torch.Tensor],
    img_type: str,
    standardize: bool = False,
    normalizer: Optional["StainNormalizer"] = None,
    augment: bool = False,
    device: Optional[torch.device] = None,
    transform: Optional[Any] = None,
) -> torch.Tensor:
    """Decodes image. Torch implementation; different than sf.io.tensorflow"""

    if img_type != 'numpy':
        np_data = torch.from_numpy(np.fromstring(image, dtype=np.uint8))
        image = cwh_to_whc(torchvision.io.decode_image(np_data))
        # Alternative method using PIL decoding:
        # image = np.array(Image.open(BytesIO(img_string)))
    assert isinstance(image, torch.Tensor)

    def random_jpeg_compression(img):
        img = torchvision.io.encode_jpeg(
            whc_to_cwh(img),
            quality=(torch.rand(1)[0]*50 + 50)
        )
        return cwh_to_whc(torchvision.io.decode_image(img))

    if augment is True or (isinstance(augment, str) and 'j' in augment):
        image = torch.where(
            torch.rand(1)[0] < 0.5,
            random_jpeg_compression(image),
            image
        )
    if augment is True or (isinstance(augment, str) and 'r' in augment):
        # Rotate randomly 0, 90, 180, 270 degrees
        image = torch.rot90(image, np.random.choice(range(5)))
    if augment is True or (isinstance(augment, str) and 'x' in augment):
        image = torch.where(
            torch.rand(1)[0] < 0.5,
            torch.fliplr(image),
            image
        )
    if augment is True or (isinstance(augment, str) and 'y' in augment):
        image = torch.where(
            torch.rand(1)[0] < 0.5,
            torch.flipud(image),
            image
        )
    if augment is True or (isinstance(augment, str) and 'b' in augment):
        # image = image.to(device)
        image = whc_to_cwh(image)
        image = torch.where(
            (torch.rand(1)[0] < 0.1),  # .to(device),
            torch.where(
                (torch.rand(1)[0] < 0.5),  # .to(device),
                torch.where(
                    (torch.rand(1)[0] < 0.5),  # .to(device),
                    torch.where(
                        (torch.rand(1)[0] < 0.5),  # .to(device),
                        auto_gaussian(image, sigma=2.0),
                        auto_gaussian(image, sigma=1.5),
                    ),
                    auto_gaussian(image, sigma=1.0)
                ),
                auto_gaussian(image, sigma=0.5),
            ),
            image
        )
        image = cwh_to_whc(image)
        # image = image.cpu()
    if normalizer:
        image = normalizer.torch_to_torch(
            image,
            augment=(isinstance(augment, str) and 'n' in augment)
        )
    if standardize:
        # Note: not the same as tensorflow's per_image_standardization
        # Convert back: image = (image + 1) * (255/2)
        image = image / 127.5 - 1
    if transform:
        image = cwh_to_whc(transform(whc_to_cwh(image)))
    return image


def worker_init_fn(worker_id) -> None:
    np.random.seed(np.random.get_state()[1][0])  # type: ignore


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
        augment (str): Image augmentations to perform. String containing
            characters designating augmentations. 'x' indicates random
            x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG
            compression/decompression at random quality levels. Passing either
            'xyrj' or True will use all augmentations.

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
                features['image_raw'] = _decode_image(
                    img,
                    img_type,
                    standardize,
                    normalizer,
                    augment
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


def interleave(
    paths: List[str],
    prob_weights: Optional[Dict[str, float]] = None,
    incl_loc: bool = False,
    clip: Optional[Dict[str, int]] = None,
    infinite: bool = True,
    augment: Union[bool, str] = False,
    standardize: bool = True,
    normalizer: Optional["StainNormalizer"] = None,
    num_threads: int = 4,
    chunk_size: int = 1,
    num_replicas: int = 1,
    rank: int = 0,
    indices: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    from_wsi: bool = False,
    tile_px: Optional[int] = None,
    tile_um: Optional[int] = None,
    rois: Optional[List[str]] = None,
    roi_method: str = 'auto',
    pool: Optional[Any] = None,
    transform: Optional[Any] = None,
    tfrecord_parser: Optional[Callable] = None,
):

    """Returns a generator that interleaves records from a collection of
    tfrecord files, sampling from tfrecord files randomly according to
    balancing if provided (requires manifest). Assumes TFRecord files are
    named by slide.

    Different than tensorflow backend implementation (sf.io.tensorflow).
    Supports Pytorch. Use interleave_dataloader for the torch DataLoader class;
    use this function directly to get images from a generator with no PyTorch
    data processing.

    Args:
        paths (list(str)): List of paths to TFRecord files or slides.
        prob_weights (dict, optional): Dict mapping tfrecords to probability of
            including in batch. Defaults to None.
        incl_loc (bool, optional): Include loc_x and loc_y as additional
            returned variables. Defaults to False.
        clip (dict, optional): Dict mapping tfrecords to number of tiles to
            take per tfrecord. Defaults to None.
        infinite (bool, optional): Create an finite dataset. WARNING: If
            infinite is False && balancing is used, some tiles will be skipped.
            Defaults to True.
        labels (dict, optional): Dict mapping slide names to outcome labels,
            used for balancing. Defaults to None.
        augment (str): Image augmentations to perform. String containing
            characters designating augmentations. 'x' indicates random
            x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG
            compression/decompression at random quality levels. Passing either
            'xyrj' or True will use all augmentations.
        standardize (bool, optional): Standardize images to (0,1).
            Defaults to True.
        normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
            Normalizer to use on images. Defaults to None.
        manifest (dict, optional): Dataset manifest containing number of tiles
            per tfrecord.
        num_threads (int, optional): Number of threads to use decoding images.
            Defaults to 4.
        chunk_size (int, optional): Chunk size for image decoding.
            Defaults to 8.
        num_replicas (int, optional): Number of total workers reading the
            dataset with this interleave function, defined as number of
            gpus * number of torch DataLoader workers. Used to interleave
            results among workers without duplications. Defaults to 1.
        rank (int, optional): Worker ID to identify which worker this
            represents. Used to interleave results among workers without
            duplications. Defaults to 0 (first worker).
        indices (list(str)): Paths to TFRecord index files. If not provided,
            will generate. Defaults to None.
        pool (multiprocessing.Pool): Shared multiprocessing pool. Useful
            if from_wsi=True, for sharing a unified processing pool between
            dataloaders. Defaults to None.
    """
    if not len(paths):
        raise errors.TFRecordsNotFoundError
    if rank == 0:
        _path_type = "slides" if from_wsi else "tfrecords"
        log.debug(
            f'Interleaving {len(paths)} {_path_type}: '
            f'infinite={infinite}, num_replicas={num_replicas}'
        )
    if from_wsi and (not tile_um or not tile_px):
        raise ValueError("`tile_um` and `tile_px` required for interleave() "
                         "if `from_wsi=True`")
    if prob_weights is not None:
        assert len(prob_weights) == len(paths)
    else:
        prob_weights = None
    should_close = False if pool is not None else True

    if incl_loc:
        features_to_return = ['image_raw', 'slide', 'loc_x', 'loc_y']
    else:
        features_to_return = ['image_raw', 'slide']

    if from_wsi:
        assert tile_um is not None and tile_px is not None
        if rank == 0:
            log.info(f"Reading {len(paths)} slides and thresholding...")

        # ---- Load slides and apply Otsu thresholding ------------------------
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
            if isinstance(path, bytes):
                path= path.decode('utf-8')
            try:
                wsi = sf.WSI(
                    path,
                    tile_px,
                    tile_um,
                    rois=rois,
                    roi_method=roi_method,
                    verbose=False
                )
                wsi_list += [wsi]
            except errors.SlideLoadError as e:
                log.error(f"Error reading slide {path}: {e}")
                to_remove += [path]
        for path in to_remove:
            paths.remove(path)
        for wsi in pool.imap(_apply_otsu, wsi_list):
            otsu_list += [wsi]
        est_num_tiles = sum([wsi.estimated_num_tiles for wsi in otsu_list])

        # ---- Prepare parsing -----------------------------------------------
        img_type = 'numpy'

        def base_parser(record):
            if type(features_to_return) == dict:
                return {
                    label: record[f]
                    for label, f in features_to_return.items()
                }
            else:
                return [record[f] for f in features_to_return]

        # ---- Interleave from slides -----------------------------------------
        random_sampler = multi_slide_loader(
            otsu_list,
            pool=pool,
            splits=prob_weights,
            shard=(rank, num_replicas),
            incl_slidenames=True,
            incl_loc=incl_loc,
            grayspace_fraction=1,
            infinite=infinite
        )
        sampler_iter = iter(random_sampler)
    else:
        # ---- Get the base TFRecord parser, based on the first tfrecord ------
        _, img_type = detect_tfrecord_format(paths[0])
        if tfrecord_parser is not None:
            base_parser = tfrecord_parser
        else:
            base_parser = get_tfrecord_parser(
                paths[0],
                features_to_return,
                decode_images=False,
                to_numpy=False
            )
        # ---- Set up TFRecord indexes for sharding ---------------------------
        # Index files not created in this interleave function, as there may be
        # multiple instances of this function running across processes,
        # & having each create indices would result in conflicts / corruption.
        if indices is None:
            indices = []

            def load_index(tfr):
                tfr = tfr.decode('utf-8')
                try:
                    index = tfrecord2idx.load_index(tfr)
                except OSError:
                    raise errors.TFRecordsError(
                        f"Could not find index path for TFRecord {tfr}"
                    )
                return index

            if pool is None:
                pool = mp.dummy.Pool(16)
            log.debug("Loading indices...")
            for index in pool.imap(load_index, paths):
                indices += [index]
            pool.close()

        # ---- Interleave and batch datasets ----------------------------------
        random_sampler = MultiTFRecordDataset(
            paths,
            indices,
            prob_weights,
            shard=(rank, num_replicas),
            clip=[clip[(t if isinstance(t, str) else t.decode('utf-8'))] for t in paths] if clip else None,
            infinite=infinite
        )
        sampler_iter = iter(random_sampler)

    # Worker to decode images and process records
    def threading_worker(record):
        record = base_parser(record)
        record[0] = _decode_image(
            record[0],  # Image is the first returned variable
            img_type=img_type,
            standardize=standardize,
            normalizer=normalizer,
            augment=augment,
            transform=transform,
            device=device
        )
        return record

    # Randomly interleaves datasets according to weights, reading parsed
    # records to a buffer and sending parsed results to a queue after
    # reaching a set buffer size
    class QueueRetriever:
        def __init__(self, sampler, num_threads):
            self.sampler = sampler
            self.closed = False
            self.raw_q = Queue(num_threads)
            self.proc_q = Queue(num_threads)
            self.n_threads = num_threads
            self.n_closed = 0
            self.il_closed = False
            self._close_complete = False

            def interleaver():
                msg = []
                while not self.closed:
                    try:
                        record = next(sampler_iter)
                        msg += [record]
                        if len(msg) < chunk_size:
                            continue
                        else:
                            self.raw_q.put(msg)
                            msg = []
                    except (StopIteration):
                        break
                    except (ValueError, OSError):  # Occurs when files closed
                        break
                self.raw_q.put(msg)
                for _ in range(self.n_threads):
                    self.raw_q.put(None)
                self.il_closed = True

            # Reads a buffer batch of images/labels and processes images
            def decoder():
                while True:
                    records = self.raw_q.get()
                    if records is None:
                        break
                    decoded = [threading_worker(record) for record in records]
                    self.proc_q.put(decoded)
                self.proc_q.put(None)

            # Parallelize the tfrecord reading interleaver
            # and the image processing decoder
            self.il_thread = threading.Thread(target=interleaver)
            self.il_thread.start()
            self.proc_threads = [
                threading.Thread(target=decoder)
                for _ in range(self.n_threads)
            ]
            for proc in self.proc_threads:
                proc.start()

        def __iter__(self):
            while True:
                record = self.proc_q.get()
                if record is None:
                    self.n_closed += 1
                    if self.n_closed == self.n_threads:
                        break
                else:
                    for item in record:
                        yield item

        def __del__(self):
            self.close()

        def close(self):
            if self._close_complete:
                return
            log.debug("Closing QueueRetriever")
            self.closed = True

            # Clear out the queue
            while self.n_closed < self.n_threads:
                record = self.proc_q.get()
                if record is None:
                    self.n_closed += 1

            if from_wsi and should_close:
                pool.close()
            else:
                self.sampler.close()
            self._close_complete = True

    return QueueRetriever(random_sampler, num_threads)


def interleave_dataloader(
    tfrecords: List[str],
    img_size: int,
    batch_size: Optional[int],
    *,
    num_replicas: int = 1,
    labels: Optional[Labels] = None,
    prefetch_factor: int = 2,
    num_workers: Optional[int] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    from_wsi: bool = False,
    **kwargs
) -> torch.utils.data.DataLoader:

    """Prepares a PyTorch DataLoader with a new InterleaveIterator instance,
    interleaving tfrecords and processing labels and tiles, with support for
    scaling the dataset across GPUs and dataset workers.

    Args:
        tfrecords (list(str)): List of paths to TFRecord files.
        img_size (int): Tile size in pixels.
        batch_size (int): Batch size.

    Keyword Args:
        prob_weights (dict, optional): Dict mapping tfrecords to probability
            of including in batch. Defaults to None.
        clip (dict, optional): Dict mapping tfrecords to number of tiles to
            take per tfrecord. Defaults to None.
        onehot (bool, optional): Onehot encode labels. Defaults to False.
        incl_slidenames (bool, optional): Include slidenames as third returned
            variable. Defaults to False.
        incl_loc (bool, optional): Include loc_x and loc_y as additional
            returned variables. Defaults to False.
        infinite (bool, optional): Infinitely repeat data. Defaults to True.
        rank (int, optional): Worker ID to identify this worker.
            Used to interleave results.
            among workers without duplications. Defaults to 0 (first worker).
        num_replicas (int, optional): Number of GPUs or unique instances which
            will have their own DataLoader. Used to interleave results among
            workers without duplications. Defaults to 1.
        labels (dict, optional): Dict mapping slide names to outcome labels,
            used for balancing. Defaults to None.
        normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
            Normalizer to use on images. Defaults to None.
        chunk_size (int, optional): Chunk size for image decoding.
            Defaults to 1.
        prefetch_factor (int, optional): Number of batches to prefetch in each
            SlideflowIterator. Defaults to 1.
        manifest (dict, optional): Dataset manifest containing number of tiles
            per tfrecord.
        balance (str, optional): Batch-level balancing. Options: category,
            patient, and None. If infinite is not True, will drop tiles to
            maintain proportions across the interleaved dataset.
        augment (str, optional): Image augmentations to perform. String
            containing characters designating augmentations. 'x' indicates
            random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG
            compression/decompression at random quality levels. Passing either
            'xyrj' or True will use all augmentations.
        standardize (bool, optional): Standardize images to (0,1).
            Defaults to True.
        num_workers (int, optional): Number of DataLoader workers.
            Defaults to 2.
        persistent_workers (bool, optional): Sets the DataLoader
            persistent_workers flag. Defaults toNone (4 if not using a SPAMS
            normalizer, 1 if using SPAMS).
        pin_memory (bool, optional): Pin memory to GPU. Defaults to True.
        drop_last (bool, optional): Drop the last non-full batch.
            Defaults to False.

    Returns:
        torch.utils.data.DataLoader

    """
    if batch_size is None:
        replica_batch_size = None
    else:
        replica_batch_size = batch_size // num_replicas
    if from_wsi and num_workers:
        raise ValueError("Option `from_wsi=True` incompatible with "
                         "num_workers > 0")

    if num_workers is None and os.cpu_count():
        num_workers = os.cpu_count() // 4  # type: ignore
    elif num_workers is None:
        num_workers = 8
    log.debug(f"Using num_workers={num_workers}")
    if 'num_threads' not in kwargs and os.cpu_count():
        kwargs['num_threads'] = int(math.ceil(os.cpu_count() / max(num_workers, 1)))
        log.debug(f"Threads per worker={kwargs['num_threads']}")

    iterator = InterleaveIterator(
        tfrecords=tfrecords,
        img_size=img_size,
        use_labels=(labels is not None),
        num_replicas=num_replicas,
        labels=labels,
        from_wsi=from_wsi,
        **kwargs
    )
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = torch.utils.data.DataLoader(
        iterator,
        batch_size=replica_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor
    )
    dataloader.num_tiles = iterator.num_tiles
    dataloader.dataset.dataloader = dataloader  # type: ignore
    # Give a closing function to the DataLoader
    # to cleanup open files from iter()
    dataloader.close = iterator.close
    return dataloader
