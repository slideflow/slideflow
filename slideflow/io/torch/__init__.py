import multiprocessing as mp
import threading
import numpy as np
import torchvision
import torch
import math
from torchvision import transforms
from queue import Queue
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Tuple, Union)


import slideflow as sf
from slideflow import errors
from slideflow.io import convert_dtype
from slideflow.io.io_utils import detect_tfrecord_format
from slideflow.tfrecord.torch.dataset import MultiTFRecordDataset, IndexedMultiTFRecordDataset
from slideflow.tfrecord.iterator_utils import RandomSampler
from slideflow.util import Labels, log, tfrecord2idx

from .img_utils import is_cwh, is_whc, as_cwh, as_whc, cwh_to_whc, whc_to_cwh
from .interleavers import (
    IndexedInterleaver, InterleaveIterator, 
    StyleGAN2Interleaver, TileLabelInterleaver
)
from .augment import (
    RandomCardinalRotation, RandomGaussianBlur, RandomJPEGCompression,
    RandomColorDistortion, decode_augmentation_string, compose_augmentations,
    random_jpeg_compression, compose_color_distortion
)

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer
    from torchvision.transforms import InterpolationMode

# -----------------------------------------------------------------------------

FEATURE_DESCRIPTION = {
    'image_raw': 'byte',
    'slide': 'byte',
    'loc_x': 'int',
    'loc_y': 'int'
}

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


def _apply_otsu(wsi):
    wsi.qc('otsu')
    return wsi


def multi_slide_loader(
    slides: List["sf.WSI"],
    weights: Optional[Union[List[float], np.ndarray]] = None,
    shard: Optional[Tuple[int, int]] = None,
    infinite: bool = True,
    **kwargs
) -> Iterable[Union[Dict[str, np.ndarray],
                    Tuple[Dict[str, np.ndarray],
                    Dict[str, List[np.ndarray]]]]]:
    """Create an iterator by reading and merging multiple slide dataloaders.

    Args:
        slides (list of str): List of slide paths.
        weights (list of float):  Weights for sampling from each slide.
            If not provided, will perform uniform sampling.
        shard (tuple(int, int), optional): If provided, will only extract
            tiles from the shard with index `shard[0]` out of `shard[1]`
            shards. Defaults to None.
        infinite (bool, optional): Whether the returned iterator should be
            infinite or not. Defaults to True.

    Returns:

        it (iterator): A repeating iterator that generates batches of data,
        interleaving from the provided slides.

    """
    if weights is not None:
        weights_list = weights
    else:
        weights_list = np.array(  # type: ignore
            [0.5 for t in range(len(slides))]
        )
    loaders = [slide.torch(lazy_iter=True,
                           shard=shard,
                           infinite=infinite,
                           **kwargs)
               for slide in slides]
    return RandomSampler(loaders, weights_list, shard=None)

# -------------------------------------------------------------------------

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


def decode_image(
    image: Union[bytes, str, torch.Tensor],
    *,
    img_type: Optional[str] = None,
    device: Optional[torch.device] = None,
    transform: Optional[Any] = None,
) -> torch.Tensor:
    """Decodes image string/bytes to Tensor (W x H x C).

    Torch implementation; different than sf.io.tensorflow.

    Args:
        image (Union[bytes, str, torch.Tensor]): Image to decode.

    Keyword args:
        img_type (str, optional): Image type. Defaults to None.
        device (torch.device, optional): Device to move image to.
            Defaults to None.
        transform (Callable, optional): Arbitrary torchvision transform function.
            Performs transformation after augmentations but before standardization.
            Defaults to None.

    """
    if img_type != 'numpy':
        np_data = torch.from_numpy(np.fromstring(image, dtype=np.uint8))
        image = cwh_to_whc(torchvision.io.decode_image(np_data))
        # Alternative method using PIL decoding:
        # image = np.array(Image.open(BytesIO(img_string)))

    assert isinstance(image, torch.Tensor)

    if device is not None:
        image = image.to(device)

    if transform is not None:
        image = transform(image)

    return image

# -------------------------------------------------------------------------

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
        augment (str or bool): Image augmentations to perform. Augmentations include:

            * ``'x'``: Random horizontal flip
            * ``'y'``: Random vertical flip
            * ``'r'``: Random 90-degree rotation
            * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
            * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
            * ``'n'``: Random :ref:`stain_augmentation` (requires stain normalizer)

            Combine letters to define augmentations, such as ``'xyrjn'``.
            A value of True will use ``'xyrjb'``.
        standardize (bool, optional): Standardize images to (0,1).
            Defaults to True.
        normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
            Normalizer to use on images. Defaults to None.
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
        from_wsi (bool): Generate predictions from tiles dynamically
            extracted from whole-slide images, rather than TFRecords.
            Defaults to False (use TFRecords).
        tile_px (int, optional): Size of tiles to extract from WSI, in pixels.
            Only used if from_wsi=True. Defaults to None.
        tile_um (int, optional): Size of tiles to extract from WSI, in
            microns. Only used if from_wsi=True. Defaults to None.
        rois (list(str), optional): List of ROI paths. Only used if
            from_wsi=True.  Defaults to None.
        roi_method (str, optional): Method for extracting ROIs. Only used if
            from_wsi=True. Defaults to 'auto'.
        pool (multiprocessing.Pool): Shared multiprocessing pool. Useful
            if ``from_wsi=True``, for sharing a unified processing pool between
            dataloaders. Defaults to None.
        transform (Callable, optional): Arbitrary torchvision transform
            function. Performs transformation after augmentations but
            before standardization. Defaults to None.
        tfrecord_parser (Callable, optional): Custom parser for TFRecords.
            Defaults to None.

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
                sf.util.num_cpu(default=8),
                initializer=sf.util.set_ignore_sigint
            )
        elif pool is None:
            pool = mp.dummy.Pool(sf.util.num_cpu(default=16))
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
            weights=prob_weights,
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
                decode_images=False
            )
        # ---- Set up TFRecord indexes for sharding ---------------------------
        # Index files not created in this interleave function, as there may be
        # multiple instances of this function running across processes,
        # & having each create indices would result in conflicts / corruption.
        if indices is None:
            indices = []
            if pool is None:
                pool = mp.dummy.Pool(16)
            log.debug("Loading indices...")
            for index in pool.imap(load_index, paths):
                indices += [index]
            pool.close()
        else:
            log.debug("Using provided indices.")

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

    # Compose augmentation transformations
    transform_fn = compose_augmentations(
        augment=augment,
        standardize=standardize,
        normalizer=normalizer,
        transform=transform,
        whc=True
    )

    # Worker to decode images and process records
    def threading_worker(record):
        record = base_parser(record)
        record[0] = decode_image(
            record[0],  # Image is the first returned variable
            img_type=img_type,
            transform=transform_fn,
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
    collate_fn: Optional[Callable] = None,
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
        augment (str or bool): Image augmentations to perform. Augmentations include:

            * ``'x'``: Random horizontal flip
            * ``'y'``: Random vertical flip
            * ``'r'``: Random 90-degree rotation
            * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
            * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
            * ``'n'``: Random :ref:`stain_augmentation` (requires stain normalizer)

            Combine letters to define augmentations, such as ``'xyrjn'``.
            A value of True will use ``'xyrjb'``.
        chunk_size (int, optional): Chunk size for image decoding.
            Defaults to 1.
        clip (dict, optional): Dict mapping tfrecords to number of tiles to
            take per tfrecord. Defaults to None.
        drop_last (bool, optional): Drop the last non-full batch.
            Defaults to False.
        from_wsi (bool): Generate predictions from tiles dynamically
            extracted from whole-slide images, rather than TFRecords.
            Defaults to False (use TFRecords).
        incl_loc (bool, optional): Include loc_x and loc_y as additional
            returned variables. Defaults to False.
        incl_slidenames (bool, optional): Include slidenames as third returned
            variable. Defaults to False.
        infinite (bool, optional): Infinitely repeat data. Defaults to True.
        indices (numpy.ndarray, optional): Indices in form of array,
            with np.loadtxt(index_path, dtype=np.int64) for each tfrecord.
            Defaults to None.
        labels (dict, optional): Dict mapping slide names to outcome labels,
            used for balancing. Defaults to None.
        max_size (bool, optional): Unused argument present for legacy
            compatibility; will be removed.
        model_type (str, optional): Used to generate random labels
            (for StyleGAN2). Not required. Defaults to 'categorical'.
        num_replicas (int, optional): Number of GPUs or unique instances which
            will have their own DataLoader. Used to interleave results among
            workers without duplications. Defaults to 1.
        num_tiles (int, optional): Dict mapping tfrecord names to number
            of total tiles. Defaults to None.
        num_workers (int, optional): Number of DataLoader workers.
            Defaults to 2.
        normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
            Normalizer. Defaults to None.
        onehot (bool, optional): Onehot encode labels. Defaults to False.
        persistent_workers (bool, optional): Sets the DataLoader
            persistent_workers flag. Defaults toNone (4 if not using a SPAMS
            normalizer, 1 if using SPAMS).
        pin_memory (bool, optional): Pin memory to GPU. Defaults to True.
        pool (multiprocessing.Pool): Shared multiprocessing pool. Useful
            if ``from_wsi=True``, for sharing a unified processing pool between
            dataloaders. Defaults to None.
        prefetch_factor (int, optional): Number of batches to prefetch in each
            SlideflowIterator. Defaults to 1.
        prob_weights (dict, optional): Dict mapping tfrecords to probability
            of including in batch. Defaults to None.
        rank (int, optional): Worker ID to identify this worker.
            Used to interleave results.
            among workers without duplications. Defaults to 0 (first worker).
        rois (list(str), optional): List of ROI paths. Only used if
            from_wsi=True.  Defaults to None.
        roi_method (str, optional): Method for extracting ROIs. Only used if
            from_wsi=True. Defaults to 'auto'.
        standardize (bool, optional): Standardize images to mean 0 and
            variance of 1. Defaults to True.
        tile_um (int, optional): Size of tiles to extract from WSI, in
            microns. Only used if from_wsi=True. Defaults to None.
        transform (Callable, optional): Arbitrary torchvision transform
            function. Performs transformation after augmentations but
            before standardization. Defaults to None.
        tfrecord_parser (Callable, optional): Custom parser for TFRecords.
            Defaults to None.

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

    if num_workers is None and sf.util.num_cpu():
        num_workers = max(sf.util.num_cpu() // 4, 1)  # type: ignore
    elif num_workers is None:
        num_workers = 8
    log.debug(f"Using num_workers={num_workers}")
    if 'num_threads' not in kwargs and sf.util.num_cpu():
        kwargs['num_threads'] = int(math.ceil(sf.util.num_cpu() / max(num_workers, 1)))
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
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )
    dataloader.num_tiles = iterator.num_tiles
    dataloader.dataset.dataloader = dataloader  # type: ignore
    # Give a closing function to the DataLoader
    # to cleanup open files from iter()
    dataloader.close = iterator.close
    return dataloader
