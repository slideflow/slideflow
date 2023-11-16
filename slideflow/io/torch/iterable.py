"""Iterable-style TFRecord interleavers for PyTorch.

Includes support for streaming data from whole-slide images, as well as StyleGAN2
compatibility.

"""

import multiprocessing as mp
import random
import threading
import numpy as np
import pandas as pd
import torchvision
import torch
from queue import Queue
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Tuple, Union)

import slideflow as sf
from slideflow import errors
from slideflow.io import detect_tfrecord_format
from slideflow.tfrecord.torch.dataset import MultiTFRecordDataset
from slideflow.tfrecord.iterator_utils import RandomSampler
from slideflow.util import Labels, log

from .img_utils import whc_to_cwh
from .img_utils import decode_image
from .data_utils import process_labels, load_index, get_tfrecord_parser

from .augment import compose_augmentations

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

class InterleaveIterator(torch.utils.data.IterableDataset):
    """Pytorch Iterable Dataset that interleaves tfrecords with the
    interleave() function below. Serves as a bridge between the python
    generator returned by interleave() and the pytorch DataLoader class.
    """

    def __init__(
        self,
        tfrecords: List[str],
        *,
        img_size: Optional[int] = None,
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

        Keyword Args:
            img_size (int): Image width in pixels.
            labels (dict, optional): Dict mapping slide names to labels.
                Defaults to None.
            incl_slidenames (bool, optional): Include slide names when iterated
                (returns image, label, slide). Defaults to False.
            incl_loc (bool, optional): Include location info (tile center
                coordinates). Returns samples in the form ``(returns ..., loc_x,
                loc_y)``. Defaults to False.
            rank (int, optional): Which GPU replica this dataset is used for.
                Assists with synchronization across GPUs. Defaults to 0.
            num_replicas (int, optional): Total number of GPU replicas.
                Defaults to 1.
            augment (str or bool): Image augmentations to perform. Augmentations include:

                * ``'x'``: Random horizontal flip
                * ``'y'``: Random vertical flip
                * ``'r'``: Random 90-degree rotation
                * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
                * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
                * ``'n'``: Random :ref:`stain_augmentation` (requires stain normalizer)

                Combine letters to define augmentations, such as ``'xyrjn'``.
                A value of True will use ``'xyrjb'``.
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
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
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
        if normalizer is not None and not isinstance(normalizer, sf.norm.StainNormalizer):
            raise ValueError(
                f"Expected normalizer to be type StainNormalizer, got: {type(normalizer)}"
            )
        self.tfrecords = np.array(tfrecords).astype(np.string_)
        self.prob_weights = None if prob_weights is None else np.array(prob_weights)
        self.clip = clip
        self.indices = indices
        self.img_size = img_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.augment = augment
        self.standardize = standardize
        self.infinite = infinite
        self.use_labels = use_labels
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.onehot = onehot
        self.incl_slidenames = incl_slidenames
        self.incl_loc = incl_loc
        self.num_tiles = num_tiles
        self.model_type = model_type
        self.from_wsi = from_wsi
        self.tile_um = tile_um
        self.rois = rois
        self.roi_method = roi_method
        self.pool = pool
        self.transform = transform
        self.interleave_kwargs = interleave_kwargs
        self._label_shape = None
        (self.labels,
         self.unique_labels,
         self.label_prob,
         self.num_outcomes) = process_labels(labels, onehot=onehot)
        if isinstance(self.labels, pd.DataFrame):
            self._prepare_tfrecord_subsample()

    @property
    def name(self) -> str:
        return 'slideflow-interleave-iterator'

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

        if self.labels is not None and not isinstance(self.labels, pd.DataFrame):
            label = self.labels[slide]
        elif self.labels is not None:
            label = self.labels.loc[f'{slide}-{loc_x}-{loc_y}'].label
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
            incl_loc=True,         # Read from TFRecord. Handle with ._parser()
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

    def _prepare_tfrecord_subsample(self):
        """Prepare custom TFRecord indices to only read tiles in the labels dataframe."""

        # Prepare TFRecord subsample if there are fewer tiles in the
        # tiles dataframe than there are in the TFRecords
        if (self.num_tiles != len(self.labels)):

            self.indices = []
            if self.rank == 0:
                log.debug("Subsampling TFRecords using tile-level labels...")

            n_tiles = 0
            with mp.dummy.Pool(16) as pool:

                # Load the original (full) indices
                for index, tfr in zip(pool.imap(load_index, self.tfrecords), self.tfrecords):
                    tfr = tfr.decode('utf-8')
                    slide = sf.util.path_to_name(tfr)
                    loc = sf.io.get_locations_from_tfrecord(tfr)

                    # Check which TFRecord indices are in the labels dataframe
                    in_df = np.array([f'{slide}-{x}-{y}' in self.labels.index for (x,y) in loc])

                    # Subsample indices based on what is in the labels dataframe
                    ss_index = index[in_df]
                    n_tiles += len(ss_index)

                    self.indices += [ss_index]

            if not n_tiles:
                raise ValueError("No tiles found in TFRecords matching the "
                                 "labels dataframe.")

            if self.rank == 0:
                diff = self.num_tiles - n_tiles
                log.debug(
                    "TFRecord subsampling complete (kept: {}, removed: {}).".format(
                        n_tiles, diff
                ))
                if len(self.labels) - n_tiles:
                    log.debug(
                        "{} labels in the dataframe have no corresponding tile.".format(
                            len(self.labels) - n_tiles
                        )
                    )
                if diff:
                    log.warning(f"Labels not found for {diff} tiles. These "
                                "tiles will be skipped.")
            self.num_tiles = n_tiles

    def close(self) -> None:
        pass


class StyleGAN2Interleaver(InterleaveIterator):
    """Iterator to enable compatibility with StyleGAN2."""

    def __init__(
        self,
        resolution=None,  # Ignored argument, for StyleGAN2/3 compatibility.
        xflip=None,       # Ignored argument, for StyleGAN2/3 compatibility.
        normalizer=None,
        normalizer_source=None,
        crop=None,
        resize=None,
        max_size=None,    #ignore argument, for StyleGAN2/3 compatibility.
        **kwargs
    ):
        super().__init__(**kwargs)

        # Assemble crop/resize transformations.
        transforms = []
        if crop is not None:
            transforms.append(torchvision.transforms.RandomCrop(crop))
        if resize is not None:
            transforms.append(torchvision.transforms.Resize(resize))
        if len(transforms):
            self.transform = torchvision.transforms.Compose(transforms)

        # Update the final image size.
        if resize is not None:
            self.img_size = resize
        elif crop is not None:
            self.img_size = crop
        if self.img_size is None:
            raise ValueError("Must specify either crop, resize, or img_size.")

        if normalizer:
            self.normalizer = sf.norm.autoselect(
                normalizer,
                source=normalizer_source,
                device='cpu',
                backend='torch'
            )

    @property
    def resolution(self) -> int:
        """For use with StyleGAN2"""
        return self.img_size

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """For use with StyleGAN2"""
        return (3, self.resolution, self.resolution)

    @property
    def num_channels(self) -> int:
        """For use with StyleGAN2"""
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def label_shape(self) -> Union[int, Tuple[int, ...]]:
        """For use with StyleGAN2"""
        if self.use_labels and self.unique_labels is not None:
            return self.unique_labels[0].shape
        elif self._label_shape is not None:
            return self._label_shape
        else:
            return 0

    @property
    def label_dim(self) -> int:
        """For use with StyleGAN2"""
        if self.use_labels:
            assert len(self.label_shape) == 1  # type: ignore
            return self.label_shape[0]  # type: ignore
        else:
            return 0

    @property
    def has_labels(self) -> bool:
        """For use with StyleGAN2"""
        return (self.use_labels
                and any(x != 0 for x in self.label_shape))  # type: ignore

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


class TileLabelInterleaver(StyleGAN2Interleaver):
    """Pytorch Iterable Dataset that interleaves tfrecords with the
    as the `InterleaveIterator`, but applies tile-specific labels.

    Labels should be onehot encoded.

    """
    def __init__(
        self,
        tile_labels: str,
        resolution: Any = None,  # Ignored, for StyleGAN2/3 compatibility.
        xflip: Any = None,       # Ignored, for StyleGAN2/3 compatibility.
        labels: Any = None,      # Ignored, for StyleGAN2/3 compatibility.
        **kwargs: Any,
    ) -> None:
        """Initializes an InterleaveIterator modified to use tile-level labels.

        Args:
            tile_labels (str): Location of parquet-format pandas DataFrame
                containing tile-level labels. Labels are indexed by the slide
                name and X/Y location, with the format {slide}-{loc_x}-{loc_y}.
                Labels are determined by the `label` columns. Labels should
                be onehot encoded.
        """
        super().__init__(labels=tile_labels, **kwargs)
        self._process_labels_df()
        if not isinstance(self.labels, pd.DataFrame):
            raise ValueError("Labels must be a pandas DataFrame.")

    def _process_labels_df(self) -> None:
        assert isinstance(self.labels, pd.DataFrame)
        first_row  = next(self.labels.itertuples())
        self._label_shape = first_row.label.shape
        if self.rank == 0 and (self.num_tiles != len(self.labels)):
            log.warning(f"Number of tiles ({self.num_tiles}) does not equal the "
                        f"number of labels ({len(self.labels)}). ")

    def get_label(self, idx: Any) -> Any:
        """Returns a random label. Used for compatibility with StyleGAN2."""
        idx = np.random.randint(len(self.labels))
        return self.labels.iloc[idx].label

# -----------------------------------------------------------------------------

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
        weights_list = np.array(                # type: ignore
            [0.5 for _ in range(len(slides))]
        )
    loaders = [slide.torch(lazy_iter=True,
                           shard=shard,
                           infinite=infinite,
                           **kwargs)
               for slide in slides]
    return RandomSampler(loaders, weights_list, shard=None)


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
        incl_loc (bool, optional): Include location info (tile center
            coordinates). Returns samples in the form ``(returns ..., loc_x,
            loc_y)``. Defaults to False.
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
