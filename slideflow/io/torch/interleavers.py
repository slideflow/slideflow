import multiprocessing as mp
import random
import numpy as np
import pandas as pd
import torchvision
import torch
from typing import (TYPE_CHECKING, Any, Callable, Dict, List,
                    Optional, Tuple, Union)

import slideflow as sf
from slideflow.io.io_utils import detect_tfrecord_format
from slideflow.tfrecord.torch.dataset import IndexedMultiTFRecordDataset
from slideflow.util import Labels, log, to_onehot, detuple

from .img_utils import whc_to_cwh
from .augment import compose_augmentations

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

class IndexedInterleaver(IndexedMultiTFRecordDataset):

    def __init__(
        self,
        tfrecords: List[str],
        *,
        labels: Optional[Labels] = None,
        incl_slidenames: bool = False,
        incl_loc: bool = False,
        rank: int = 0,
        num_replicas: int = 1,
        augment: Union[bool, str] = False,
        standardize: bool = True,
        normalizer: Optional["StainNormalizer"] = None,
        clip: Optional[Dict[str, int]] = None,
        use_labels: bool = True,
        onehot: bool = False,
        indices: Optional[List[np.ndarray]] = None,
        transform: Optional[Any] = None,
        tfrecord_parser: Optional[Callable] = None,
        **kwargs
    ):
        """Interleave TFRecords with an indexable ``torch.utils.data.Dataset``.

        Provides an alternative TFRecord IO pipeline to ``InterleaveIterator``,
        which only supports Iterable-style datasets. This class supports
        both Iterable and Indexable datasets.

        Differences from ``InterleaveIterator``:
         - Supports direct indexing.
         - Does not support infinite looping ("infinite" argument).
         - Does not support weighted random sampling ("prob_weights" argument).
         - Does not support dynamic reading from WSI ("from_wsi", "tile_um", "rois", "roi_method", and "pool" arguments).

        Args:
            tfrecords (list(str)): Path to tfrecord files to interleave.

        Keyword Args:
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
            normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
                Normalizer. Defaults to None.
            clip (list(int), optional): Array of maximum tiles to take for each
                tfrecord. Defaults to None.
            use_labels (bool, optional): Enable use of labels (disabled for
                non-conditional GANs). Defaults to True.
            onehot (bool, optional): Onehot encode outcomes. Defaults to False.
            indices (numpy.ndarray, optional): Indices in form of array,
                with np.loadtxt(index_path, dtype=np.int64) for each tfrecord.
                Defaults to None.
            transform (Callable, optional): Arbitrary torchvision transform
                function. Performs transformation after augmentations but
                before standardization. Defaults to None.
            tfrecord_parser (Callable, optional): Custom parser for TFRecords.
                Defaults to None.
            compression_type (str, optional): Compression type for TFRecords.
                Either 'gzip' or None. Defaults to None.
            shuffle (bool): Shuffle records within TFRecord files during
                reading. Defaults to False.
            seed (int, optional): Seed for random TFRecord interleaving and
                intra-tfrecord shuffling. Defaults to None.

        """
        self.readers = []
        self.tfrecords = np.array(tfrecords).astype(np.string_)
        if not len(self.tfrecords):
            raise ValueError("No tfrecords provided.")
        self.indices = self._load_indices(indices)
        self.incl_slidenames = incl_slidenames
        self.incl_loc = incl_loc
        self.use_labels = use_labels
        self.onehot = onehot
        self.parser = self.build_parser(tfrecord_parser)
        self.img_format = detect_tfrecord_format(self.tfrecords[0])[1]
        (self.labels,
         self.unique_labels,
         self.label_prob,
         self.num_outcomes) = _process_labels(labels, onehot=onehot)

        # Automatically set shard to rank/num_replicas
        self.rank = rank
        self.num_replicas = num_replicas
        if self.rank == 0:
            log.info(
                f'Interleaving {len(self.tfrecords)} tfrecords: '
                f'num_replicas={self.num_replicas}'
            )

        # Clip tfrecords.
        if clip is not None:
            _clip = [clip[(t if isinstance(t, str) else t.decode('utf-8'))] for t in self.tfrecords]
        else:
            _clip = None

        # Set up image transformations.
        self._image_transform = compose_augmentations(
            augment=augment,
            standardize=standardize,
            normalizer=normalizer,
            transform=transform
        )

        # Prepare TFRecord interleaver.
        super().__init__(
            tfrecords,
            indices=self.indices,
            shard=(self.rank, self.num_replicas),
            clip=_clip,
            transform=self.parser,
            **kwargs
        )

    def __repr__(self) -> str:
        n_records = self.tfrecords.shape[0]
        msg = f"<IndexedInterleaver object (num_records={n_records}, num_tiles"
        msg += f"={self.num_tiles}, infinite={self.infinite}, rank=("
        msg += f"{self.rank} of {self.num_replicas}), augment={self.augment}, "
        msg += f"standardize={self.standardize})>"
        return msg

    def _load_indices(self, indices: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Load TFRecord index files."""
        if indices is None:
            indices = []
            with mp.dummy.Pool(16) as pool:
                log.debug("Loading indices...")
                for index in pool.imap(load_index, self.tfrecords):
                    indices += [index]
            return indices
        else:
            log.debug("Using provided indices.")
            return indices

    def _label_parser(
        self,
        slide: str,
        loc_x: Optional[int] = None,
        loc_y: Optional[int] = None
    ):
        """Parse labels and location information from a record."""

        # Label.
        if self.labels is not None and self.num_outcomes > 1:
            label = self.labels[slide]
            label = {
                f'out-{i}': torch.tensor(l)
                for i, l in enumerate(label)  # type: ignore
            }
        elif self.labels is not None:
            label = self.labels[slide]
            label = torch.tensor(label)
        else:
            label = torch.tensor(0)

        # Slide/location information.
        if self.incl_slidenames and self.incl_loc:
            return label, slide, loc_x, loc_y
        elif self.incl_slidenames:
            return label, slide
        elif self.incl_loc:
            return label, loc_x, loc_y
        else:
            return label,

    def build_parser(
        self,
        tfrecord_parser: Optional[Callable] = None
    ) -> Callable:
        """Build a parser function for TFRecords.

        The parser will be responsible for processing images and labels.

        """
        ftrs = ['image_raw', 'slide']
        if self.incl_loc:
            ftrs += ['loc_x', 'loc_y']
        base_parser = tfrecord_parser or get_tfrecord_parser(self.tfrecords[0],
                                                             ftrs,
                                                             decode_images=False)

        def parser(*args):
            """Parse an image and slide/location information."""
            img, *out = base_parser(*args)
            img = decode_image(img, img_type=self.img_format)
            img = whc_to_cwh(img)
            img = self._image_transform(img)
            out = self._label_parser(*out)
            return detuple(img, out)

        return parser


class InterleaveIterator(torch.utils.data.IterableDataset):
    """Pytorch Iterable Dataset that interleaves tfrecords with the
    interleave() function below. Serves as a bridge between the python
    generator returned by interleave() and the pytorch DataLoader class.
    """

    def __init__(
        self,
        tfrecords: List[str],
        img_size: int,
        *,
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
            img_size (int): Image width in pixels.

        Keyword Args:
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
        (self.labels,
         self.unique_labels,
         self.label_prob,
         self.num_outcomes) = _process_labels(labels, onehot=onehot)

    @property
    def name(self) -> str:
        return 'slideflow-interleave-iterator'

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
        *args: Any,
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
        super().__init__(*args, **kwargs)

        self.df = pd.read_parquet(tile_labels)
        if 'label' not in self.df.columns:
            raise ValueError('Could not find column "label" in the '
                             'tile_labels dataset.')

        self.incl_loc = True
        first_row  = next(self.df.itertuples())
        self._label_shape = first_row.label.shape
        if self.rank == 0:
            log.warning(f"Number of tiles ({self.num_tiles}) does not equal the "
                        f"number of labels ({len(self.df)}). ")

        self._prepare_tfrecord_subsample()

    def _prepare_tfrecord_subsample(self):
        """Prepare custom TFRecord indices to only read tiles in the labels dataframe."""

        # Prepare TFRecord subsample if there are fewer tiles in the
        # tiles dataframe than there are in the TFRecords
        if self.indices is None and (self.num_tiles != len(self.df)):

            self.indices = []
            worker_info = torch.utils.data.get_worker_info()
            if self.rank == 0:
                log.info("Subsampling TFRecords using tile-level labels...")

            n_tiles = 0
            with mp.dummy.Pool(16) as pool:

                # Load the original (full) indices
                for index, tfr in zip(pool.imap(load_index, self.tfrecords), self.tfrecords):
                    tfr = tfr.decode('utf-8')
                    slide = sf.util.path_to_name(tfr)
                    loc = sf.io.get_locations_from_tfrecord(tfr)

                    # Check which TFRecord indices are in the labels dataframe
                    in_df = np.array([f'{slide}-{x}-{y}' in self.df.index for (x,y) in loc])

                    # Subsample indices based on what is in the labels dataframe
                    ss_index = index[in_df]
                    n_tiles += len(ss_index)

                    self.indices += [ss_index]
            self.num_tiles = n_tiles

            if self.rank == 0:
                log.info("TFRecord subsampling complete (total tiles: {}).".format(self.num_tiles))

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
        df_idx = self.df.index.get_loc(label_key)
        label = torch.tensor(self.df.iloc[df_idx].label)

        image = whc_to_cwh(image)
        to_return = [image, label]  # type: List[Any]

        if self.incl_slidenames:
            to_return += [slide]
        return to_return

    def get_label(self, idx: Any) -> Any:
        """Returns a random label. Used for compatibility with StyleGAN2."""
        idx = np.random.randint(len(self.df))
        return self.df.iloc[idx].label

# -------------------------------------------------------------------------

def _process_labels(
    labels: Optional[Dict[str, Any]] = None,
    onehot: bool = False
) -> Tuple[Optional[Dict[str, Any]],
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
    if labels is not None:
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
    else:
        unique_labels = None
        label_prob = None  # type: ignore
        num_outcomes = 1
    return labels, unique_labels, label_prob, num_outcomes