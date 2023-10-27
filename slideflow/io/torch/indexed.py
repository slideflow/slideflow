"""Indexable, map-style multi-TFRecord dataset & weighted sampler."""

import slideflow as sf
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch

from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from slideflow.io import detect_tfrecord_format
from slideflow.tfrecord.torch.dataset import IndexedMultiTFRecordDataset
from slideflow.util import Labels, detuple, log

from .img_utils import decode_image, whc_to_cwh
from .data_utils import process_labels, get_tfrecord_parser, load_index
from .augment import compose_augmentations

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

class WeightedInfiniteSampler(torch.utils.data.Sampler):
    """Sample from a dataset with weighted TFRecord probabilities.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        weights (list(float)): TFRecord weights for each sample in the dataset.
            If None, will sample from all TFRecords with equal probability.
            Defaults to None.

    """
    def __init__(self, dataset, weights=None):
        self.dataset = dataset
        if weights is None:
            weights = [0.5 for _ in range(len(dataset.tfrecords))]
        self.weights = weights / np.sum(weights)
        self.num_tfrecords = len(weights)

    def __iter__(self):
        while True:
            # Choose a random TFRecord.
            tfr_idx = np.random.choice(self.num_tfrecords, p=self.weights)
            # Find matching tiles in the sampled tfrecord
            all_tile_idx = (self.dataset.interleave_index[:, 0] == tfr_idx)
            if not len(all_tile_idx):
                # TFRecord is empty.
                continue
            # Return a random tile from the tfrecord
            yield np.random.choice(np.where(all_tile_idx)[0])

    def __len__(self):
        return self.num_samples


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
         - No "infinite" argument. Looping is handled by the dataloader.
         - No "prob_weights" argument. Sampling is handled by the dataloader.
         - Does not support dynamic reading from WSI ("from_wsi", "tile_um", "rois", "roi_method", and "pool" arguments).

        Args:
            tfrecords (list(str)): Path to tfrecord files to interleave.

        Keyword Args:
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
        self.rank = rank
        self.num_replicas = num_replicas
        self.parser = self.build_parser(tfrecord_parser)
        self.img_format = detect_tfrecord_format(self.tfrecords[0])[1]
        (self.labels,
         self.unique_labels,
         self.label_prob,
         self.num_outcomes) = process_labels(labels, onehot=onehot)
        if isinstance(self.labels, pd.DataFrame):
            self._prepare_tfrecord_subsample()

        # Automatically set shard to rank/num_replicas
        if self.rank == 0:
            log.info(
                f'Interleaving {len(self.tfrecords)} tfrecords: '
                f'num_replicas={self.num_replicas}'
            )

        # Clip tfrecords.
        if clip:
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
        msg += f"={self.num_tiles}, rank=({self.rank} of {self.num_replicas}))"
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
        elif isinstance(self.labels, pd.DataFrame):
            label = self.labels.loc[f'{slide}-{loc_x}-{loc_y}'].label
            label = torch.tensor(label)
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

    def _prepare_tfrecord_subsample(self):
        """Prepare custom TFRecord indices to only read tiles in the labels dataframe."""

        # Prepare TFRecord subsample if there are fewer tiles in the
        # tiles dataframe than there are in the TFRecords

        self.indices = []
        if self.rank == 0:
            log.debug("Subsampling TFRecords using tile-level labels...")

        n_tiles = 0
        orig_n_tiles = 0
        with mp.dummy.Pool(16) as pool:

            # Load the original (full) indices
            for index, tfr in zip(pool.imap(load_index, self.tfrecords), self.tfrecords):
                orig_n_tiles += len(index)
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
            diff = orig_n_tiles - n_tiles
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


    def build_parser(
        self,
        tfrecord_parser: Optional[Callable] = None
    ) -> Callable:
        """Build a parser function for TFRecords.

        The parser will be responsible for processing images and labels.

        """
        ftrs = ['image_raw', 'slide', 'loc_x', 'loc_y']
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
