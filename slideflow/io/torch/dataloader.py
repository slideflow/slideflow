"""PyTorch dataloaders."""

import math
import torch
import numpy as np
import slideflow as sf
from typing import Callable, Dict, List, Optional
from slideflow.util import Labels, log

from .iterable import InterleaveIterator


# -----------------------------------------------------------------------------

def worker_init_fn(worker_id) -> None:
    np.random.seed(np.random.get_state()[1][0])  # type: ignore


def interleave_dataloader(
    tfrecords: List[str],
    batch_size: Optional[int] = None,
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
    infinite: bool = True,
    prob_weights: Optional[Dict[str, float]] = None,
    num_tiles: Optional[int] = None,
    **kwargs
) -> torch.utils.data.DataLoader:

    """Prepares a PyTorch DataLoader with a new InterleaveIterator instance,
    interleaving tfrecords and processing labels and tiles, with support for
    scaling the dataset across GPUs and dataset workers.

    Args:
        tfrecords (list(str)): List of paths to TFRecord files.
        batch_size (int): Batch size.

    Keyword Args:
        img_size (int): Tile size in pixels. Required if from_wsi=True.
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
        incl_loc (bool, optional): Include location info (tile center
            coordinates). Returns samples in the form ``(returns ..., loc_x,
            loc_y)``. Defaults to False.
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
    torch.multiprocessing.set_sharing_strategy('file_system')

    if 'num_threads' not in kwargs and sf.util.num_cpu():
        n_cpu = sf.util.num_cpu() or 8
        kwargs['num_threads'] = int(math.ceil(n_cpu / max(num_workers, 1)))
        log.debug(f"Threads per worker={kwargs['num_threads']}")
    dataset = InterleaveIterator(
        tfrecords=tfrecords,
        use_labels=(labels is not None),
        num_replicas=num_replicas,
        labels=labels,
        from_wsi=from_wsi,
        infinite=infinite,
        prob_weights=prob_weights,
        num_tiles=num_tiles,
        **kwargs
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=replica_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **({'prefetch_factor': prefetch_factor} if num_workers else {})
    )
    dataloader.num_tiles = dataset.num_tiles
    dataloader.dataset.dataloader = dataloader  # type: ignore
    # Give a closing function to the DataLoader
    # to cleanup open files from iter()
    dataloader.close = dataset.close
    return dataloader
