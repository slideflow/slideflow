"""Submodule for tissue segmentation.

Tissue segmentation utilizes the ``segmentation-models-pytorch`` library, which
provides a number of pre-trained segmentation models across a variety of
architectures. The ``SegmentConfig`` class provides a convenient way to configure
a segmentation model, and the ``train`` function can be used to train a model
on a slideflow dataset.

"""

import os
import torch
import slideflow as sf
import numpy as np
import multiprocessing as mp
import pandas as pd
from typing import Optional, List
from functools import partial
from os.path import join, exists, isdir, dirname
from rich.progress import track
from scipy.ndimage import label

from slideflow.util import path_to_name
from slideflow.model import torch_utils

from .model import SegmentModel
from .data import (
    get_thumb_and_mask,
    BufferedMaskDataset,
    BufferedRandomCropDataset,
    RandomCropDataset,
    ThumbMaskDataset,
    TileMaskDataset
)
from .utils import topleft_pad, center_square_pad, outlines_list

# -----------------------------------------------------------------------------

def generate_rois(
    wsi: "sf.WSI",
    model: str
) -> List[np.ndarray]:
    """Generate ROIs for a single slide using a U-Net model.

    Args:
        wsi (sf.WSI): Slideflow WSI object.
        model (str): Path to '.pth' model file, as generated via :func:``slideflow.segment.train``.

    Returns:
        List[np.ndarray]: List of ROIs, where each ROI is a numpy array of
            shape (n, 2), where n is the number of vertices in the ROI.

    """
    # Remove any existing ROIs from the slide.
    wsi.rois = []

    # Load the model configuration.
    model, cfg = load_model_and_config(model)

    # Get thumbnail at given MPP.
    thumb = np.array(wsi.thumb(mpp=cfg.mpp))

    # Run tiled inference.
    preds = model.run_tiled_inference(thumb)

    # Threshold the predictions.
    labeled, n_rois = label(preds > 0)

    # Convert to ROIs.
    outlines = outlines_list(labeled)
    outlines = [o for o in outlines if o.shape[0]]

    # Scale the outlines.
    outlines = [o * (cfg.mpp / wsi.mpp) for o in outlines]

    # Load ROIs.
    for outline in outlines:
        try:
            wsi.load_roi_array(outline, process=False)
        except sf.errors.InvalidROIError:
            continue
    wsi.process_rois()

# -----------------------------------------------------------------------------

def export_thumbs_and_masks(
    dataset: "sf.Dataset",
    mpp: float,
    dest: str,
    *,
    overwrite: bool = False,
    skip_missing_roi: bool = True,
    mode: str = 'binary',
    labels: Optional[List[str]] = None,
):
    """Export thumbnails and segmentation masks (from ROIs) for a dataset.

    Args:
        dataset (sf.Dataset): Slideflow dataset.
        mpp (float): MPP to use for thumbnail generation.
        dest (str): Path to directory where thumbnails and masks will be saved.

    Keyword args:
        overwrite (bool): Whether to overwrite existing thumbnails/masks.
            Defaults to False.
        skip_missing_roi (bool): Whether to skip slides that do not have any
            ROIs. Defaults to True.
        mode (str): ROI label mode. Can be 'binary' or 'multiclass'.
            Defaults to 'binary'.
        labels (List[str]): List of ROI labels to include. If not provided,
            all ROI labels will be included.

    """
    # Parameter validation.
    if not isinstance(dataset, sf.Dataset):
        raise ValueError("dataset must be a slideflow Dataset.")
    if mode not in ('binary', 'multiclass', 'multilabel'):
        raise ValueError(
            "Unrecognized value for mode: {!r}. Expected "
            "'binary' or 'multiclass'.".format(mode)
        )

    if not exists(dest):
        os.makedirs(dest)

    # Write configuration.
    sf.util.write_json(dict(
        mpp=mpp,
        mode=mode,
        labels=labels,
    ), join(dest, 'mask_config.json'))

    rois = dataset.rois()
    slides_with_rois = [path_to_name(r) for r in rois]
    slides = [
        s for s in dataset.slide_paths()
        if (
            (path_to_name(s) in slides_with_rois)
            and not (exists(join(dest, f"{path_to_name(s)}.pt")) and not overwrite)
        )
    ]

    # Prepare ROI labels if this is a multiclass problem.
    if mode == 'binary':
        roi_labels = None
    else:
        _dts = dataset.filter({'slide': list(map(path_to_name, slides))})
        roi_labels = _dts.get_unique_roi_labels()
    if labels is not None:
        roi_labels = [l for l in roi_labels if l in labels]
        # Report an error if any labels were not found.
        if len(roi_labels) != len(labels):
            missing = set(labels) - set(roi_labels)
            raise ValueError("ROI labels not found: {}".format(missing))

    kw = dict(
        tile_px=299,
        tile_um=512,
        rois=rois,
        verbose=False
    )
    print("Generating data for {} slides (mode={!r}, overwrite={!r}).".format(
        len(slides), mode, overwrite)
    )
    if mode != 'binary':
        print("Labels: {}".format(roi_labels))
    if skip_missing_roi:
        print("Skipping slides with no ROIs.")
    ctx = mp.get_context('spawn')
    fn = partial(
        _export,
        kw=kw,
        mpp=mpp,
        dest=dest,
        roi_labels=roi_labels,
        skip_missing=skip_missing_roi
    )
    total = 0
    with ctx.Pool(sf.util.num_cpu()) as pool:
        for success in track(pool.imap(fn, slides),
                       description="Exporting...",
                       total=len(slides)):
            if success:
                total += 1
    print("Exported data for {} slides.".format(total))


def _export(s, kw, mpp, dest, roi_labels, skip_missing):
    try:
        wsi = sf.WSI(s, roi_filter_method=0.1, **kw)

    except Exception:
        return None
    else:
        try:
            out = get_thumb_and_mask(
                wsi,
                mpp=mpp,
                roi_labels=roi_labels,
                skip_missing=skip_missing
            )
        except Exception as e:
            sf.log.error("Error generating thumbnail/mask: {}".format(e))
        else:
            if out is not None:
                torch.save(out, join(dest, f"{wsi.name}.pt"))
                return True
    return None

# -----------------------------------------------------------------------------

class SegmentConfig:

    def __init__(
        self,
        arch: str = 'FPN',
        encoder_name: str = 'resnet34',
        *,
        size: int = 1024,
        in_channels: int = 3,
        out_classes: Optional[int] = None,
        train_batch_size: int = 8,
        val_batch_size: int = 16,
        epochs: int = 8,
        mpp: float = 20,
        lr: float = 1e-4,
        loss: str = 'dice',
        mode: str = 'binary',
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Configuration for a segmentation model.

        Args:
            arch (str): Model architecture. Defaults to 'FPN'.
            encoder_name (str): Encoder name. Defaults to 'resnet34'.

        Keyword args:
            size (int): Size of input images. Defaults to 1024.
            in_channels (int): Number of input channels. Defaults to 3.
            out_classes (int, optional): Number of output classes.
                If None, will attempt to auto-detect based the provided labels
                and loss mode. If labels are not provided, it defaults to 1.
            train_batch_size (int): Training batch size. Defaults to 8.
            val_batch_size (int): Validation batch size. Defaults to 16.
            epochs (int): Number of epochs to train for. Defaults to 8.
            mpp (float): MPP to use for training. Defaults to 10.
            loss (str): Loss function. Defaults to 'dice'.
            mode (str): Loss mode. Can be 'binary', 'multiclass', or 'multilabel'.
                Defaults to 'binary'.
            labels (List[str]): Names for ROI labels. Only used if mode
                is 'multiclass'. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model.

        """
        self.arch = arch
        self.encoder_name = encoder_name
        self.size = size
        self.in_channels = in_channels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.mpp = mpp
        self.lr = lr
        self.loss = loss
        self.mode = mode
        self.labels = labels
        self.kwargs = kwargs
        if out_classes is None:
            if mode == 'binary':
                self.out_classes = 1
            elif labels is None:
                self.out_classes = 1
            elif mode == 'multiclass':
                self.out_classes = len(labels) + 1
            elif mode == 'multilabel':
                self.out_classes = len(labels)
        else:
            self.out_classes = out_classes

    def __repr__(self) -> str:
        return (
            f"SegmentConfig(\n"
            f"    arch={self.arch!r},\n"
            f"    encoder_name={self.encoder_name!r},\n"
            f"    size={self.size!r},\n"
            f"    in_channels={self.in_channels!r},\n"
            f"    out_classes={self.out_classes!r},\n"
            f"    train_batch_size={self.train_batch_size!r},\n"
            f"    val_batch_size={self.val_batch_size!r},\n"
            f"    epochs={self.epochs!r},\n"
            f"    mpp={self.mpp!r},\n"
            f"    lr={self.lr!r},\n"
            f"    loss={self.loss!r},\n"
            f"    mode={self.mode!r},\n"
            f"    labels={self.labels!r},\n"
            f"    **{self.kwargs!r}\n"
            f")"
        )

    @classmethod
    def from_json(cls, path: str) -> "SegmentConfig":
        """Load a configuration from a JSON file.

        Args:
            path (str): Path to JSON file.

        Returns:
            SegmentConfig: SegmentConfig object.

        """
        data = sf.util.load_json(path)
        params = data['params'].copy()
        del data['params']
        return cls(**data, **params)

    def to_json(self, path: str) -> None:
        """Save the configuration to a JSON file.

        Args:
            path (str): Path to JSON file.

        """
        data = dict(
            params=dict(
                arch=self.arch,
                encoder_name=self.encoder_name,
                in_channels=self.in_channels,
                out_classes=self.out_classes,
                lr=self.lr,
                loss=self.loss,
                mode=self.mode,
                **self.kwargs
            ),
            mpp=self.mpp,
            size=self.size,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            epochs=self.epochs,
            labels=self.labels,
        )
        sf.util.write_json(data, path)

    def build_model(self) -> SegmentModel:
        """Build a segmentation model from this configuration."""
        return SegmentModel(
            self.arch,
            self.encoder_name,
            in_channels=self.in_channels,
            out_classes=self.out_classes,
            mpp=self.mpp,
            loss=self.loss,
            mode=self.mode,
            **self.kwargs
        )


def load_model_and_config(path: str):
    """Load a model and its configuration from a path.

    Args:
        path (str): Path to model file, or directory containing model file.

    Returns:
        Tuple[SegmentModel, SegmentConfig]: Tuple of model and configuration.

    """
    if not exists(path):
        raise ValueError(f"Model '{path}' does not exist.")
    if isdir(path):
        path = join(path, 'model.pth')
    if not path.endswith('pth'):
        raise ValueError(f"Model '{path}' is not a valid model path.")

    # Load the model configuration.
    cfg_path = join(dirname(path), 'segment_params.json')
    if not exists(cfg_path):
        raise ValueError(f"Model '{path}' does not contain a segment_params.json file.")
    cfg = SegmentConfig.from_json(cfg_path)

    # Build the model.
    model = cfg.build_model()

    # Load the weights.
    model.load_state_dict(torch.load(path, map_location=torch_utils.get_device()))
    model.eval()

    return model, cfg

# -----------------------------------------------------------------------------

def train(
    config: SegmentConfig,
    dataset: "sf.Dataset",
    val_dataset: Optional["sf.Dataset"] = None,
    data_source: Optional[str] = None,
    *,
    num_workers: int = 4,
    dest: Optional[str] = None,
    labels: Optional[List[str]] = None,
    skip_missing_roi: bool = True,
) -> SegmentModel:
    """Train a segmentation model.

    Args:
        config (SegmentConfig): Model configuration.
        dataset (sf.Dataset): Slideflow dataset.
        val_dataset (sf.Dataset): Slideflow dataset for validation.
        data_source (str): Path to directory containing thumbnails and masks.
            If not provided, thumbnails and masks will be generated from the
            dataset.

    Keyword args:
        num_workers (int): Number of workers to use for data loading.
            Defaults to 4.
        dest (str): Path to directory where model will be saved. If not
            provided, the model will not be saved.
        labels (List[str]): Names for ROI labels to include. Only used if mode
            is 'multiclass' and data_source is not provided. Defaults to None.

    Returns:
        SegmentModel: Trained model.

    """
    # Delayed import.
    import pytorch_lightning as pl  # type: ignore

    # Parameter validation.
    if not isinstance(dataset, sf.Dataset):
        raise ValueError("dataset must be a slideflow Dataset.")
    if val_dataset is not None and not isinstance(val_dataset, sf.Dataset):
        raise ValueError("val_dataset must be a slideflow Dataset.")
    if not isinstance(config, SegmentConfig):
        raise ValueError("config must be a SegmentConfig.")

    # Filter dataset to exclude slides with missing or empty ROIs.
    if skip_missing_roi:
        slides_with_rois = [sf.util.path_to_name(r) for r in dataset.rois() if len(pd.read_csv(r))]
        print("Using {} slides with non-empty ROIs.".format(len(slides_with_rois)))
        dataset = dataset.filter({'slide': slides_with_rois})

    # --- Validate the thumbnail/mask configuration. --------------------------
    if data_source:
        print("Training from pre-generated thumbnails and masks.")
        if not exists(data_source):
            raise ValueError(f"Data source '{data_source}' does not exist.")
        if not exists(join(data_source, 'mask_config.json')):
            sf.log.warning("Data source does not contain a mask_config.json file, "
                        "unable to perform validation.")
        else:
            # Validate the mask configuration.
            mask_config = sf.util.load_json(join(data_source, 'mask_config.json'))
            if mask_config['mpp'] != config.mpp:
                raise ValueError(
                    "Mismatch between mask_config.json mpp ({!r}) and "
                    "config mpp ({!r}).".format(
                        mask_config['mpp'],
                        config.mpp
                    )
                )
            if config.mode in ('multiclass', 'multilabel'):
                if config.labels is not None and mask_config['labels'] is not None:
                    if set(config.labels) != set(mask_config['labels']):
                        raise ValueError(
                            "Mismatch between mask_config.json labels ({!r}) and "
                            "config labels ({!r}).".format(
                                mask_config['labels'],
                                config.labels
                            )
                        )
                config.labels = mask_config['labels']
    else:
        print("Generating thumbnails and masks during training.")
        # Get unique ROI labels from the dataset.
        all_roi_labels = dataset.get_unique_roi_labels()
        if labels is not None:
            all_roi_labels = [l for l in all_roi_labels if l in labels]
            # Report an error if any labels were not found.
            if len(all_roi_labels) != len(labels):
                missing = set(labels) - set(all_roi_labels)
                raise ValueError("ROI labels not found: {}".format(missing))

        if config.mode in ('multiclass', 'multilabel'):
            if config.labels is not None:
                if not all_roi_labels:
                    raise ValueError(
                        "No ROI labels found in dataset. Ensure that the "
                        "slides in the dataset have labeled ROIs."
                    )
                if set(config.labels) != set(all_roi_labels):
                    raise ValueError(
                        "Mismatch between model configuration labels ({!r}) and "
                        "provided ROI labels ({!r}).".format(
                            config.labels,
                            all_roi_labels
                        )
                    )
            config.labels = all_roi_labels

    if config.mode == 'multiclass':
        if len(config.labels) != (config.out_classes-1):
            raise ValueError(
                "Mismatch between config labels ({!r}) and "
                "config out_classes ({!r}). Expected out_classes to be one greater "
                "than the number of labels when mode='multiclass' (one class is background).".format(
                    config.labels,
                    config.out_classes
                )
            )
    elif config.mode == 'multilabel':
        if len(config.labels) != config.out_classes:
            raise ValueError(
                "Mismatch between config labels ({!r}) and "
                "config out_classes ({!r}). Expected out_classes to be equal to "
                "the number of labels when mode='multilabel'.".format(
                    config.labels,
                    config.out_classes
                )
            )
    # -------------------------------------------------------------------------

    # Save training configuration.
    if dest:
        if not exists(dest):
            os.makedirs(dest)
        config.to_json(join(dest, 'segment_params.json'))

    # Build datasets.
    if data_source is None:
        dts_kw = dict(
            mpp=config.mpp,
            size=config.size,
            mode=config.mode,
            roi_labels=all_roi_labels,
        )
        train_ds = RandomCropDataset(dataset, **dts_kw)
        if val_dataset is not None:
            val_ds = RandomCropDataset(val_dataset, **dts_kw)
        else:
            val_ds = None
    else:
        dts_kw = dict(source=data_source, size=config.size, mode=config.mode)
        train_ds = BufferedRandomCropDataset(dataset, **dts_kw)
        if val_dataset is not None:
            val_ds = BufferedRandomCropDataset(val_dataset, **dts_kw)
        else:
            val_ds = None

    # Build dataloaders.
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True
    )
    if val_ds is not None:
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True
        )
    else:
        val_dl = None

    # Build model.
    model = config.build_model()

    # Build trainer.
    trainer = pl.Trainer(max_epochs=config.epochs)

    # Train.
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    # Save model.
    if dest:
        torch.save(model.state_dict(), join(dest, 'model.pth'))
        print("Saved model to {}".format(join(dest, 'model.pth')))

    return model

# -----------------------------------------------------------------------------