
import os
import torch
import slideflow as sf
import numpy as np
import multiprocessing as mp
import tempfile

from typing import Optional
from functools import partial
from os.path import join, exists, isdir, dirname
from rich.progress import track
from cellpose.utils import outlines_list
from scipy.ndimage import label

from slideflow.util import path_to_name

from .model import SegmentModel
from .data import (
    get_thumb_and_mask, 
    BufferedMaskDataset, 
    BufferedRandomCropDataset, 
    TileMaskDataset
)
from .utils import topleft_pad, center_square_pad

# -----------------------------------------------------------------------------

def generate_rois(
    wsi: "sf.WSI",
    model: str,
):
    """Generate ROIs for a single slide using a U-Net model."""

    # Remove any existing ROIs.
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
        wsi.load_roi_array(outline, process=False)
    wsi.process_rois()

# -----------------------------------------------------------------------------

def export_thumbs_and_masks(
    dataset: "sf.Dataset", 
    mpp: float, 
    dest: str, 
    *,
    overwrite: bool = False
):
    """Export thumbnails and segmentation masks (from ROIs) for a dataset."""
    if not exists(dest):
        os.makedirs(dest)
    rois = dataset.rois()
    slides_with_rois = [path_to_name(r) for r in rois]
    slides = [
        s for s in dataset.slide_paths()
        if (
            (path_to_name(s) in slides_with_rois)
            and not (exists(join(dest, f"{path_to_name(s)}.pt")) and not overwrite)
        )
    ]
    kw = dict(
        tile_px=299,
        tile_um=302,
        rois=rois,
        verbose=False
    )
    print("Generating data for {} slides.".format(len(slides)))
    ctx = mp.get_context('spawn')
    with ctx.Pool(sf.util.num_cpu()) as pool:
        for wsi in track(pool.imap(partial(_export, kw=kw, mpp=mpp, dest=dest), slides), description="Exporting...", total=len(slides)):
            pass


def _export(s, kw, mpp, dest):
    try:
        wsi = sf.WSI(s, roi_filter_method=0.1, **kw)

    except Exception:
        return None
    else:
        try:
            out = get_thumb_and_mask(wsi, mpp=mpp)
        except Exception as e:
            sf.log.error("Error generating thumbnail/mask: {}".format(e))
            return None
        else:
            torch.save(out, join(dest, f"{wsi.name}.pt"))

# -----------------------------------------------------------------------------

class SegmentConfig:

    def __init__(
        self,
        arch: str = 'FPN',
        encoder_name: str = 'resnet34',
        *,
        size: int = 1024,
        in_channels: int = 3,
        out_classes: int = 1,
        train_batch_size: int = 8,
        val_batch_size: int = 16,
        epochs: int = 8,
        mpp: float = 10,
        **kwargs
    ):
        self.arch = arch
        self.encoder_name = encoder_name
        self.size = size
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.mpp = mpp
        self.kwargs = kwargs

    @classmethod
    def from_json(cls, path: str):
        data = sf.util.load_json(path)
        params = data['params'].copy()
        del data['params']
        return cls(**data, **params)
    
    def to_json(self, path: str):
        data = dict(
            params=dict(
                arch=self.arch,
                encoder_name=self.encoder_name,
                in_channels=self.in_channels,
                out_classes=self.out_classes,
                **self.kwargs
            ),
            mpp=self.mpp,
            size=self.size,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            epochs=self.epochs,
        )
        sf.util.write_json(data, path)

    def build_model(self):
        return SegmentModel(
            self.arch, 
            self.encoder_name, 
            in_channels=self.in_channels, 
            out_classes=self.out_classes,
            mpp=self.mpp,
            **self.kwargs
        )
    

def load_model_and_config(path: str):
    if not exists(path):
        raise ValueError(f"Model '{path}' does not exist.")
    if isdir(path):
        path = join(path, 'model.pth')
    if not path.endswith('pth'):
        raise ValueError(f"Model '{path}' is not a valid model path.")
    
    # Load the model configuration.
    cfg_path = join(dirname(path), 'segment_params.json')
    cfg = SegmentConfig.from_json(cfg_path)

    # Build the model.
    model = cfg.build_model()

    # Load the weights.
    model.load_state_dict(torch.load(path))
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
) -> SegmentModel: 
    """Train a segmentation model."""

    import pytorch_lightning as pl

    # Parameter validation.
    if not isinstance(dataset, sf.Dataset):
        raise ValueError("dataset must be a slideflow Dataset.")
    if val_dataset is not None and not isinstance(val_dataset, sf.Dataset):
        raise ValueError("val_dataset must be a slideflow Dataset.")
    if not isinstance(config, SegmentConfig):
        raise ValueError("config must be a SegmentConfig.")
    
    # Generate thumbnails and masks.
    if data_source is None:
        data_dest = join(tempfile.gettempdir(), 'segmentation_masks')
        print("Generating thumbnails and masks (saved to {}).".format(data_dest))
        export_thumbs_and_masks(dataset, mpp=config.mpp, dest=data_dest)
        if val_dataset is not None:
            export_thumbs_and_masks(val_dataset, mpp=config.mpp, dest=data_dest)

    # Save training configuration.
    if dest:
        if not exists(dest):
            os.makedirs(dest)
        config.to_json(join(dest, 'segment_params.json'))

    # Build datasets.
    dts_kw = dict(source=data_source, size=config.size)
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
        num_workers=num_workers
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