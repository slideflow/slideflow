import torch
import slideflow as sf
import numpy as np

from os.path import join, exists
from cellpose.utils import outlines_list
from cellpose.transforms import make_tiles, average_tiles
from scipy.ndimage import label

from .model import SegmentModel
from .utils import topleft_pad

# -----------------------------------------------------------------------------

def generate_rois(
    wsi: "sf.WSI",
    model: str,
):
    """Generate ROIs for a slide using a U-Net model.
    """

    # Remove any existing ROIs.
    wsi.rois = []

    # Load the model configuration.
    if not exists(model):
        raise ValueError(f"Model '{model}' does not exist.")
    if model.endswith('zip'):
        cfg_path = model.replace('.zip', '.json')
        cfg = sf.util.load_json(cfg_path)
        if cfg['weights'] != model:
            sf.log.warning("Model weights in config file ({}) does not match model path ({}).".format(
                cfg['weights'], model
            ))
        cfg['weights'] = model
    elif model.endswith('json'):
        cfg = sf.util.load_json(model)
    else:
        raise ValueError(f"Model '{model}' is not a valid model path.")
    
    # Load the model.
    model = SegmentModel("FPN", "resnet34", in_channels=3, out_classes=1)

    # Load the weights.
    model.load_state_dict(torch.load(cfg['weights']))
    model.eval()

    # Get thumbnail at given MPP.
    thumb = wsi.thumb(mpp=cfg['mpp'])

    # Pad to at least the target size.
    img = np.array(thumb)
    if img.shape[-1] == 4:
        img = img[..., :3]
    orig_dims = img.shape
    img = topleft_pad(img, 1024).transpose(2, 0, 1)

    # Tile the thumbnail.
    tiles, ysub, xsub, ly, lx = make_tiles(img, 1024)
    batched_tiles = tiles.reshape(-1, 3, 1024, 1024)

    # Generate UNet predictions.
    with torch.no_grad():
        tile_preds = model(torch.from_numpy(batched_tiles))

    # Merge predictions across the tiles.
    tiled_preds = average_tiles(tile_preds.numpy(), ysub, xsub, ly, lx)[0]
    
    # Crop predictions to the original size.
    tiled_preds = tiled_preds[:orig_dims[0], :orig_dims[1]]

    # Threshold the predictions.
    labeled, n_rois = label(tiled_preds > 0)

    # Convert to ROIs.
    outlines = outlines_list(labeled)
    outlines = [o for o in outlines if o.shape[0]]

    # Scale the outlines.
    outlines = [o * (cfg['mpp'] / wsi.mpp) for o in outlines]

    # Load ROIs.
    for outline in outlines:
        wsi.load_roi_array(outline, process=False)
    wsi.process_rois()
