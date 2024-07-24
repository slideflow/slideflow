import time
import rasterio
import cv2
import threading
import multiprocessing as mp
import numpy as np
import cellpose
import cellpose.models
import logging
import slideflow as sf
import zarr
import torch
import shapely.affinity as sa
from queue import Queue
from numcodecs import Blosc
from matplotlib.colors import to_rgb
from tqdm import tqdm
from typing import Tuple, Union, Callable, Optional, Iterable, TYPE_CHECKING, List
from functools import partial
from PIL import Image, ImageDraw
from cellpose.utils import outlines_list
from cellpose.models import Cellpose
from cellpose import transforms, plot, dynamics
from slideflow.slide.utils import draw_roi
from slideflow.util import batch_generator, log
from slideflow.model import torch_utils

from . import seg_utils

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID
    import shapely.geometry

# -----------------------------------------------------------------------------

class Segmentation:

    def __init__(
        self,
        masks: np.ndarray,
        *,
        slide: Optional[sf.WSI] = None,
        flows: Optional[np.ndarray] = None,
        styles: Optional[np.ndarray] = None,
        diams: Optional[np.ndarray] = None,
        wsi_dim: Optional[Tuple[int, int]] = None,
        wsi_offset: Optional[Tuple[int, int]] = None
    ):
        """Organizes a collection of cell segmentation masks for a slide.

        Args:
            masks (np.ndarray): Array of masks, dtype int32, where 0 represents
                non-segmented background, and each segmented mask is represented
                by unique increasing integers.

        Keyword args:
            slide (slideflow.WSI): If provided, ``Segmentation`` can coordinate
                extracting tiles at mask centroids. Defaults to None.
            flows (np.ndarray): Array of flows, dtype float32. Defaults to None.
            wsi_dim (tuple(int, int)): Size of ``masks`` in the slide
                pixel space (highest magnification). Used to align the mask
                array to a corresponding slide. Required for calculating
                centroids. Defaults to None.
            wsi_offset (tuple(int, int)): Top-left starting location for
                ``masks``, in slide pixel space (highest magnification).
                Used to align the mask array to a corresponding slide.
                Required for calculating centroids. Defaults to None.
            styles (np.ndarray): Array of styles, currently ignored.
            diams (np.ndarray): Array of diameters, currently ignored.

        """
        if not isinstance(masks, np.ndarray):
            raise ValueError("First argument (masks) must be a numpy array.")
        self.slide = slide
        self.masks = masks
        self.flows = flows
        self._outlines = None
        self._centroids = None
        self.wsi_dim = wsi_dim
        self.wsi_offset = wsi_offset

    @classmethod
    def load(cls, path) -> "Segmentation":
        """Alternate class initializer; load a saved Segmentation from *.zip.

        Args:
            path (str): Path to *.zip containing saved Segmentation, as created
                through :meth:`slideflow.cellseg.Segmentation.save`.

        """
        loaded = zarr.load(path)
        if 'masks' not in loaded:
            raise TypeError(f"Unable to load '{path}'; 'masks' index not found.")
        flows = None if 'flows' not in loaded else loaded['flows']
        obj = cls(slide=None, masks=loaded['masks'], flows=flows)
        obj.wsi_dim = loaded['wsi_dim']
        obj.wsi_offset = loaded['wsi_offset']
        if 'centroids' in loaded:
            obj._centroids = loaded['centroids']
        return obj

    @property
    def outlines(self) -> np.ndarray:
        """Calculate and return mask outlines as ``np.ndarray``."""
        if self._outlines is None:
            self.calculate_outlines()
        return self._outlines

    @property
    def wsi_ratio(self) -> Optional[float]:
        """Ratio of WSI base dimension to the mask shape.

        Returns `None` if ``wsi_dim`` was not set.
        """
        if self.wsi_dim is not None:
            return self.wsi_dim[1] / self.masks.shape[0]
        else:
            return None

    def apply_rois(
        self,
        scale: float,
        annpolys: List["shapely.geometry.Polygon"]
    )  -> None:
        """Apply regions of interest (ROIs), excluding masks outside ROIs.

        Args:
            scale (float): ROI scale (roi size / WSI base dimension size).
            annpolys (list(``shapely.geometry.Polygon``)): List of ROI
                polygons, as available in ``slideflow.WSI.rois``.

        """
        if self.wsi_ratio is not None and len(annpolys):
            roi_seg_scale = scale / self.wsi_ratio
            scaled_polys = [
                sa.scale(
                    poly,
                    xfact=roi_seg_scale,
                    yfact=roi_seg_scale,
                    origin=(0, 0)
                ) for poly in annpolys
            ]
            roi_seg_mask = rasterio.features.rasterize(
                scaled_polys,
                out_shape=self.masks.shape,
                all_touched=False
            ).astype(bool)
            self.masks *= roi_seg_mask
            self.calculate_centroids(force=True)
        elif self.wsi_ratio is None:
            log.warning("Unable to apply ROIs; WSI dimensions not set.")
            return
        else:
            # No ROIs to apply
            return

    def centroids(self, wsi_dim: bool = False) -> np.ndarray:
        """Calculate and return mask centroids.

        Args:
            wsi_dim (bool): Convert centroids from mask space to WSI space.
                Requires that ``wsi_dim`` was provided during initialization.

        Returns:
            A ``np.ndarray`` with shape ``(2, num_masks)``.

        """
        if self._centroids is None:
            self.calculate_centroids()
        if wsi_dim:
            if self.wsi_dim is None:
                raise ValueError("Unable to calculate wsi_dim for centroids - "
                                 "wsi_dim is not set.")
            ratio = self.wsi_dim[1] / self.masks.shape[0]
            return ((self._centroids * ratio)[:, ::-1] + self.wsi_offset).astype(np.int32)
        else:
            return self._centroids

    def _draw_centroid(self, img, color='green'):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for c in self.centroids():
            x, y = np.int32(c[1]), np.int32(c[0])
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        return np.asarray(pil_img)

    def calculate_centroids(self, force: bool = False) -> None:
        """Calculate centroids.

        Centroid values are buffered into ``Segmentation._centroids`` to
        reduce unnecessary recalculations.

        Args:
            force (bool): Recalculate centroids, even if calculated before.

        """
        if self._centroids is not None and not force:
            return
        mask_s = seg_utils.sparse_mask(self.masks)
        self._centroids = seg_utils.get_sparse_centroid(self.masks, mask_s)

    def calculate_outlines(self, force: bool = False) -> None:
        """Calculate mask outlines.

        Mask outlines are buffered into ``Segmentation._outlines`` to
        reduce unnecessary recalculations.

        Args:
            force (bool): Recalculate outlines, even if calculated before.

        """
        if self._outlines is not None and not force:
            return
        self._outlines = outlines_list(self.masks)

    def centroid_to_image(self, color: str = 'green') -> np.ndarray:
        """Render an image with the location of all centroids as a point.

        Args:
            color (str): Centroid color. Defaults to 'green'.

        """
        img = np.zeros((self.masks.shape[0], self.masks.shape[1], 3), dtype=np.uint8)
        return self._draw_centroid(img, color=color)

    def extract_centroids(
        self,
        slide: str,
        tile_px: int = 128,
    ) -> Callable:
        """Return a generator which extracts tiles from a slide at mask centroids.

        Args:
            slide (str): Path to a slide.
            tile_px (int): Height/width of tile to extract at centroids.
                Defaults to 128.

        Returns:
            A generator which yields a numpy array, with shape
                ``(tile_px, tile_px, 3)``, at each mask centroid.
        """
        reader = sf.slide.wsi_reader(slide)
        factor = reader.dimensions[1] / self.masks.shape[0]

        def generator():
            for c in self._centroids:
                cf = c * factor + self.wsi_offset
                yield reader.read_from_pyramid(
                    (cf[1]-(tile_px/2), cf[0]-(tile_px/2)),
                    (tile_px, tile_px),
                    (tile_px, tile_px),
                    convert='numpy',
                    flatten=True
                )

        return generator

    def mask_to_image(self, centroid=False, color='cyan', centroid_color='green'):
        """Render an image of all masks.

        Masks are rendered on a black background.

        Args:
            centroid (bool): Include centroids as points on the image.
                Defaults to False.
            color (str): Color of the masks. Defaults to 'cyan'.
            centroid_color (str): Color of centroid points. Defaults to 'green'.

        Returns:
            np.ndarray
        """
        if isinstance(color, str):
            color = [int(c * 255) for c in to_rgb(color)]
        else:
            assert len(color) == 3
        img = np.zeros((self.masks.shape[0], self.masks.shape[1], 3), dtype=np.uint8)
        img[self.masks > 0] = color
        if centroid:
            return self._draw_centroid(img, color=centroid_color)
        else:
            return img

    def outline_to_image(self, centroid=False, color='red', centroid_color='green'):
        """Render an image with the outlines of all masks.

        Args:
            centroid (bool): Include centroids as points on the image.
                Defaults to False.
            color (str): Color of the mask outlines. Defaults to 'red'.
            centroid_color (str): Color of centroid points. Defaults to 'green'.

        Returns:
            np.ndarray
        """
        empty = np.zeros((self.masks.shape[0], self.masks.shape[1], 3), dtype=np.uint8)
        img = draw_roi(empty, self.outlines, color=color)
        if centroid:
            return self._draw_centroid(img, color=centroid_color)
        else:
            return img

    def save(
        self,
        filename: str,
        centroids: bool = True,
        flows: bool = True
    ) -> None:
        """Save segmentation masks and metadata to \*.zip.

        A :class:`slideflow.cellseg.Segmentation` object can be loaded from
        this file with ``.load()``.

        Args:
            filename (str): Destination filename (ends with \*.zip)
            centroids (bool): Save centroid locations.
            flows (bool): Save flows.

        """
        if not filename.endswith('zip'):
            filename += '.zip'
        save_dict = dict(
            masks=self.masks,
            compressor=Blosc(
                cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE
            )
        )
        if centroids:
            self.calculate_centroids()
        if self._centroids is not None and centroids:
            save_dict['centroids'] = self._centroids
        if self.flows is not None and flows:
            save_dict['flows'] = self.flows
        if self.wsi_dim is not None:
            save_dict['wsi_dim'] = self.wsi_dim
        if self.wsi_offset is not None:
            save_dict['wsi_offset'] = self.wsi_offset
        seg_utils.save_zarr_compressed(filename, **save_dict)

# -----------------------------------------------------------------------------


def follow_flows(dP_and_cellprob, cp_thresh, gpus=(0,), **kwargs):
    dP, cellprob = dP_and_cellprob
    if gpus is not None:
        _id = mp.current_process()._identity
        proc = 0 if not len(_id) else _id[0]
        kwargs['device'] = torch.device(f'cuda:{gpus[proc % len(gpus)]}')
    if np.any(cellprob > cp_thresh):
        return dynamics.follow_flows(
            dP * (cellprob > cp_thresh) / 5.,
            use_gpu=(gpus is not None),
            **kwargs
        )
    else:
        return (None, None)


def remove_bad_flow(mask_and_dP, flow_threshold, gpus=(0,), **kwargs):
    mask, dP = mask_and_dP
    if gpus is not None:
        _id = mp.current_process()._identity
        proc = 0 if not len(_id) else _id[0]
        kwargs['device'] = torch.device(f'cuda:{gpus[proc % len(gpus)]}')
    if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        mask = dynamics.remove_bad_flow_masks(
            mask,
            dP,
            threshold=flow_threshold,
            use_gpu=(gpus is not None),
            **kwargs
        )
    return mask


def resize_and_clean_mask(mask, target_size=None):
    # Resizing
    recast = mask.max() >= 2**16-1
    if target_size:
        if recast:
            mask = mask.astype(np.float32)
        else:
            mask = mask.astype(np.uint16)
        mask = cv2.resize(
            mask,
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint32)
    elif not recast:
        mask = mask.astype(np.uint16)
    mask = dynamics.utils.fill_holes_and_remove_small_masks(mask, min_size=15)
    if mask.dtype == np.uint32 and mask.max() == 65535:
        log.warn(f'more than 65535 masks in image, masks returned as np.uint32')
    return mask


def get_empty_mask(shape):
    mask = np.zeros(shape, np.uint16)
    p = np.zeros((len(shape), *shape), np.uint16)
    return mask, p


def normalize_img(X):
    X = X.float()
    i99 = torch.quantile(X, 0.99)
    i1 = torch.quantile(X, 0.01)
    return (X - i1) / (i99 - i1)


def process_image(img, nchan):
    return transforms.convert_image(
        img,
        channels=[[0, 0]],
        channel_axis=None,
        z_axis=None,
        do_3D=False,
        normalize=False,
        invert=False,
        nchan=nchan)


def process_batch(img_batch):
    # Ensure Ly and Lx are divisible by 4
    assert not (img_batch.shape[1] % 16 or img_batch.shape[2] % 16)

    # Normalize and permute axes.
    img_batch = normalize_img(img_batch)
    img_batch = torch.permute(img_batch, (0, 3, 1, 2))
    return img_batch


def get_masks(args, cp_thresh):
    p, inds, cellprob = args
    if inds is None:
        mask, p = get_empty_mask(cellprob.shape)
    else:
        mask = dynamics.get_masks(p, iscell=(cellprob > cp_thresh))
    return mask, p


def tile_processor(slide, q, batch_size, nchan):
    tiles = batch_generator(
        slide.torch(
            incl_loc='grid',
            num_threads=4,
            to_tensor=False,
            grayspace_fraction=1,
            lazy_iter=True
        ),
        batch_size
    )
    for tile_dict in tiles:
        imgs = [t['image_raw'] for t in tile_dict]
        imgs = np.array([process_image(img, nchan) for img in imgs])
        c = [(t['loc_x'], t['loc_y']) for t in tile_dict]
        q.put((imgs, c))
    q.put(None)


def segment_slide(
    slide: Union[sf.WSI, str],
    model: Union["cellpose.models.Cellpose", str] = 'cyto2',
    *,
    diam_um: Optional[float] = None,
    diam_mean: Optional[int] = None,
    window_size: Optional[int] = None,
    downscale: Optional[float] = None,
    batch_size: int = 8,
    gpus: Optional[Union[int, Iterable[int]]] = (0,),
    spawn_workers: bool = True,
    pb: Optional["Progress"] = None,
    pb_tasks: Optional[List["TaskID"]] = None,
    show_progress: bool = True,
    save_flow: bool = True,
    cp_thresh: float = 0.0,
    flow_threshold: float = 0.4,
    interp: bool = True,
    tile: bool = True,
    verbose: bool = True,
    device: Optional[str] = None,
) -> Segmentation:
    """Segment cells in a whole-slide image, returning masks and centroids.

    Args:
        slide (str, :class:`slideflow.WSI`): Whole-slide image. May be a path
            (str) or WSI object (`slideflow.WSI`).

    Keyword arguments:
        model (str, :class:`cellpose.models.Cellpose`): Cellpose model to use
            for cell segmentation. May be any valid cellpose model. Defaults
            to 'cyto2'.
        diam_um (float, optional): Cell diameter to detect, in microns.
            Determines tile extraction microns-per-pixel resolution to match
            the given pixel diameter specified by `diam_mean`. Not used if
            `slide` is a `sf.WSI` object.
        diam_mean (int, optional): Cell diameter to detect, in pixels (without
            image resizing). If None, uses Cellpose defaults (17 for the
            'nuclei' model, 30 for all others).
        window_size (int): Window size, in pixels, at which to segment cells.
            Not used if slide is a `sf.WSI` object.
        downscale (float): Factor by which to downscale generated masks after
            calculation. Defaults to None (keep masks at original size).
        batch_size (int): Batch size for cell segmentation. Defaults to 8.
        gpus (int, list(int)): GPUs to use for cell segmentation.
            Defaults to 0 (first GPU).
        spawn_workers (bool): Enable spawn-based multiprocessing. Increases
            cell segmentation speed at the cost of higher memory utilization.
        pb (:class:`rich.progress.Progress`, optional): Progress bar instance.
            Used for external progress bar tracking. Defaults to None.
        pb_tasks (list(:class:`rich.progress.TaskID`)): Progress bar tasks.
            Used for external progress bar tracking. Defaults to None.
        show_progress (bool): Show a tqdm progress bar. Defaults to True.
        save_flow (bool): Save flow values for the whole-slide image.
            Increases memory utilization. Defaults to True.
        cp_thresh (float): Cell probability threshold. All pixels with value
            above threshold kept for masks, decrease to find more and larger
            masks. Defaults to 0.
        flow_threshold (float): Flow error threshold (all cells with errors
            below threshold are kept). Defaults to 0.4.
        interp (bool): Interpolate during 2D dynamics. Defaults to True.
        tile (bool): Tiles image to decrease GPU/CPU memory usage.
            Defaults to True.
        verbose (bool): Verbose log output at the INFO level. Defaults to True.

    Returns:
        :class:`slideflow.cellseg.Segmentation`
    """

    # Quiet the logger to suppress warnings of empty masks
    logging.getLogger('cellpose').setLevel(40)
    if diam_mean is None:
        diam_mean = 30 if model != 'nuclei' else 17

    # Initial validation checks. ----------------------------------------------
    if isinstance(slide, str):
        assert diam_um is not None, "Must supply diam_um if slide is a path to a slide"
        assert window_size is not None, "Must supply window_size if slide is a path to a slide"
        tile_um = int(window_size * (diam_um / diam_mean))
        slide = sf.WSI(slide, tile_px=window_size, tile_um=tile_um, verbose=False)
    elif window_size is not None or diam_um is not None:
        raise ValueError("Invalid argument: cannot provide window_size or diam_um "
                         "when slide is a sf.WSI object")
    else:
        window_size = slide.tile_px
        diam_um = diam_mean * (slide.tile_um/slide.tile_px)
    if window_size % 16:
        raise ValueError("Window size (tile_px) must be a multiple of 16.")
    if downscale is None:
        target_size = window_size
    else:
        target_size = int(window_size / downscale)
    if slide.stride_div != 1:
        log.warn("Whole-slide cell segmentation not configured for strides "
                 f"other than 1 (got: {slide.stride_div}).")

    # Set up model and parameters. --------------------------------------------
    start_time = time.time()
    device = torch_utils.get_device(device)
    if device.type == 'cpu':
        # Run from CPU if CUDA is not available
        model = Cellpose(gpu=False, device=device)
        gpus = None
        log.info("No GPU detected - running from CPU")
    else:
        model = Cellpose(gpu=True, device=device)
    cp = model.cp
    cp.batch_size = batch_size
    cp.net.load_model(cp.pretrained_model[0], cpu=(not cp.gpu))  # Modify to accept different models
    cp.net.eval()
    rescale = 1  # No rescaling, as we are manually setting diameter = diam_mean
    mask_dim = (slide.stride * (slide.shape[0]-1) + slide.tile_px,
                slide.stride * (slide.shape[1]-1) + slide.tile_px)
    all_masks = np.zeros((slide.shape[1] * target_size,
                          slide.shape[0] * target_size),
                         dtype=np.uint32)
    if save_flow:
        all_flows = np.zeros((slide.shape[1] * target_size,
                              slide.shape[0] * target_size, 3),
                             dtype=np.uint8)

    log_fn = log.info if verbose else log.debug
    log_fn("=== Segmentation parameters ===")
    log_fn(f"Diameter (px):     {diam_mean}")
    log_fn(f"Diameter (um):     {diam_um}")
    log_fn(f"Window size:       {window_size}")
    log_fn(f"Target size:       {target_size}")
    log_fn(f"Perform tiled:     {tile}")
    log_fn(f"Slide dimensions:  {slide.dimensions}")
    log_fn(f"Slide shape:       {slide.shape}")
    log_fn(f"Slide stride (px): {slide.stride}")
    log_fn(f"Est. tiles:        {slide.estimated_num_tiles}")
    log_fn(f"Save flow:         {save_flow}")
    log_fn(f"Mask dimensions:   {mask_dim}")
    log_fn(f"Mask size:         {all_masks.shape}")
    log_fn("===============================")

    # Processes and pools. ----------------------------------------------------
    tile_q = mp.Queue(4)
    y_q = Queue(2)
    ctx = mp.get_context('spawn')
    fork_pool = mp.Pool(
        batch_size,
        initializer=sf.util.set_ignore_sigint
    )
    if spawn_workers:
        spawn_pool = ctx.Pool(
            4,
            initializer=sf.util.set_ignore_sigint
        )
    else:
        spawn_pool = mp.dummy.Pool(4)
    proc_fn = mp.Process if sf.slide_backend() != 'libvips' else threading.Thread
    tile_process = proc_fn(
        target=tile_processor,
        args=(slide, tile_q, batch_size, cp.nchan)
    )
    tile_process.start()

    def net_runner():
        while True:
            item = tile_q.get()
            if item is None:
                y_q.put(None)
                break
            imgs, c = item
            torch_batch = cp._to_device(imgs)
            torch_batch = process_batch(torch_batch)
            if tile:
                y, style = cp._run_tiled(
                    torch_batch.cpu().numpy(),
                    augment=False,
                    bsize=224,
                    return_conv=False
                )
            else:
                y, style = cp.network(torch_batch)
            y_q.put((y, style, c))

    runner = threading.Thread(target=net_runner)
    runner.start()

    # Main loop. --------------------------------------------------------------
    running_max = 0
    if show_progress:
        tqdm_pb = tqdm(total=slide.estimated_num_tiles)
    while True:
        item = y_q.get()
        if item is None:
            break
        y, style, c = item

        # Initial preparation
        #style /= (style**2).sum()**0.5
        y = np.transpose(y, (0,2,3,1))
        cellprob = y[:, :, :, 2].astype(np.float32)
        dP = y[:, :, :, :2].transpose((3,0,1,2))
        del y, style
        #styles = style.squeeze()

        # Calculate flows
        batch_p, batch_ind = zip(*spawn_pool.map(
            partial(follow_flows,
                    niter=(1 / rescale * 200),
                    interp=interp,
                    cp_thresh=cp_thresh,
                    gpus=gpus),
            zip([dP[:, i] for i in range(len(c))], cellprob)
        ))

        # Calculate masks
        batch_masks, batch_p = zip(*fork_pool.map(
            partial(get_masks, cp_thresh=cp_thresh),
            zip(batch_p, batch_ind, cellprob)))

        # Remove bad flow
        batch_masks = spawn_pool.map(
            partial(remove_bad_flow, flow_threshold=flow_threshold, gpus=gpus),
            zip(batch_masks, [dP[:, i] for i in range(len(c))]))

        # Resize masks and clean (remove small masks/holes)
        batch_masks = fork_pool.map(
            partial(resize_and_clean_mask,
                    target_size=(None if target_size == window_size
                                      else target_size)),
            batch_masks)

        dP = dP.squeeze()
        cellprob = cellprob.squeeze()
        #p = np.stack(batch_p, axis=0)
        #flows = [plot.dx_to_circ(dP), dP, cellprob, p]

        for i in range(len(c)):
            x, y = c[i][0], c[i][1]
            img_masks = batch_masks[i].astype(np.uint32)
            max_in_mask = img_masks.max()
            img_masks[np.nonzero(img_masks)] += running_max
            running_max += max_in_mask
            all_masks[y * target_size: (y+1)*target_size,
                      x * target_size: (x+1)*target_size] = img_masks
            if save_flow:
                flow_plot = plot.dx_to_circ(dP[:, i])
                if target_size != window_size:
                    flow_plot = cv2.resize(flow_plot, (target_size, target_size))
                all_flows[y * target_size: (y+1)*target_size,
                          x * target_size: (x+1)*target_size, :] = flow_plot

        # Final cleanup
        del dP, cellprob

        # Update progress bars
        if show_progress:
            tqdm_pb.update(batch_size)
        if pb is not None and pb_tasks:
            for task in pb_tasks:
                pb.advance(task, batch_size)

    # Close pools/processes and log time.
    spawn_pool.close()
    spawn_pool.join()
    fork_pool.close()
    fork_pool.join()
    runner.join()
    tile_process.join()
    ttime = time.time() - start_time
    log.info(f"Segmented {running_max} cells for {slide.name} ({ttime:.0f} s)")

    # Calculate WSI dimensions and return final segmentation.
    wsi_dim = (slide.shape[0] * slide.full_extract_px,
               slide.shape[1] * slide.full_extract_px)
    wsi_offset = (0, 0)

    return Segmentation(
        slide=slide,
        masks=all_masks,
        flows=None if not save_flow else all_flows,
        wsi_dim=wsi_dim,
        wsi_offset=wsi_offset)
