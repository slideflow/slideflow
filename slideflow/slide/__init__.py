'''This module includes tools to convolutionally section whole slide images
into tiles. These tessellated tiles can be exported as PNG or JPG as raw
images or stored in the binary format TFRecords, with or without augmentation.'''

from __future__ import absolute_import, division, print_function

import csv
import json
import multiprocessing as mp
import os
import random
import time
import warnings
import cv2
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry as sg
import shapely.affinity as sa
import shapely.validation as sv
import skimage
import skimage.filters
from PIL import Image, ImageDraw
from rich.progress import Progress
from skimage import img_as_ubyte
from slideflow import errors
from functools import partial
from os.path import exists, join
from types import SimpleNamespace
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    TYPE_CHECKING)

import slideflow as sf
import slideflow.slide.qc
from slideflow.util import SUPPORTED_FORMATS  # noqa F401
from slideflow.util import log, path_to_name  # noqa F401
from .report import ExtractionPDF  # noqa F401
from .report import ExtractionReport, SlideReport
from .utils import *
from .backends import tile_worker, wsi_reader


warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 100000000000

# -----------------------------------------------------------------------

def _update_kw_with_defaults(kwargs) -> Dict:
    """Updates a set of keyword arguments with default extraction values.
    for whitepsace/grayspace filtering.
    """
    if kwargs['whitespace_fraction'] is None:
        kwargs['whitespace_fraction'] = DEFAULT_WHITESPACE_FRACTION
    if kwargs['whitespace_threshold'] is None:
        kwargs['whitespace_threshold'] = DEFAULT_WHITESPACE_THRESHOLD
    if kwargs['grayspace_fraction'] is None:
        kwargs['grayspace_fraction'] = DEFAULT_GRAYSPACE_FRACTION
    if kwargs['grayspace_threshold'] is None:
        kwargs['grayspace_threshold'] = DEFAULT_GRAYSPACE_THRESHOLD
    if kwargs['img_format'] is None:
        kwargs['img_format'] = 'jpg'
    return kwargs


def _polyArea(x: List[float], y: List[float]) -> float:
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def _convert_img_to_format(image: np.ndarray, img_format: str) -> str:
    if img_format.lower() == 'png':
        return cv2.imencode(
            '.png',
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )[1].tobytes()
    elif img_format.lower() in ('jpg', 'jpeg'):
        return cv2.imencode(
            '.jpg',
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )[1].tostring()
    else:
        raise ValueError(f"Unknown image format {img_format}")


def log_extraction_params(**kwargs) -> None:
    '''Logs tile extraction parameters.'''

    if 'whitespace_fraction' not in kwargs:
        ws_f = DEFAULT_WHITESPACE_FRACTION
    else:
        ws_f = kwargs['whitespace_fraction']
    if 'whitespace_threshold' not in kwargs:
        ws_t = DEFAULT_WHITESPACE_THRESHOLD
    else:
        ws_t = kwargs['whitespace_threshold']
    if 'grayspace_fraction' not in kwargs:
        gs_f = DEFAULT_GRAYSPACE_FRACTION
    else:
        gs_f = kwargs['grayspace_fraction']
    if 'grayspace_threshold' not in kwargs:
        gs_t = DEFAULT_GRAYSPACE_THRESHOLD
    else:
        gs_t = kwargs['grayspace_threshold']

    if 'normalizer' in kwargs:
        log.info(f'Extracting tiles using [magenta]{kwargs["normalizer"]}[/] '
                 'normalization')
    if ws_f < 1:
        log.info('Filtering tiles by whitespace fraction')
        excl = f'(exclude if >={ws_f*100:.0f}% whitespace)'
        log.debug(f'Whitespace defined as RGB avg > {ws_t} {excl}')
    if gs_f < 1:
        log.info('Filtering tiles by grayspace fraction')
        excl = f'(exclude if >={gs_f*100:.0f}% grayspace)'
        log.debug(f'Grayspace defined as HSV avg < {gs_t} {excl}')


def predict(
    slide: str,
    model: str,
    *,
    stride_div: int = 1,
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate a whole-slide prediction from a saved model.

    Args:
        slide (str): Path to slide.
        model (str): Path to saved model trained in Slideflow.

    Keyword args:
        stride_div (int, optional): Divisor for stride when convoluting
                across slide. Defaults to 1.
        roi_dir (str, optional): Directory in which slide ROI is contained.
            Defaults to None.
        rois (list, optional): List of paths to slide ROIs. Alternative to
            providing roi_dir. Defaults to None.
        roi_method (str): Either 'inside', 'outside', 'auto', or 'ignore'.
            Determines how ROIs are used to extract tiles.
            If 'inside' or 'outside', will extract tiles in/out of an ROI,
            and raise errors.MissingROIError if an ROI is not available.
            If 'auto', will extract tiles inside an ROI if available,
            and across the whole-slide if no ROI is found.
            If 'ignore', will extract tiles across the whole-slide
            regardless of whether an ROI is available.
            Defaults to 'auto'.
        batch_size (int, optional): Batch size for calculating predictions.
            Defaults to 32.
        num_threads (int, optional): Number of tile worker threads. Cannot
            supply both ``num_threads`` (uses thread pool) and
            ``num_processes`` (uses multiprocessing pool). Defaults to
            CPU core count.
        num_processes (int, optional): Number of child processes to spawn
            for multiprocessing pool. Defaults to None (does not use
            multiprocessing).
        enable_downsample (bool, optional): Enable the use of downsampled
            slide image layers. Defaults to True.
        img_format (str, optional): Image format (png, jpg) to use when
            extracting tiles from slide. Must match the image format
            the model was trained on. If 'auto', will use the format
            logged in the model params.json. Defaults to 'auto'.
        generator_kwargs (dict, optional): Keyword arguments passed to
            the :meth:`slideflow.WSI.build_generator()`.
        device (torch.device, optional): PyTorch device. Defaults to
            initializing a new CUDA device.

    Returns:
        np.ndarray: Predictions for each outcome, with shape = (num_classes, )

        np.ndarray, optional: Uncertainty for each outcome, if the model was
        trained with uncertainty, with shape = (num_classes,)

    """
    from slideflow import Heatmap
    log.info("Calculating whole-slide prediction...")
    heatmap = Heatmap(slide, model, generate=True, stride_div=stride_div, **kwargs)
    preds = heatmap.predictions.reshape(-1, heatmap.predictions.shape[-1])
    preds = np.ma.masked_where(preds == -99, preds).mean(axis=0).filled()
    if heatmap.uncertainty is not None:
        unc = heatmap.uncertainty.reshape(-1, heatmap.uncertainty.shape[-1])
        unc = np.ma.masked_where(unc == -99, unc).mean(axis=0).filled()
        return preds, unc
    else:
        return preds


class ROI:
    '''Object container for ROI annotations.'''

    def __init__(self, name: str, coordinates: np.ndarray = None) -> None:
        self.name = name
        if coordinates is None:
            self.coordinates = []  # type: List[Tuple[int, int]]
        else:
            self.coordinates = coordinates

    def __repr__(self):
        return f"<ROI (coords={len(self.coordinates)})>"

    def add_coord(self, coord: Tuple[int, int]) -> None:
        self.coordinates.append(coord)

    def scaled_area(self, scale: float) -> np.ndarray:
        return np.multiply(self.coordinates, 1/scale)

    def print_coord(self) -> None:
        for c in self.coordinates:
            print(c)

    def add_shape(self, shape) -> None:
        for point in shape:
            self.add_coord(point)


class _BaseLoader:
    '''Loads an SVS slide and makes preparations for tile extraction.

    Should not be used directly; this class must be inherited and extended
    by either WSI or TMA child classes.
    '''

    def __init__(
        self,
        path: str,
        tile_px: int,
        tile_um: Union[int, str],
        stride_div: int,
        enable_downsample: bool = True,
        pb: Optional[Progress] = None,
        mpp: Optional[float] = None,
        **reader_kwargs
    ) -> None:

        self.pb = pb
        self.name = path_to_name(path)
        self.shortname = sf.util._shortname(self.name)
        self.tile_px = tile_px
        self.enable_downsample = enable_downsample
        self.thumb_image = None  # type: Optional[Image.Image]
        self.stride_div = stride_div
        self.path = path
        self.qc_masks = []
        self.rois = []  # type: List
        self.qc_mpp = None  # type: Optional[float]
        self.blur_burden = None  # type: Optional[float]
        self.roi_scale = 1  # type: float
        self.roi_method = None  # type: Optional[str]
        self.annPolys = []  # type: ignore
        self.filetype = sf.util.path_to_ext(path)
        self.__slide = None
        self._mpp_override = mpp
        self._reader_kwargs = reader_kwargs

        # Initiate supported slide reader
        if not os.path.exists(path):
            raise errors.SlideNotFoundError(f"Could not find slide {path}.")
        if self.filetype.lower() not in sf.util.SUPPORTED_FORMATS:
            raise errors.SlideLoadError(
                f"{self.name}: unsupported filetype '{self.filetype}'"
            )

        # Collect basic slide information
        try:
            self.mpp = float(self.slide.mpp)
        except Exception as e:
            raise errors.SlideLoadError(
                f"Slide [green]{self.name}[/] missing MPP ({OPS_MPP_X})"
            )

        # Calculate downsample by magnification
        if isinstance(tile_um, str):
            sf.util.assert_is_mag(tile_um)
            _mag_lvl = 10 / (np.array(self.slide.level_downsamples) * self.mpp)
            mag_levels = _mag_lvl.tolist()
            closest_mag = min(
                mag_levels,
                key=lambda x: abs(x - sf.util.to_mag(tile_um))  # type: ignore
            )
            if abs(closest_mag - sf.util.to_mag(tile_um)) > 2:
                raise errors.SlideLoadError(
                    f"{self.name}: Could not find magnification level "
                    f"matching {tile_um} (closest: {closest_mag:.1f})"
                )
            ds_level = mag_levels.index(closest_mag)
            if not enable_downsample and ds_level != 0:
                raise ValueError(f"Unable to use magnification {tile_um} with "
                                 "enable_downsample=False")
            self.downsample_factor = self.slide.level_downsamples[ds_level]
            self.extract_px = tile_px
            self.full_extract_px = int(self.downsample_factor * tile_px)
            self.tile_um = int(self.downsample_factor * self.mpp * tile_px)
            log.debug(f"Using magnification {closest_mag:.1f}x (level="
                      f"{ds_level}, tile_um={self.tile_um})")

        # Calculate downsample level by tile micron size
        else:
            assert isinstance(tile_um, int)
            self.tile_um = tile_um
            self.full_extract_px = int(tile_um / self.mpp)
            ds = self.full_extract_px / tile_px
            if enable_downsample:
                ds_level = self.slide.best_level_for_downsample(ds)
            else:
                ds_level = 0
            self.downsample_factor = self.slide.level_downsamples[ds_level]
            self.extract_px = self.full_extract_px // self.downsample_factor

        # Calculate filter dimensions (low magnification for filtering out
        # white background and performing edge detection)
        self.filter_dimensions = self.slide.level_dimensions[-1]
        self.filter_magnification = (self.filter_dimensions[0]
                                    / self.dimensions[0])
        self.filter_px = int(self.full_extract_px * self.filter_magnification)

        # Calculate shape and stride
        self.downsample_level = ds_level
        self.downsample_dimensions = self.slide.level_dimensions[ds_level]
        self.stride = int(self.extract_px // stride_div)
        self.full_stride = int(self.full_extract_px // stride_div)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if '__slide' in state:
            state['__slide'] = None
        if '_BaseLoader__slide' in state:
            state['_BaseLoader__slide'] = None
        if 'pb' in state:
            state['pb'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Dimensions of highest-magnification level (width, height)"""
        return self.slide.dimensions

    @property
    def levels(self) -> Dict:
        """List of dict, with metadata for each level.

        Each dict has the keys 'dimensions', 'downsample', 'height', and 'weight'.

        - **'dimensions'**: (height, width) of the level.
        - **'downsample'**: Downsample level, where higher numbers indicate
            lower magnification and the highest magnification is 1.
        - **`height'**: Height of the level.
        - **`height'**: Width of the level.

        """
        return self.slide.levels

    @property
    def level_dimensions(self) -> List[List[int]]:
        """List of list, with dimensions for each slide level."""
        return self.slide.level_dimensions

    @property
    def level_downsamples(self) -> List[float]:
        """Downsample of each level (starts at 1, increases with lower mag)."""
        return self.slide.level_downsamples

    @property
    def level_mpp(self) -> List[float]:
        """Microns-per-pixel (MPP) for each level."""
        return [d * self.mpp for d in self.level_downsamples]

    @property
    def properties(self) -> Dict:
        """Dictionary of metadata loaded from the slide."""
        return self.slide.properties

    @property
    def vendor(self) -> Optional[str]:
        """Slide scanner vendor, if available."""
        if OPS_VENDOR in self.slide.properties:
            return self.slide.properties[OPS_VENDOR]
        else:
            return None

    @property
    def slide(self) -> Union["_JPGReader", "_SlideReader"]:
        if self.__slide is not None:
            return self.__slide
        try:
            self.__slide = wsi_reader(
                self.path,
                self._mpp_override,
                **self._reader_kwargs)
            return self.__slide  # type: ignore
        except errors.SlideMissingMPPError:
            raise
        except Exception as e:
            raise errors.SlideLoadError(
                f"Error loading slide {self.shortname}: {e}"
            )

    @property
    def qc_mask(self) -> Optional[np.ndarray]:
        """Returns union of all QC masks."""
        if not self.qc_masks:
            return None
        elif len(self.qc_masks) == 1:
            return self.qc_masks[0]
        else:
            _, smallest = min((m.shape[0], idx)
                               for (idx, m) in enumerate(self.qc_masks))
            shape = self.qc_masks[smallest].shape
            mask = skimage.transform.resize(self.qc_masks[0], shape).astype(bool)
            for _next in self.qc_masks[1:]:
                _next = skimage.transform.resize(_next, shape).astype(bool)
                mask = np.logical_or(mask, _next)
            return mask

    def _build_coord(self):
        raise NotImplementedError

    def mpp_to_dim(self, mpp: float) -> Tuple[int, int]:
        width = int((self.mpp * self.dimensions[0]) / mpp)
        height = int((self.mpp * self.dimensions[1]) / mpp)
        return (width, height)

    def dim_to_mpp(self, dimensions: Tuple[float, float]) -> float:
        return (self.dimensions[0] * self.mpp) / dimensions[0]

    def remove_qc(self) -> None:
        self._build_coord()
        self.qc_masks = []
        log.debug(f'QC removed from slide {self.shortname}')

    def qc(
        self,
        method: Union[str, Callable, List[Callable]],
        *,
        blur_radius: int = 3,
        blur_threshold: float = 0.02,
        filter_threshold: float = 0.6,
        blur_mpp: Optional[float] = None
    ) -> Optional[Image.Image]:
        """Applies quality control to a slide, performing filtering based on
        a whole-slide image thumbnail.

        'blur' method filters out blurry or out-of-focus slide sections.
        'otsu' method filters out background based on automatic saturation
        thresholding in the HSV colorspace.
        'both' applies both methods of filtering.

        Args:
            method (str, Callable, list(Callable)): Quality control method(s).
                If a string, may be 'blur', 'otsu', or 'both'.
                If a callable (or list of callables), each must accept a sf.WSI
                object and return a np.ndarray (dtype=np.bool).
            blur_radius (int, optional): Blur radius. Only used if method is
                'blur' or 'both'.
            blur_threshold (float, optional): Blur threshold. Only used if
                method is 'blur' or 'both.'
            filter_threshold (float): Percent of a tile detected as
                background that will trigger a tile to be discarded.
                Defaults to 0.6.
            blur_mpp (float, optional): Size of WSI thumbnail on which to
                perform blur QC, in microns-per-pixel. Defaults to 4 times the
                tile extraction MPP (e.g. for a tile_px/tile_um combination
                at 10X effective magnification, where tile_px=tile_um, the
                default blur_mpp would be 4, or effective magnification 2.5x).
                Only used if method is 'blur' or 'both'.

        Returns:
            Image: Image of applied QC mask.
        """

        # Prepare known QC methods - 'blur', 'otsu', and 'both'.
        if not isinstance(method, list):
            method = [method]           # type: ignore
        if 'both' in method:
            idx = method.index('both')  # type: ignore
            method.remove('both')       # type: ignore
            method.insert(idx, 'otsu')  # type: ignore
            # Blur should be performed before Otsu's thresholding
            method.insert(idx, 'blur')  # type: ignore
        if 'blur' in method:
            idx = method.index('blur')  # type: ignore
            method.remove('blur')       # type: ignore
            method.insert(idx, sf.slide.qc.Gaussian(mpp=blur_mpp,
                                                    sigma=blur_radius,
                                                    threshold=blur_threshold))
        if 'otsu' in method:
            idx = method.index('otsu')  # type: ignore
            method.remove('otsu')       # type: ignore
            method.insert(idx, sf.slide.qc.Otsu())

        starttime = time.time()
        img = None
        log.debug(f"Applying QC: {method}")
        for qc in method:
            if isinstance(method, str):
                raise errors.QCError(f"Unknown QC method {method}")
            mask = qc(self)
            if mask is not None:
                img = self.apply_qc_mask(mask, filter_threshold=filter_threshold)
        dur = f'(time: {time.time()-starttime:.2f}s)'
        log.debug(f'QC ({method}) complete for slide {self.shortname} {dur}')
        return img

    def apply_qc_mask(
        self,
        mask: np.ndarray,
        filter_threshold: float = 0.6,
    ) -> "Image":
        """Apply custom slide-level QC by filtering grid coordinates.

        Args:
            mask (np.ndarray): Boolean QC mask.
            filter_threshold (float): Percent of a tile detected as
                background that will trigger a tile to be discarded.
                Defaults to 0.6.

        Returns:
            Image: Image of applied QC mask.
        """
        assert isinstance(mask, np.ndarray)
        assert len(mask.shape) == 2
        assert mask.dtype == bool

        downsample = self.dimensions[0] / mask.shape[1]
        qc_ratio = 1 / downsample
        qc_width = int(np.round(self.full_extract_px * qc_ratio))
        for i, (x, y, xi, yi) in enumerate(self.coord):  # type: ignore
            qc_x = int(np.round(x * qc_ratio))
            qc_y = int(np.round(y * qc_ratio))
            submask = mask[qc_y:(qc_y+qc_width), qc_x:(qc_x+qc_width)]
            if np.mean(submask) > filter_threshold:
                self.grid[xi, yi] = 0

        self.qc_masks.append(mask)
        self.qc_mpp = self.mpp * downsample
        self.estimated_num_tiles = int(self.grid.sum())
        return Image.fromarray(img_as_ubyte(self.qc_mask))

    def square_thumb(
        self,
        width: int = 512,
        use_associated_image: bool = True,
        **kwargs
    ) -> Image.Image:
        '''Returns a square thumbnail of the slide, with black bar borders.

        Args:
            width (int): Width/height of thumbnail in pixels.

        Returns:
            PIL image
        '''
        thumb = self.thumb(
            width=width,
            use_associated_image=use_associated_image,
            **kwargs)
        height = int(width / (thumb.width / thumb.height))
        thumb = thumb.resize((width, height))
        square_thumb = Image.new("RGB", (width, width))
        square_thumb.paste(thumb, (0, int((width-height)/2)))
        return square_thumb

    def thumb(
        self,
        mpp: Optional[float] = None,
        width: Optional[int] = None,
        coords: Optional[List[int]] = None,
        rect_linewidth: int = 2,
        rect_color: str = 'black',
        use_associated_image: bool = False,
        low_res: bool = False
    ) -> Image.Image:
        """Returns PIL thumbnail of the slide.

        Args:
            mpp (float, optional): Microns-per-pixel, used to determine
                thumbnail size.
            width (int, optional): Alternatively, goal thumbnail width
                may be supplied.
            coords (list(int), optional): List of tile extraction coordinates
                to show as rectangles on the thumbnail, in [(x_center,
                y_center), ...] format. Defaults to None.
            use_associated_image (bool): Use the associated thumbnail image
                in the slide, rather than reading from a pyramid layer.
            low_res (bool): Create thumbnail from the lowest-mangnification
                pyramid layer. Defaults to False.

        Returns:
            PIL image

        """
        # If no values provided, create thumbnail of width 1024
        if mpp is None and width is None:
            width = 1024
        if (mpp is not None and width is not None):
            raise ValueError(
                "Either mpp or width must be given, but not both"
                f" (got mpp={mpp}, width={width})"
            )

        # Calculate goal width/height according to specified microns-per-pixel
        if mpp:
            width = int((self.mpp * self.dimensions[0]) / mpp)
        # Otherwise, calculate approximate mpp based on provided width
        # (to generate proportional height)
        else:
            assert width is not None
            mpp = (self.mpp * self.dimensions[0]) / width
        # Calculate appropriate height
        height = int((self.mpp * self.dimensions[1]) / mpp)

        if use_associated_image:
            log.debug("Requesting thumbnail using associated image")
            thumb_kw = dict(associated='thumbnail')
        elif low_res:
            log.debug("Requesting thumbnail at level={}, width={}".format(
                self.slide.level_count-1, width
            ))
            thumb_kw = dict(level=self.slide.level_count-1, width=width)
        else:
            ds = self.dimensions[0] / width
            level = self.slide.best_level_for_downsample(ds)
            log.debug("Requesting thumbnail at level={}, width={}".format(
                level, width
            ))
            thumb_kw = dict(level=level, width=width)

        np_thumb = self.slide.thumbnail(**thumb_kw)
        image = Image.fromarray(np_thumb).resize((width, height))

        if coords:
            draw = ImageDraw.Draw(image)
            ratio = width / self.dimensions[0]
            wh = (self.full_extract_px * ratio) / 2
            for (x, y) in coords:  # type: ignore
                x, y = x * ratio, y * ratio  # type: ignore
                coords = (x-wh, y-wh, x+wh, y+wh)  # type: ignore
                draw.rectangle(coords, outline=rect_color, width=rect_linewidth)
            return image
        else:
            return image

    def build_generator(
        self,
        *,
        shuffle: bool = True,
        whitespace_fraction: float = None,
        whitespace_threshold: float = None,
        grayspace_fraction: float = None,
        grayspace_threshold: float = None,
        normalizer: str = None,
        normalizer_source: str = None,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
        img_format: str = 'numpy',
        full_core: bool = False,
        yolo: bool = False,
        draw_roi: bool = False,
        pool: Optional["mp.pool.Pool"] = None,
        dry_run: bool = False
    ) -> Optional[Callable]:
        """Build a tile generator."""
        lead_msg = f'Extracting {self.tile_um}um tiles'
        if self.extract_px != self.tile_px:
            resize_msg = f'(resizing {self.extract_px}px -> {self.tile_px}px)'
        else:
            resize_msg = f'({self.extract_px}px, not resizing)'
        stride_msg = f'stride: {int(self.stride)}px'
        log.debug(f"{self.shortname}: {lead_msg} {resize_msg}; {stride_msg}")
        if self.tile_px > self.extract_px:
            ups_msg = 'Tiles will be up-scaled with bilinear interpolation'
            ups_amnt = f'({self.extract_px}px -> {self.tile_px}px)'
            warn = f"[red]'!WARN!'[/]"
            log.warn(f"{self.shortname}: {warn} {ups_msg} {ups_amnt}")
        return None

    def extract_tiles(
        self,
        tfrecord_dir: Optional[str] = None,
        tiles_dir: Optional[str] = None,
        img_format: str = 'jpg',
        report: bool = True,
        **kwargs
    ) -> Optional[SlideReport]:
        """Extracts tiles from slide using the build_generator() method,
        saving tiles into a TFRecord file or as loose JPG tiles in a directory.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file
                (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images in a subdirectory
                 (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal
                storage in tfrecords. PNG (lossless) format recommended for
                fidelity, JPG (lossy) for efficiency. Defaults to 'jpg'.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not
                perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            normalizer (str, optional): Normalization to use on image tiles.
                Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                If None, will use slideflow.slide.norm_tile.jpg.
                Defaults to None.
            full_core (bool, optional): Extract an entire detected core, rather
                than subdividing into image tiles. Defaults to False.
            shuffle (bool): Shuffle images during extraction.
            num_threads (int): Number of threads to allocate to workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI
                annotations (.txt) in the tile directory. Requires that
                tiles_dir is set. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles.
                Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted,
                but do not export any images. Defaults to None.
            num_threads (int): If specified, will extract tiles with a
                ThreadPool using the specified number of threads. Cannot
                supply both `num_threads` and `num_processes`. Libvips is
                particularly slow with ThreadPools. Defaults to None in the
                Libvips backend, and the number of CPU cores when using cuCIM.
            num_processes (int): If specified, will extract tiles with a
                multiprocessing pool using the specified number of processes.
                Cannot supply both `num_threads` and `num_processes`.
                With the libvips backend, this defaults to half the number of
                CPU cores, and with cuCIM, this defaults to None.
        """
        if img_format not in ('png', 'jpg', 'jpeg'):
            raise ValueError(f"Invalid image format {img_format}")

        dry_run = kwargs['dry_run'] if 'dry_run' in kwargs else False

        # Make base directories
        if tfrecord_dir and not dry_run:
            if not exists(tfrecord_dir):
                os.makedirs(tfrecord_dir)
        if tiles_dir and not dry_run:
            tiles_dir = os.path.join(tiles_dir, self.name)
            if not os.path.exists(tiles_dir):
                os.makedirs(tiles_dir)

        # Log to keep track of when tiles have finished extracting
        # To be used in case tile extraction is interrupted, so the slide
        # can be flagged for re-extraction

        if (tfrecord_dir or tiles_dir) and not dry_run:
            unfinished_marker = join(
                (tfrecord_dir if tfrecord_dir else tiles_dir),  # type: ignore
                f'{self.name}.unfinished'
            )
            with open(unfinished_marker, 'w') as marker_file:
                marker_file.write(' ')
        if tfrecord_dir and not dry_run:
            writer = sf.io.TFRecordWriter(join(
                tfrecord_dir,
                self.name+".tfrecords"
            ))

        generator = self.build_generator(
            img_format=img_format,
            **kwargs
        )
        if not generator:
            if tfrecord_dir:
                os.remove(join(tfrecord_dir, self.name+".tfrecords"))
            return None

        sample_tiles = []  # type: List
        generator_iterator = generator()
        locations = []
        grid_locations = []
        ws_fractions = []
        gs_fractions = []
        num_wrote_to_tfr = 0
        slidename_bytes = bytes(self.name, 'utf-8')

        for index, tile_dict in enumerate(generator_iterator):
            location = tile_dict['loc']
            locations += [location]
            grid_locations += [tile_dict['grid']]
            if 'ws_fraction' in tile_dict:
                ws_fractions += [tile_dict['ws_fraction']]
            if 'gs_fraction' in tile_dict:
                gs_fractions += [tile_dict['gs_fraction']]

            if dry_run:
                continue

            image_string = tile_dict['image']
            if len(sample_tiles) < 10:
                sample_tiles += [image_string]
            elif (not tiles_dir and not tfrecord_dir) and not dry_run:
                break
            if tiles_dir:
                img_f = join(
                    tiles_dir,
                    f'{self.shortname}_{index}.{img_format}'
                )
                with open(img_f, 'wb') as outfile:
                    outfile.write(image_string)
                if 'yolo' in tile_dict and len(tile_dict['yolo']):
                    yolo_f = join(tiles_dir, f'{self.shortname}_{index}.txt')
                    with open(yolo_f, 'w') as outfile:
                        for ann in tile_dict['yolo']:
                            yolo_str_fmt = "0 {:.3f} {:.3f} {:.3f} {:.3f}\n"
                            outfile.write(yolo_str_fmt.format(
                                ann[0],
                                ann[1],
                                ann[2],
                                ann[3]
                            ))
            if tfrecord_dir:
                record = sf.io.serialized_record(
                    slidename_bytes,
                    image_string,
                    location[0],
                    location[1]
                )
                writer.write(record)
                num_wrote_to_tfr += 1
        if tfrecord_dir and not dry_run:
            writer.close()
            if not num_wrote_to_tfr:
                os.remove(join(tfrecord_dir, self.name+".tfrecords"))
                log.info(f'No tiles extracted for [green]{self.name}')
        if self.pb is None:
            generator_iterator.close()

        if (tfrecord_dir or tiles_dir) and not dry_run:
            try:
                os.remove(unfinished_marker)
            except OSError:
                log.error(f"Unable to mark slide {self.name} as complete")

        # Generate extraction report
        if report:
            log.debug("Generating slide report")
            loc_np = np.array(locations, dtype=np.int64)
            grid_np = np.array(grid_locations, dtype=np.int64)
            df_dict = {
                'loc_x': [] if not len(loc_np) else pd.Series(loc_np[:, 0], dtype=int),
                'loc_y': [] if not len(loc_np) else pd.Series(loc_np[:, 1], dtype=int),
                'grid_x': [] if not len(grid_np) else pd.Series(grid_np[:, 0], dtype=int),
                'grid_y': [] if not len(grid_np) else pd.Series(grid_np[:, 1], dtype=int)
            }
            if ws_fractions:
                df_dict.update({'ws_fraction': pd.Series(ws_fractions, dtype=float)})
            if gs_fractions:
                df_dict.update({'gs_fraction': pd.Series(gs_fractions, dtype=float)})
            report_data = dict(
                blur_burden=self.blur_burden,
                num_tiles=len(locations),
                qc_mask=self.qc_mask,
                locations=pd.DataFrame(df_dict),
                num_rois=(0 if self.roi_method == 'ignore' else len(self.rois)),
                tile_px=self.tile_px,
                tile_um=self.tile_um,
            )
            slide_report = SlideReport(
                sample_tiles,
                self.slide.path,
                data=report_data,
                thumb_coords=locations,
                tile_px=self.tile_px,
                tile_um=self.tile_um,
            )
            return slide_report
        else:
            log.debug("Skipping slide report")
            return None

    def preview(
        self,
        rois: bool = True,
        thumb_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Image.Image]:
        """Performs a dry run of tile extraction without saving any images,
        returning a PIL image of the slide thumbnail annotated with a grid of
        tiles that were marked for extraction.

        Args:
            rois (bool, optional): Draw ROI annotation(s) onto the image.
                Defaults to True.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered
                whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not
                perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            full_core (bool, optional): Extract an entire detected core, rather
                than subdividing into image tiles. Defaults to False.
            num_threads (int): Number of threads to allocate to workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI
                annotations (.txt) in the tile directory. Requires that
                tiles_dir is set. Defaults to False.
        """
        if 'show_progress' not in kwargs:
            kwargs['show_progress'] = (self.pb is None)
        generator = self.build_generator(
            dry_run=True,
            deterministic=False,
            **kwargs
        )
        if thumb_kwargs is None:
            thumb_kwargs = dict()
        if generator is None:
            return self.thumb(rois=rois, low_res=True, **thumb_kwargs)
        locations = []
        for tile_dict in generator():
            locations += [tile_dict['loc']]
        log.debug(f"Previewing with {len(locations)} extracted tile locations.")
        return self.thumb(
            coords=locations, rois=rois, low_res=True, **thumb_kwargs
        )


class WSI(_BaseLoader):
    '''Loads a slide and its annotated region of interest (ROI).'''

    def __init__(
        self,
        path: str,
        tile_px: int,
        tile_um: Union[int, str],
        stride_div: int = 1,
        enable_downsample: bool = True,
        roi_dir: Optional[str] = None,
        rois: Optional[List[str]] = None,
        roi_method: str = 'auto',
        roi_filter_method: Union[str, float] = 'center',
        randomize_origin: bool = False,
        pb: Optional[Progress] = None,
        verbose: bool = True,
        **kwargs
    ) -> None:
        """Loads slide and ROI(s).

        Args:
            path (str): Path to slide.
            tile_px (int): Size of tiles to extract, in pixels.
            tile_um (int or str): Size of tiles to extract, in microns (int) or
                magnification (str, e.g. "20x").
            stride_div (int, optional): Stride divisor for tile extraction
                (1 = no tile overlap; 2 = 50% overlap, etc). Defaults to 1.
            enable_downsample (bool, optional): Allow use of downsampled
                intermediate layers in the slide image pyramid, which greatly
                improves tile extraction speed. May result in artifacts for
                slides with incompletely generated intermediates pyramids.
                Defaults to True.
            roi_dir (str, optional): Directory in which to search for ROI CSV
                files. Defaults to None.
            rois (list(str)): Alternatively, a list of ROI paths can be
                explicitly provided. Defaults to None.
            roi_method (str): Either 'inside', 'outside', 'auto', or 'ignore'.
                Determines how ROIs are used to extract tiles.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and raise errors.MissingROIError if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.
            roi_filter_method (str or float): Method of filtering tiles with
                ROIs. Either 'center' or float (0-1). If 'center', tiles are
                filtered with ROIs based on the center of the tile. If float,
                tiles are filtered based on the proportion of the tile inside
                the ROI, and ``roi_filter_method`` is interpreted as a
                threshold. If the proportion of a tile inside the ROI is
                greater than this number, the tile is included. For example,
                if ``roi_filter_method=0.7``, a tile that is 80% inside of an
                ROI will be included, and a tile that is 50% inside of an ROI
                will be excluded. Defaults to 'center'.
            randomize_origin (bool, optional): Offset the starting grid by a
                random amount. Defaults to False.
            pb (:class:`Progress`, optional): Multiprocessing
                capable Progress instance; will update progress bar during
                tile extraction if provided.
            verbose (bool, optional): Controls verbosity of output. If False,
                suppresses warnings about slide skipping when ROIs are missing.
                Defaults to True.
            mpp (float, optional): Override the microns-per-pixel value for
                the slide. Defaults to None (auto-detects).
            ignore_missing_mpp (bool, optional): If a slide does not have
                microns-per-pixel (MPP) information stored in EXIF data
                (key 65326), set the MPP to a default value
                (``sf.slide.DEFAULG_JPG_MPP``). If False and MPP data is
                missing, raises ``sf.errors.SlideMissingMPPError``.
        """
        super().__init__(
            path=path,
            tile_px=tile_px,
            tile_um=tile_um,
            stride_div=stride_div,
            enable_downsample=enable_downsample,
            pb=pb,
            **kwargs
        )

        # Initialize calculated variables
        self.extracted_x_size = 0  # type: int
        self.extracted_y_size = 0  # type: int
        self.estimated_num_tiles = 0  # type: int
        self.annPolys = []  # type: List
        self.roi_scale = 10  # type: float
        self.roi_method = roi_method
        self.roi_filter_method = roi_filter_method
        self.randomize_origin = randomize_origin
        self.verbose = verbose
        self.segmentation = None
        self.grid = None

        if (not isinstance(roi_filter_method, (int, float))
           and roi_filter_method != 'center'):
            raise ValueError(
                "Unrecognized value for argument 'roi_filter_method': {} ."
                "Expected either float or 'center'.".format(roi_filter_method)
            )
        if (isinstance(roi_filter_method, (int, float))
           and (roi_filter_method < 0 or roi_filter_method > 1)):
            raise ValueError(
                "If 'roi_filter_method' is a float, it must be between 0-1."
            )

        if rois is not None and not isinstance(rois, (list, tuple)):
            rois = [rois]

        # Look in ROI directory if available
        if roi_dir and exists(join(roi_dir, self.name + ".csv")):
            self.load_csv_roi(join(roi_dir, self.name + ".csv"), process=False)

        # Else try loading ROI from same folder as slide
        elif exists(self.name + ".csv"):
            self.load_csv_roi(path_to_name(path) + ".csv", process=False)
        elif rois and self.name in [path_to_name(r) for r in rois]:
            matching_rois = []
            for rp in rois:
                rn = path_to_name(rp)
                if rn == self.name:
                    matching_rois += [rp]
            mr = matching_rois[0]
            if len(matching_rois) > 1:
                log.warning(
                    f"Multiple ROIs found for {self.name}; using {mr}"
                )
            self.load_csv_roi(mr, process=False)

        # Handle missing ROIs
        if (not len(self.rois)
           and roi_method != 'ignore'
           and not (rois or roi_dir)):
            # No ROIs found because the user did not provide rois or roi_dir,
            # but the roi_method is not set to 'ignore',
            # indicating that this may be user error.
            warn_msg = f"No ROIs provided for {self.name}"
            if verbose and not (rois is None and roi_dir is None):
                log.warning(warn_msg)
            else:
                log.debug(warn_msg)
        if not len(self.rois) and roi_method in ('inside', 'outside'):
            raise errors.MissingROIError(
                f"Slide [green]{self.name}[/] missing ROI."
            )
        elif not len(self.rois):
            info_msg = f"No ROI for {self.name}, using whole slide."
            if verbose and roi_method == 'auto':
                log.info(info_msg)
            else:
                log.debug(info_msg)
        elif len(self.rois) and roi_method == 'auto':
            log.debug(f"Slide {self.name}: extracting tiles from inside ROI.")
            self.roi_method = 'inside'

        # Build coordinate grid
        self.process_rois()

        mpp_roi_msg = f'{self.mpp} um/px | {len(self.rois)} ROI(s)'
        size_msg = f'Size: {self.dimensions[0]} x {self.dimensions[1]}'
        log.debug(f"{self.shortname}: Slide info: {mpp_roi_msg} | {size_msg}")
        grid_msg = f"{self.shortname}: Grid shape: {self.grid.shape} "
        grid_msg += f"| Tiles to extract: {self.estimated_num_tiles}"
        log.debug(grid_msg)

    def __repr__(self) -> str:
        base = "WSI(\n"
        base += "  path = {!r},\n".format(self.path)
        base += "  tile_px = {!r},\n".format(self.tile_px)
        base += "  tile_um = {!r},\n".format(self.tile_um)
        base += "  stride_div = {!r},\n".format(self.stride_div)
        base += "  enable_downsample = {!r},\n".format(self.enable_downsample)
        base += "  roi_method = {!r},\n".format(self.roi_method)
        base += ")"
        return base

    def __getitem__(self, index):
        # Verify indices are valid
        if (not isinstance(index, (tuple, list, np.ndarray))
           or not len(index) == 2):
            raise IndexError("Must supply exactly two indices: (x, y)")
        if not (index[0] < self.shape[0]):
            raise IndexError(
                "index {} is out of bounds for axis 0 with size {}".format(
                    index[0],
                    self.shape[0]
                )
            )
        if not (index[1] < self.shape[1]):
            raise IndexError(
                "index {} is out of bounds for axis 0 with size {}".format(
                    index[1],
                    self.shape[1]
                )
            )

        # Find the corresponding coordinate given the provided indices.
        coord_idx, = np.where((
            (self.coord[:, 2] == index[0])
            & (self.coord[:, 3] == index[1])
        ))
        if not len(coord_idx):
            return None
        assert len(coord_idx) == 1
        x, y, grid_x, grid_y = self.coord[coord_idx[0]]

        # Check if indices correspond to a tile that is filtered out,
        # either by ROI or QC. If so, return None.
        if not self.grid[grid_x, grid_y]:
            return None

        # Extract the numpy image at this grid location.
        image_dict = tile_worker(
            (x, y, grid_x, grid_y),
            SimpleNamespace(
                full_extract_px=self.full_extract_px,
                mpp_override=self._mpp_override,
                reader_kwargs=self._reader_kwargs,
                roi_scale=self.roi_scale,
                rois=self.rois,
                grid=self.grid,
                downsample_level=self.downsample_level,
                path=self.path,
                extract_px=self.extract_px,
                tile_px=self.tile_px,
                full_stride=self.full_stride,
                normalizer=None,
                whitespace_fraction=1,
                whitespace_threshold=1,
                grayspace_fraction=1,
                grayspace_threshold=1,
                img_format='numpy',
                yolo=False,
                draw_roi=False,
                dry_run=False,
                has_segmentation=False,
            )
        )
        return image_dict['image']

    def _build_coord(self) -> None:
        '''Set up coordinate grid.'''

        log.debug("Setting up coordinate grid.")

        # Calculate window sizes, strides, and coordinates for windows
        self.extracted_x_size = self.dimensions[0] - self.full_extract_px
        self.extracted_y_size = self.dimensions[1] - self.full_extract_px

        # Randomize origin, if desired
        if self.randomize_origin:
            start_x = random.randint(0, self.full_stride-1)
            start_y = random.randint(0, self.full_stride-1)
            log.info(f"Random origin: X: {start_x}, Y: {start_y}")
        else:
            start_x = start_y = 0

        # Coordinates must be in level 0 (full) format
        # for the read_region function
        self.coord = []  # type: Union[List, np.ndarray]
        y_range = np.arange(
            start_y,
            (self.dimensions[1]+1) - self.full_extract_px,
            self.full_stride
        )
        x_range = np.arange(
            start_x,
            (self.dimensions[0]+1) - self.full_extract_px,
            self.full_stride
        )
        self.grid = np.ones((len(x_range), len(y_range)), dtype=bool)

        # ROI filtering
        roi_by_center = (self.roi_filter_method == 'center')
        if self.has_rois():

            # Full extraction size and stride
            full_extract = self.tile_um / self.mpp
            stride = full_extract / self.stride_div

            # Coverage size of the extracted image tiles
            xtrim = int(stride * (self.grid.shape[0]))  # type: ignore
            ytrim = int(stride * (self.grid.shape[1]))  # type: ignore

            # Degree to which the ROIs will need to be scaled
            # to match the extracted image tile grid
            xfact = self.grid.shape[0] / (xtrim / self.roi_scale)
            yfact = self.grid.shape[1] / (ytrim / self.roi_scale)

            # Offset to align the ROI polygons with the image tile grid
            x_offset = - (full_extract/2 - stride/2)
            y_offset = - (full_extract/2 - stride/2)

            # Translate ROI polygons
            translated = [
                sa.translate(poly, x_offset/self.roi_scale, y_offset/self.roi_scale)
                for poly in self.annPolys
            ]

            # Set scale to 50 times greater than grid size
            # if filtering by float
            o = 1 if roi_by_center else 50

            # Scale ROI polygons
            scaled = [
                sa.scale(poly, xfact=xfact * o, yfact=yfact * o, origin=(0, 0))
                for poly in translated
            ]

            # Rasterize polygons to the size of the tile extraction grid
            self.roi_mask = rasterio.features.rasterize(
                scaled,
                out_shape=(self.grid.shape[1] * o, self.grid.shape[0] * o),
                all_touched=False).astype(bool)
        else:
            self.roi_mask = None

        for yi, y in enumerate(y_range):
            for xi, x in enumerate(x_range):
                y = int(y)
                x = int(x)
                self.coord.append([x, y, xi, yi])

                # ROI filtering
                if self.has_rois() and roi_by_center:
                    point_in_roi = self.roi_mask[yi, xi]
                    # If the extraction method is 'inside',
                    # skip the tile if it's not in an ROI
                    if (((self.roi_method == 'inside') and not point_in_roi)
                       or ((self.roi_method == 'outside') and point_in_roi)):
                        self.grid[xi, yi] = 0

        # If roi_filter_method is a float, then perform tile selection
        # based on what proportion of the tile is in an ROI,
        # rather than choosing a tile by centroid (roi_filter_method='center')
        if self.has_rois() and not roi_by_center:
            self.apply_qc_mask(
                ~self.roi_mask,
                filter_threshold=(1-self.roi_filter_method)
            )

        self.coord = np.array(self.coord)
        self.estimated_num_tiles = int(self.grid.sum())

    @property
    def shape(self):
        return self.grid.shape

    def apply_segmentation(self, segmentation):
        # Filter out masks outside of ROIs, if present.
        if self.has_rois():
            log.debug(f"Applying {len(self.annPolys)} ROIs to segmentation.")
            segmentation.apply_rois(self.roi_scale, self.annPolys)

        self.segmentation = segmentation
        if self.segmentation.slide is None:
            self.segmentation.slide = self
        centroids = segmentation.centroids(wsi_dim=True)
        self.seg_coord = np.concatenate(
            (centroids, np.expand_dims(np.arange(centroids.shape[0]), axis=-1)),
            axis=-1)
        nonzero = self.seg_coord[:, 0] > 0
        self.seg_coord[:, 0:2][nonzero] -= int(self.full_extract_px/2)
        self.estimated_num_tiles = centroids.shape[0]

    def get_tile_mask(self, index, sparse_mask):

        # Get the corresponding segmentation mask, reading from the sparse matrix
        seg = self.segmentation
        mask_idx = self.seg_coord[index][2] + 1  # sparse mask index starts at 1
        mask_y, mask_x = np.unravel_index(sparse_mask[mask_idx].data, seg.masks.shape)

        # This is the top-left coordinate, in WSI base dimension,
        # of the tile extraction window.
        wsi_tile_top_left = self.seg_coord[index][0:2]

        # Determine the mask array offset (top-left), in mask coordinate space.
        wsi_mask_x_offset = np.round(seg.wsi_offset[0] / seg.wsi_ratio).astype(np.int32)
        wsi_mask_y_offset = np.round(seg.wsi_offset[1] / seg.wsi_ratio).astype(np.int32)

        # Offset the mask to reflect WSI space (but still in mask coordinates).
        wsi_mask_x = mask_x + wsi_mask_x_offset
        wsi_mask_y = mask_y + wsi_mask_y_offset

        # Determine the tile window offset (top-left), in mask coordinate space.
        tile_offset_x_in_mask_space = np.round(wsi_tile_top_left[0] / seg.wsi_ratio).astype(np.int32)
        tile_offset_y_in_mask_space = np.round(wsi_tile_top_left[1] / seg.wsi_ratio).astype(np.int32)

        # Adjust the mask coordinate space, using the tile window offset as origin.
        tile_mask_x = (wsi_mask_x - tile_offset_x_in_mask_space)
        tile_mask_y = (wsi_mask_y - tile_offset_y_in_mask_space)

        # Calculate the size of the tile window, in mask coordinate space.
        mask_tile_size = int(self.full_extract_px / seg.wsi_ratio)

        # Clip the mask to the tile window view.
        tile_mask_x = tile_mask_x.clip(0, mask_tile_size-1)
        tile_mask_y = tile_mask_y.clip(0, mask_tile_size-1)

        # Convert mask coordinates (in sparse format) to 2D array.
        unsized = np.zeros((mask_tile_size, mask_tile_size), dtype=np.int32)
        unsized[tile_mask_y, tile_mask_x] = 1

        # Resize mask from mask coordinates to tile extraction WSI coordinates.
        return unsized

    def export_rois(self, dest: Optional[str] = None) -> str:
        """Export loaded ROIs to a given destination, in CSV format.

        Args:
            dest (str): Path to destination folder. If not provided, will
                export ROIs in the current folder. Defaults to None.

        Returns:
            None

        """
        labels, x, y = [], [], []
        for roi in self.rois:
            c = np.array(roi.coordinates)
            assert len(c.shape) == 2
            labels += [roi.name] * c.shape[0]
            x += list(c[:, 0])
            y += list(c[:, 1])
        df = pd.DataFrame({
            'roi_name': labels,
            'x_base': x,
            'y_base': y
        })
        if dest is None:
            dest = f'{self.name}.csv'
        df.to_csv(dest, index=False)
        log.info(f"{len(self.rois)} ROIs exported to {dest}")
        return dest

    def extract_tiles(
        self,
        tfrecord_dir: Optional[str] = None,
        tiles_dir: Optional[str] = None,
        img_format: str = 'jpg',
        report: bool = True,
        **kwargs: Any
    ) -> Optional[SlideReport]:
        """Extracts tiles from slide using the build_generator() method,
        saving tiles into a TFRecord file or as loose JPG tiles in a directory.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file
                (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images into a
                subdirectory (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal
                storage in tfrecords. PNG (lossless) format recommended for
                fidelity, JPG (lossy) for efficiency. Defaults to 'jpg'.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not
                perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            normalizer (str, optional): Normalization for image tiles.
                Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                If None, will use slideflow.slide.norm_tile.jpg.
                Defaults to None.
            full_core (bool, optional): Extract an entire detected core, rather
                than subdividing into image tiles. Defaults to False.
            shuffle (bool): Shuffle images during extraction.
            num_threads (int): Number of threads to allocate to workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI
                annotations (.txt) in the tile directory. Requires that
                tiles_dir is set. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles.
                Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted,
                but do not export any images. Defaults to None.
        """
        return super().extract_tiles(
            tfrecord_dir,
            tiles_dir,
            img_format,
            report,
            **kwargs
        )

    def extract_cells(
        self,
        tfrecord_dir: Optional[str] = None,
        tiles_dir: Optional[str] = None,
        img_format: str = 'jpg',
        report: bool = True,
        apply_masks: bool = True,
        **kwargs
    ) -> Optional[SlideReport]:
        """Extract tiles from cell segmentation centroids.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file
                (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images into a
                subdirectory (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal
                storage in tfrecords. PNG (lossless) format recommended for
                fidelity, JPG (lossy) for efficiency. Defaults to 'jpg'.
            report (bool): Generate and return PDF report of tile extraction.
            apply_masks (bool): Apply cell segmentation masks to the extracted
                tiles. Defaults to True.

        Keyword Args:
            **kwargs: All keyword arguments are passed to :meth:`WSI.extract_tiles()`.
        """
        if self.segmentation is None:
            raise ValueError(
                "Cannot build generator from segmentation centroids; "
                "segmentation not yet applied. Use WSI.apply_segmentation()."
            )
        return super().extract_tiles(
            tfrecord_dir,
            tiles_dir,
            img_format,
            report,
            apply_masks=apply_masks,
            from_centroids=True,
            **kwargs
        )

    def has_rois(self) -> bool:
        """Checks if the slide has loaded ROIs and they are not being ignored."""
        return (self.roi_method != 'ignore'
                and len(self.rois)
                and self.annPolys is not None)

    def build_generator(
        self,
        *,
        shuffle: bool = True,
        whitespace_fraction: float = None,
        whitespace_threshold: float = None,
        grayspace_fraction: float = None,
        grayspace_threshold: float = None,
        normalizer: str = None,
        normalizer_source: str = None,
        context_normalize: bool = False,
        num_threads: Optional[int] = None,
        num_processes: Optional[int] = None,
        show_progress: bool = False,
        img_format: str = 'numpy',
        full_core: bool = False,
        yolo: bool = False,
        draw_roi: bool = False,
        pool: Optional["mp.pool.Pool"] = None,
        dry_run: bool = False,
        lazy_iter: bool = False,
        shard: Optional[Tuple[int, int]] = None,
        max_tiles: Optional[int] = None,
        from_centroids: bool = False,
        apply_masks: bool = True,
        deterministic: bool = True
    ) -> Optional[Callable]:
        """Builds tile generator to extract tiles from this slide.

        Keyword args:
            shuffle (bool): Shuffle images during extraction.
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not
                perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            normalizer (str, optional): Normalization strategy to use on image
                tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                If None, will use slideflow.slide.norm_tile.jpg.
                Defaults to None.
            context_normalize (bool): If normalizing, use context from
                the rest of the slide when calculating stain matrix
                concentrations. Defaults to False (normalize each image tile
                as separate images).
            num_threads (int): If specified, will extract tiles with a
                ThreadPool using the specified number of threads. Cannot
                supply both `num_threads` and `num_processes`. Libvips is
                particularly slow with ThreadPools. Defaults to None in the
                Libvips backend, and the number of CPU cores when using cuCIM.
            num_processes (int): If specified, will extract tiles with a
                multiprocessing pool using the specified number of processes.
                Cannot supply both `num_threads` and `num_processes`.
                With the libvips backend, this defaults to half the number of
                CPU cores, and with cuCIM, this defaults to None.
            show_progress (bool, optional): Show a progress bar.
            img_format (str, optional): Image format. Either 'numpy', 'jpg',
                or 'png'. Defaults to 'numpy'.
            yolo (bool, optional): Include yolo-formatted tile-level ROI
                annotations in the return dictionary, under the key 'yolo'.
                Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles.
                Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted,
                but do not export any images. Defaults to None.
            max_tiles (int, optional): Only extract this many tiles per slide.
                Defaults to None.
            from_centroids (bool): Extract tiles from cell segmentation
                centroids, rather than in a grid-wise pattern. Requires that
                cell segmentation has already been applied with
                `WSI.apply_segmentation()`. Defaults to False.
            apply_masks (bool): Apply cell segmentation masks to tiles. Ignored
                if cell segmentation has been applied to the slide.
                Defaults to True.
            deterministic (bool): Return tile images in reproducible,
                deterministic order. May slightly decrease iteration time.
                Defaults to True.

        Returns:
            dict: Dict with keys 'image' (image data), 'yolo' (optional
            yolo-formatted annotations, (x_center, y_center,
            width, height)) and 'grid' ((x, y) slide or grid coordinates)

        """
        if (isinstance(num_threads, int)
           and isinstance(num_processes, int)
           and num_threads > 1
           and num_processes > 1):
            raise ValueError("num_threads and num_processes cannot both be "
                             "non-zero.")
        if (shard is not None
           and (not isinstance(shard, (tuple, list))
                or len(shard) != 2
                or any(not isinstance(s, int) for s in shard))):
            raise ValueError("If shard is provided, it must be a tuple of "
                             "two int (shard_idx, shard_count)")

        if from_centroids and self.segmentation is None:
            raise ValueError(
                "Cannot build generator from segmentation centroids; "
                "segmentation not yet applied. Use WSI.apply_segmentation()."
            )

        super().build_generator()
        if self.estimated_num_tiles == 0:
            log.warning(f"No tiles extracted for slide [green]{self.name}")
            return None

        # Set whitespace / grayspace fraction to defaults if not provided
        if whitespace_fraction is None:
            whitespace_fraction = DEFAULT_WHITESPACE_FRACTION
        if whitespace_threshold is None:
            whitespace_threshold = DEFAULT_WHITESPACE_THRESHOLD
        if grayspace_fraction is None:
            grayspace_fraction = DEFAULT_GRAYSPACE_FRACTION
        if grayspace_threshold is None:
            grayspace_threshold = DEFAULT_GRAYSPACE_THRESHOLD

        # Get information about highest level downsample, as we will filter
        # on that layer if downsampling is enabled
        if self.enable_downsample:
            downsamples = np.array(self.slide.level_downsamples)
            filter_lev = np.max(np.argwhere(downsamples < self.extract_px))
            filter_downsample_factor = self.slide.level_downsamples[filter_lev]
            lev_ds = self.slide.level_downsamples[self.downsample_level]
            filter_downsample_ratio = filter_downsample_factor // lev_ds
        else:
            filter_lev = self.downsample_level
            filter_downsample_ratio = 1

        # Prepare stain normalization
        if normalizer and not isinstance(normalizer, sf.norm.StainNormalizer):
            if sf.slide_backend() == 'cucim':
                normalizer = sf.norm.autoselect(  # type: ignore
                    method=normalizer,
                    source=normalizer_source
                )
            else:
                # Libvips with spawn multiprocessing
                # is not compatible with Tensorflow-native stain normalization
                # due to GPU memory issues
                normalizer = sf.norm.StainNormalizer(normalizer)  # type: ignore
                if normalizer_source is not None:
                    normalizer.fit(normalizer_source)  # type: ignore

        if normalizer and context_normalize:
            assert isinstance(normalizer, sf.norm.StainNormalizer)
            log.debug("Preparing whole-slide context for normalizer")
            normalizer.set_context(self)

        w_args = SimpleNamespace(**{
            'full_extract_px': self.full_extract_px,
            'mpp_override': self._mpp_override,
            'reader_kwargs': self._reader_kwargs,
            'roi_scale': self.roi_scale,
            'rois': self.rois,
            'grid': self.grid,
            'downsample_level': self.downsample_level,
            'filter_downsample_level': filter_lev,
            'filter_downsample_ratio': filter_downsample_ratio,
            'path': self.path,
            'extract_px': self.extract_px,
            'tile_px': self.tile_px,
            'full_stride': self.full_stride,
            'normalizer': normalizer,
            'whitespace_fraction': whitespace_fraction,
            'whitespace_threshold': whitespace_threshold,
            'grayspace_fraction': grayspace_fraction,
            'grayspace_threshold': grayspace_threshold,
            'img_format': img_format,
            'yolo': yolo,
            'draw_roi': draw_roi,
            'dry_run': dry_run,
            'has_segmentation': from_centroids
        })

        def generator():
            nonlocal pool, num_threads, num_processes
            should_close = False
            n_extracted = 0

            # Skip tiles filtered out with QC or ROI
            if not from_centroids:
                non_roi_coord = self.coord[
                    self.grid[tuple(self.coord[:, 2:4].T)].astype(bool)
                ]
                # Shuffle coordinates to randomize extraction order
                if shuffle:
                    np.random.shuffle(non_roi_coord)
                num_possible_tiles = len(non_roi_coord)
            else:
                from slideflow.cellseg import seg_utils

                log.info("Building generator from segmentation centroids.")
                nonzero = self.seg_coord[:, 0] > 0
                num_possible_tiles = nonzero.sum()
                if apply_masks:
                    sparse = seg_utils.sparse_mask(self.segmentation.masks)

                def _sparse_generator():

                    def proc(c):
                        mask = None if not apply_masks else self.get_tile_mask(c[2], sparse)
                        return c, mask

                    if shuffle:
                        for idx in np.random.permutation(self.seg_coord.shape[0]):
                            if nonzero[idx]:
                                yield proc(self.seg_coord[idx])
                    else:
                        for c in self.seg_coord[nonzero]:
                            yield proc(c)

                non_roi_coord = _sparse_generator()

            if shard is not None:
                shard_idx, shard_count = shard
                sharded_coords = np.array_split(non_roi_coord, shard_count)
                non_roi_coord = sharded_coords[shard_idx]

            # Set up worker pool
            if pool is None:
                if num_threads is None and num_processes is None:
                    # Libvips is extremely slow with ThreadPools.
                    # In the cuCIM backend, ThreadPools are used by default
                    # to reduce memory utilization.
                    # In the Libvips backend, a multiprocessing pool is default
                    # to significantly improve performance.
                    n_cores = os.cpu_count() if os.cpu_count() else 8
                    if sf.slide_backend() == 'libvips':
                        num_processes = max(int(n_cores/2), 1)
                    else:
                        num_threads = n_cores
                if num_threads is not None and num_threads > 1:
                    log.debug(f"Building generator ThreadPool({num_threads})")
                    pool = mp.dummy.Pool(processes=num_threads)
                    should_close = True
                elif num_processes is not None and num_processes > 1:
                    log.debug(f"Building generator with Pool({num_processes})")
                    ctx = mp.get_context('spawn')
                    pool = ctx.Pool(
                        processes=num_processes,
                        initializer=sf.util.set_ignore_sigint,
                    )
                    should_close = True
                else:
                    log.debug(f"Building generator without multithreading")
                    def _generator():
                        for c in non_roi_coord:
                            yield tile_worker(c, args=w_args)
                    i_mapped = _generator()
            else:
                log.debug("Building generator with a shared pool")
            if show_progress:
                pbar = Progress(transient=sf.getLoggingLevel() > 20)
                task = pbar.add_task('Extracting...', total=self.estimated_num_tiles)
                pbar.start()
            else:
                pbar = None

            if pool is not None:
                map_fn = pool.imap if deterministic else pool.imap_unordered
                if lazy_iter:
                    if max_tiles:
                        batch_size = min(pool._processes, max_tiles)
                    else:
                        batch_size = pool._processes
                    batched_coord = sf.util.batch(non_roi_coord, batch_size)
                    def _generator():
                        for batch in batched_coord:
                            yield from map_fn(
                                partial(tile_worker, args=w_args),
                                batch
                            )
                    i_mapped = _generator()

                else:
                    csize = max(min(int(self.estimated_num_tiles/pool._processes), 64), 1)
                    log.debug(f"Using imap chunksize={csize}")
                    i_mapped = map_fn(
                        partial(tile_worker, args=w_args),
                        non_roi_coord,
                        chunksize=csize
                    )

            with sf.util.cleanup_progress(pbar):
                for e, result in enumerate(i_mapped):
                    if show_progress:
                        pbar.advance(task, 1)
                    elif self.pb is not None:
                        self.pb.advance(0)
                    if result is None:
                        continue
                    else:
                        yield result
                        n_extracted += 1
                        if max_tiles and n_extracted >= max_tiles:
                            break

            if should_close:
                pool.close()

            # Reset stain normalizer context
            if normalizer and context_normalize:
                assert isinstance(normalizer, sf.norm.StainNormalizer)
                normalizer.clear_context()

            name_msg = f'[green]{self.shortname}[/]'
            num_msg = f'({n_extracted} tiles of {num_possible_tiles} possible)'
            log_fn = log.info if self.verbose else log.debug
            log_fn(f"Finished tile extraction for {name_msg} {num_msg}")

        return generator

    def thumb(
        self,
        mpp: Optional[float] = None,
        width: Optional[int] = None,
        coords: Optional[List[int]] = None,
        rois: bool = False,
        linewidth: int = 2,
        color: str = 'black',
        use_associated_image: bool = False,
        low_res: bool = False,
        **kwargs
    ) -> Image.Image:
        """Returns PIL Image of thumbnail with ROI overlay.

        Args:
            mpp (float, optional): Microns-per-pixel, used to determine
                thumbnail size.
            width (int, optional): Goal thumbnail width (alternative to mpp).
            coords (list(int), optional): List of tile extraction coordinates
                to show as rectangles on the thumbnail, in [(x_center,
                y_center), ...] format. Defaults to None.
            rois (bool, optional): Draw ROIs onto thumbnail. Defaults to False.
            linewidth (int, optional): Width of ROI line. Defaults to 2.
            color (str, optional): Color of ROI. Defaults to black.
            use_associated_image (bool): Use the associated thumbnail image
                in the slide, rather than reading from a pyramid layer.
            low_res (bool): Create thumbnail from the lowest-mangnification
                pyramid layer. Defaults to False.

        Returns:
            PIL image

        """
        if rois and len(self.rois):
            if (mpp is not None and width is not None):
                raise ValueError(
                    "Either mpp or width must be given, but not both"
                    f" (got mpp={mpp}, width={width})"
                )
            # If no values provided, create thumbnail of width 1024
            if mpp is None and width is None:
                width = 1024
            if mpp is not None:
                roi_scale = (self.dimensions[0]
                             / (int((self.mpp * self.dimensions[0]) / mpp)))
            else:
                roi_scale = self.dimensions[0] / width  # type: ignore

        thumb = super().thumb(
            mpp=mpp,
            width=width,
            coords=coords,
            use_associated_image=use_associated_image,
            low_res=low_res,
            **kwargs
        )

        if rois and len(self.rois):
            annPolys = [
                sg.Polygon(annotation.scaled_area(roi_scale))
                for annotation in self.rois
            ]
            draw = ImageDraw.Draw(thumb)
            for poly in annPolys:
                x, y = poly.exterior.coords.xy
                zipped = list(zip(x.tolist(), y.tolist()))
                draw.line(zipped, joint='curve', fill=color, width=linewidth)
            return thumb
        else:
            return thumb

    def load_roi_array(self, array: np.ndarray, process: bool = True):
        existing = [
            int(r.name[4:]) for r in self.rois
            if r.name.startswith('ROI_') and r.name[4:].isnumeric()
        ]
        roi_id = list(set(list(range(len(existing)+1))) - set(existing))[0]
        self.rois.append(ROI(f'ROI_{roi_id}', array))
        if process:
            self.process_rois()

    def load_csv_roi(self, path: str, process: bool = True) -> int:
        '''Loads CSV ROI from a given path.'''

        # Clear any previously loaded ROIs.
        self.rois = []
        self.annPolys = []

        roi_dict = {}
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            try:
                headers = next(reader, None)
                if headers is None:
                    raise Exception
                headers = [h.lower() for h in headers]
                index_name = headers.index("roi_name")
                index_x = headers.index("x_base")
                index_y = headers.index("y_base")
            except Exception:
                raise errors.ROIError(
                    f'Unable to read CSV ROI [green]{path}[/]. Please ensure '
                    'headers contain "ROI_name", "X_base and "Y_base".'
                )
            for row in reader:
                roi_name = row[index_name]
                x_coord = int(float(row[index_x]))
                y_coord = int(float(row[index_y]))

                if roi_name not in roi_dict:
                    roi_dict.update({roi_name: ROI(roi_name)})
                roi_dict[roi_name].add_coord((x_coord, y_coord))

            for roi_object in roi_dict.values():
                self.rois.append(roi_object)
        if process:
            self.process_rois()
        return len(self.rois)

    def load_json_roi(
        self,
        path: str,
        scale: int = 10,
        process: bool = True
    ) -> int:
        '''Loads ROI from a JSON file.'''

        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], scale)
            self.rois.append(ROI(f"Object{len(self.rois)}"))
            self.rois[-1].add_shape(area_reduced)
        if process:
            self.process_rois()
        return len(self.rois)

    def masked_thumb(self, background: str = 'white', **kwargs) -> np.ndarray:
        """Return a masked thumbnail of a slide, using QC and/or ROI masks.
        Masked areas will be white.
        """
        if background not in ('white', 'black'):
            raise ValueError(
                f"Unexpected background option: '{background}'. Expected "
                "'black' or 'white'."
            )
        qc_mask = self.qc_mask
        roi_mask = self.roi_mask
        image = np.asarray(self.thumb(**kwargs))
        if qc_mask is None and roi_mask is None:
            # Apply Otsu's threshold to background area
            # to prevent whitespace from interfering with normalization
            from slideflow.slide.qc import Otsu, Gaussian
            sf.log.debug(
                "Applying Otsu's thresholding & Gaussian blur filter "
                "to stain norm context"
            )
            _blur_mask = Gaussian()(image)
            qc_mask = Otsu()(image, mask=_blur_mask)
        # Mask by ROI and QC, if applied.
        # Use white as background for masked areas.
        if qc_mask is not None:
            qc_img = sf.slide.img_as_ubyte(qc_mask)
            mask = ~cv2.resize(qc_img, (image.shape[1], image.shape[0]))
        if roi_mask is not None:
            roi_img = sf.slide.img_as_ubyte(roi_mask)
            roi_mask = cv2.resize(roi_img, (image.shape[1], image.shape[0]))
            if qc_mask is not None:
                mask = mask & roi_mask
            else:
                mask = roi_mask
        if background == 'white':
            white_bg = np.full(image.shape, 255, dtype=np.uint8)
            white_mask = cv2.bitwise_or(white_bg, white_bg, mask=~mask)
            return cv2.bitwise_or(image, white_mask)
        else:
            return cv2.bitwise_or(image, image, mask=mask)

    def predict(
        self,
        model: str,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate a whole-slide prediction from a saved model.

        Args:
            model (str): Path to saved model trained in Slideflow.

        Keyword args:
            batch_size (int, optional): Batch size for calculating predictions.
                Defaults to 32.
            num_threads (int, optional): Number of tile worker threads. Cannot
                supply both ``num_threads`` (uses thread pool) and
                ``num_processes`` (uses multiprocessing pool). Defaults to
                CPU core count.
            num_processes (int, optional): Number of child processes to spawn
                for multiprocessing pool. Defaults to None (does not use
                multiprocessing).
            img_format (str, optional): Image format (png, jpg) to use when
                extracting tiles from slide. Must match the image format
                the model was trained on. If 'auto', will use the format
                logged in the model params.json. Defaults to 'auto'.
            device (torch.device, optional): PyTorch device. Defaults to
                initializing a new CUDA device.
            generator_kwargs (dict, optional): Keyword arguments passed to
                the :meth:`slideflow.WSI.build_generator()`.

        Returns:
            np.ndarray: Predictions for each outcome, with shape = (num_classes, )

            np.ndarray, optional: Uncertainty for each outcome, if the model was
            trained with uncertainty, with shape = (num_classes,)

        """
        from slideflow import Heatmap

        config = sf.util.get_model_config(model)
        if config['tile_px'] != self.tile_px or config['tile_um'] != self.tile_um:
            raise ValueError(
                "Slide tile size (tile_px={}, tile_um={}) does not match the "
                "model (tile_px={}, tile_um={}).".format(
                    self.tile_px, self.tile_um,
                    config['tile_px'], config['tile_um']
            ))
        log.info("Calculating whole-slide prediction...")
        heatmap = Heatmap(self, model, generate=True, **kwargs)
        preds = heatmap.predictions.reshape(-1, heatmap.predictions.shape[-1])
        preds = np.nanmean(np.ma.masked_where(preds == -99, preds), axis=0).filled()
        if heatmap.uncertainty is not None:
            unc = heatmap.uncertainty.reshape(-1, heatmap.uncertainty.shape[-1])
            unc = np.nanmean(np.ma.masked_where(unc == -99, unc), axis=0).filled()
            return preds, unc
        else:
            return preds

    def process_rois(self):
        """Process loaded ROIs and apply to the slide grid."""

        # Load annotations as shapely.geometry objects.
        if self.roi_method != 'ignore':
            self.annPolys = []
            for i, annotation in enumerate(self.rois):
                try:
                    poly = sv.make_valid(sg.Polygon(annotation.scaled_area(self.roi_scale)))
                    self.annPolys += [poly]
                except ValueError:
                    log.warning(
                        f"Unable to use ROI {i} for [green]{self.name}[/]."
                        " At least 3 points required to create a shape."
                    )
            # Handle polygon holes.
            outers, inners = [], []
            for o, outer in enumerate(self.annPolys):
                for i, inner in enumerate(self.annPolys):
                    if o == i:
                        continue
                    if (i in inners) or (o in inners) or (i in outers):
                        continue
                    if outer.contains(inner):
                        log.debug(f"Rendering ROI polygon {i} as hole in {o}")
                        self.annPolys[o] = self.annPolys[o].difference(inner)
                        if o not in outers:
                            outers.append(o)
                        if i not in inners:
                            inners.append(i)
            self.annPolys = [ann for (i, ann) in enumerate(self.annPolys)
                             if i not in inners]
            roi_area = sum([poly.area for poly in self.annPolys])
        else:
            roi_area = 1
        total_area = ((self.dimensions[0]/self.roi_scale)
                      * (self.dimensions[1]/self.roi_scale))
        self.roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

        # Regenerate the grid to reflect the newly-loaded ROIs.
        self._build_coord()

        return len(self.rois)

    def remove_roi(self, idx: int, process: bool = True) -> None:
        del self.rois[idx]
        if process:
            self.process_rois()

    def tensorflow(
        self,
        img_format: str = 'numpy',
        incl_slidenames: bool = False,
        incl_loc: Optional[str] = None,
        shuffle: bool = True,
        **kwargs
    ) -> Any:
        """Create a Tensorflow Dataset which extractes tiles from this slide.

        Args:
            img_format (str, optional): Image format for returned image tiles.
                Options include 'png', 'jpg', and 'numpy'. Defaults to 'numpy'.
            incl_slidenames (bool, optional): Yield slide names for each
                image tile. Defaults to False.
            incl_loc (Optional[str], optional): Yield image tile location
                with each image tile. Options include True, 'coord', or 'grid'.
                If True or 'coord', will return X/Y coordinates of the tile
                in the slide's highest magnification layer. If 'grid', returns
                the grid indices for the tile. Defaults to None.
            shuffle (bool, optional): Shuffle image tiles. Defaults to True.

        Returns:
            tf.data.Dataset

        Yields:
            Iterator[Any]: Items yielded by the Dataset are in dictionary
            format, with the keys:

            'image_raw':    Contains the image (jpg, png, or numpy)
            'slide':        Slide name (if ``incl_slidenames=True``)
            'loc_x'         Image tile x location (if ``incl_loc`` provided)
            'loc_y'         Image tile y location (if ``incl_loc`` provided)
        """

        import tensorflow as tf

        def tile_generator():
            for image_dict in self.build_generator(
                shuffle=shuffle,
                show_progress=False,
                img_format=img_format,
                **kwargs
            )():
                if not (incl_slidenames or incl_loc):
                    yield image_dict['image']
                else:
                    to_return = {
                        'image_raw': image_dict['image']
                    }
                    if incl_slidenames:
                        to_return['slide'] = self.name
                    if incl_loc == 'coord' or incl_loc == True:
                        to_return['loc_x'] = image_dict['loc'][0]
                        to_return['loc_y'] = image_dict['loc'][1]
                    if incl_loc == 'grid':
                        to_return['loc_x'] = image_dict['grid'][0]
                        to_return['loc_y'] = image_dict['grid'][1]
                    yield to_return

        # Generate dataset from the generator
        with tf.name_scope('dataset_input'):
            # Signatures for imaging data
            if img_format == 'numpy':
                image_sig = tf.TensorSpec(
                    shape=(self.tile_px, self.tile_px, 3),
                    dtype=tf.uint8
                )
            else:
                image_sig = tf.TensorSpec(shape=(), dtype=tf.string)

            # Rest of the signatures
            if incl_slidenames or incl_loc:
                sig = {'image_raw': image_sig}
                if incl_slidenames:
                    sig['slide'] = tf.TensorSpec(shape=(), dtype=tf.string)
                if incl_loc:
                    sig['loc_x'] = tf.TensorSpec(shape=(), dtype=tf.int32)
                    sig['loc_y'] = tf.TensorSpec(shape=(), dtype=tf.int32)
            else:
                sig = image_sig

            # Assemble dataset
            dataset = tf.data.Dataset.from_generator(
                tile_generator,
                output_signature=sig
            )

        return dataset

    def torch(
        self,
        img_format: str = 'numpy',
        incl_slidenames: bool = False,
        incl_loc: Optional[str] = None,
        shuffle: bool = True,
        infinite: bool = False,
        to_tensor: bool = True,
        **kwargs
    ) -> Any:
        """Create a PyTorch iterator which extractes tiles from this slide.

        Args:
            img_format (str, optional): Image format for returned image tiles.
                Options include 'png', 'jpg', and 'numpy'. Defaults to 'numpy'.
            incl_slidenames (bool, optional): Yield slide names for each
                image tile. Defaults to False.
            incl_loc (Optional[str], optional): Yield image tile location
                with each image tile. Options include True, 'coord', or 'grid'.
                If True or 'coord', will return X/Y coordinates of the tile
                in the slide's highest magnification layer. If 'grid', returns
                the grid indices for the tile. Defaults to None.
            shuffle (bool, optional): Shuffle image tiles. Defaults to True.

        Returns:
            An iterator which yields image tiles as Torch tensors.

        Yields:
            Iterator[Any]: Items yielded by the Dataset are in dictionary
            format, with the keys:

            'image_raw':    Contains the image as a Tensor (jpg, png, or numpy)
            'slide':        Slide name (if ``incl_slidenames=True``)
            'loc_x'         Image tile x location (if ``incl_loc`` provided)
            'loc_y'         Image tile y location (if ``incl_loc`` provided)
        """
        import torch

        def tile_generator():
            while True:
                for image_dict in self.build_generator(
                    shuffle=shuffle,
                    show_progress=False,
                    img_format=img_format,
                    **kwargs
                )():
                    if not (incl_slidenames or incl_loc):
                        if to_tensor:
                            yield torch.from_numpy(image_dict['image'])
                        else:
                            yield image_dict['image']
                    else:
                        if to_tensor:
                            to_return = {'image_raw': torch.from_numpy(image_dict['image'])}
                        else:
                            to_return = {'image_raw': image_dict['image']}
                        if incl_slidenames:
                            to_return['slide'] = self.name
                        if incl_loc == 'coord' or incl_loc == True:
                            to_return['loc_x'] = image_dict['loc'][0]
                            to_return['loc_y'] = image_dict['loc'][1]
                        if incl_loc == 'grid':
                            to_return['loc_x'] = image_dict['grid'][0]
                            to_return['loc_y'] = image_dict['grid'][1]
                        yield to_return
                if not infinite:
                    break

        return tile_generator()

    def view(self):
        """Open the slide in Slideflow Studio for interactive display.

        See :ref:`studio` for more information.

        """
        from slideflow.studio import Studio

        studio = Studio()
        studio.load_slide(self.path, stride=self.stride_div)
        studio.run()


class TMA(_BaseLoader):
    '''Loads a TMA-formatted slide and detects tissue cores.'''

    QUEUE_SIZE = 8
    HEIGHT_MIN = 20
    WIDTH_MIN = 20
    BLACK = (0, 0, 0)
    BLUE = (255, 100, 100)
    GREEN = (75, 220, 75)
    LIGHTBLUE = (255, 180, 180)
    RED = (100, 100, 200)
    WHITE = (255, 255, 255)

    def __init__(
        self,
        path: str,
        tile_px: int,
        tile_um: Union[str, int],
        stride_div: int = 1,
        enable_downsample: bool = True,
        report_dir: Optional[str] = None,
        pb: Optional[Progress] = None,
        **kwargs
    ) -> None:
        '''Initializer.

        Args:
            path (str): Path to slide.
            tile_px (int): Size of tiles to extract, in pixels.
            tile_um (int or str): Size of tiles to extract, in microns (int) or
                magnification (str, e.g. "20x").
            stride_div (int, optional): Stride divisor for tile extraction
                (1 = no tile overlap; 2 = 50% overlap, etc). Defaults to 1.
            enable_downsample (bool, optional): Allow use of downsampled
                layers in the slide image pyramid, which greatly improves
                tile extraction speed. Defaults to True.
            pb (Progress, optional): Progress bar; will update
                progress bar during tile extraction if provided.
                Defaults to None.
            mpp (float, optional): Override the microns-per-pixel value for
                the slide. Defaults to None (auto-detects).
        '''
        super().__init__(
            path,
            tile_px,
            tile_um,
            stride_div,
            enable_downsample,
            pb,
            **kwargs
        )
        self.object_rects = []  # type: List
        self.box_areas = []  # type: List
        self.DIM = self.slide.dimensions
        self.roi_method = 'ignore'
        self.roi_scale = 1  # type: float
        target_thumb_width = self.DIM[0] / 100
        target_thumb_mpp = self.dim_to_mpp((target_thumb_width, -1))
        self.thumb_image = np.array(self.thumb(mpp=target_thumb_mpp))
        self.thumb_image = self.thumb_image[:, :, :-1]
        self.THUMB_DOWNSCALE = (self.DIM[0]
                                / self.mpp_to_dim(target_thumb_mpp)[0])
        self.pb = pb
        self.estimated_num_tiles = self._detect_cores(report_dir=report_dir)  # type: int
        size_msg = f'Size: {self.dimensions[0]} x {self.dimensions[1]}'
        log.info(f"{self.shortname}: {self.mpp} um/px | {size_msg}")

    def _get_sub_image(self, rect: List[List[int]]) -> np.ndarray:
        '''Gets a sub-image from the slide using the specified rectangle.'''
        box = cv2.boxPoints(rect) * self.THUMB_DOWNSCALE
        box = np.int0(box)

        rect_width = int(rect[1][0]
                         * self.THUMB_DOWNSCALE
                         / self.downsample_factor)
        rect_height = int(rect[1][1]
                          * self.THUMB_DOWNSCALE
                          / self.downsample_factor)

        region_x_min = int(min([b[0] for b in box]))
        region_x_max = max([b[0] for b in box])
        region_y_min = int(min([b[1] for b in box]))
        region_y_max = max([b[1] for b in box])
        region_width = int((region_x_max - region_x_min)
                           / self.downsample_factor)
        region_height = int((region_y_max - region_y_min)
                            / self.downsample_factor)

        region = self.slide.read_region(
            (region_x_min, region_y_min),
            self.downsample_level,
            (region_width, region_height),
            to_numpy=True
        )
        extracted = region[:, :, :-1]
        relative_box = ((box - [region_x_min, region_y_min])
                        / self.downsample_factor)

        src_pts = relative_box.astype("float32")
        dst_pts = np.array([
            [0, (rect_height)-1],
            [0, 0],
            [(rect_width)-1, 0],
            [(rect_width)-1, (rect_height)-1]
        ], dtype="float32")
        P = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(extracted, P, (rect_width, rect_height))
        return warped

    def _resize_to_target(self, image_tile: np.ndarray) -> np.ndarray:
        '''Resizes image tile to the desired target output size.'''
        target_MPP = self.tile_um / self.tile_px
        current_MPP = self.mpp * self.downsample_factor
        resize_factor = current_MPP / target_MPP
        return cv2.resize(
            image_tile,
            (0, 0),
            fx=resize_factor,
            fy=resize_factor
        )

    def _split_core(self, image: np.ndarray) -> List[np.ndarray]:
        '''Splits core into desired sub-images.'''
        height, width, channels = image.shape
        num_y = int(height / self.tile_px)
        num_x = int(width / self.tile_px)

        # If the desired micron tile size is too large,
        # expand and center the source image
        if not num_y or not num_x:
            expand_y = 0 if num_y else int((self.tile_px-height)/2)+1
            expand_x = 0 if num_x else int((self.tile_px-width)/2)+1
            image = cv2.copyMakeBorder(image,
                                       expand_y,
                                       expand_y,
                                       expand_x,
                                       expand_x,
                                       cv2.BORDER_CONSTANT,
                                       value=self.WHITE)

            height, width, _ = image.shape
            num_y = int(height / self.tile_px)
            num_x = int(width / self.tile_px)

        y_start = int((height - (num_y * self.tile_px))/2)
        x_start = int((width - (num_x * self.tile_px))/2)

        subtiles = []

        for y in range(num_y):
            for x in range(num_x):
                sub_x_start = x_start + (x * self.tile_px)
                sub_y_start = y_start + (y * self.tile_px)
                subtiles += [image[sub_y_start:sub_y_start+self.tile_px,
                                   sub_x_start:sub_x_start+self.tile_px]]

        return subtiles

    def _detect_cores(self, report_dir: Optional[str] = None) -> int:
        # Prepare annotated image
        assert self.thumb_image is not None
        img_annotated = self.thumb_image.copy()

        # Create background mask for edge detection
        white = np.array([255, 255, 255])
        buffer = 28
        mask = cv2.inRange(self.thumb_image, np.array([0, 0, 0]), white-buffer)

        # Fill holes and dilate mask
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilating_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, closing_kernel)
        dilated = cv2.dilate(closing, dilating_kernel)

        # Use edge detection to find individual cores
        contours, heirarchy = cv2.findContours(
            dilated,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_TC89_L1
        )
        # Filter out small regions that likely represent background noise
        # Also generate image showing identified cores
        num_filtered = 0
        for i, component in enumerate(zip(contours, heirarchy[0])):
            cnt = component[0]
            heir = component[1]
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]
            if (width > self.WIDTH_MIN
               and height > self.HEIGHT_MIN
               and heir[3] < 0):
                moment = cv2.moments(cnt)
                self.object_rects += [(len(self.object_rects), rect)]
                cX = int(moment["m10"] / moment["m00"])
                cY = int(moment["m01"] / moment["m00"])
                cv2.drawContours(img_annotated, contours, i, self.LIGHTBLUE)
                cv2.circle(img_annotated, (cX, cY), 4, self.GREEN, -1)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                area = _polyArea([b[0] for b in box], [b[1] for b in box])
                self.box_areas += [area]
                cv2.drawContours(img_annotated, [box], 0, self.BLUE, 2)
                num_filtered += 1
            elif heir[3] < 0:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_annotated, [box], 0, self.RED, 2)
        log.info(f"Number of detected cores: {num_filtered}")

        # Write annotated image to ExtractionReport
        if report_dir:
            cv2.imwrite(
                join(report_dir, "tma_extraction_report.jpg"),
                cv2.resize(img_annotated, (1400, 1000))
            )
        return num_filtered

    def extract_tiles(
        self,
        tfrecord_dir: Optional[str] = None,
        tiles_dir: Optional[str] = None,
        img_format: str = 'jpg',
        report: bool = True,
        **kwargs
    ) -> Optional[SlideReport]:
        """Extracts tiles from slide using the build_generator() method,
        saving tiles into a TFRecord file or as loose JPG tiles in a directory.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file
                (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images into a
                subdirectory (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal
                storage in tfrecords. PNG (lossless) format recommended for
                fidelity, JPG (lossy) for efficiency.
                Defaults to 'jpg'.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not
                perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            normalizer (str, optional): Normalization for image tiles.
                Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                If None, will use slideflow.slide.norm_tile.jpg.
                Defaults to None.
            full_core (bool, optional): Extract an entire detected core, rather
                than subdividing into image tiles. Defaults to False.
            shuffle (bool): Shuffle images during extraction.
            num_threads (int): Number of threads to allocate to workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI
                annotations (.txt) in the tile directory. Requires that
                tiles_dir is set. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles.
                Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted,
                but do not export any images. Defaults to None.
        """
        return super().extract_tiles(
            tfrecord_dir,
            tiles_dir,
            img_format,
            report,
            **kwargs
        )

    def build_generator(
        self,
        *,
        shuffle: bool = True,
        whitespace_fraction: float = None,
        whitespace_threshold: float = None,
        grayspace_fraction: float = None,
        grayspace_threshold: float = None,
        normalizer: str = None,
        normalizer_source: str = None,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
        img_format: str = 'numpy',
        full_core: bool = False,
        yolo: bool = False,
        draw_roi: bool = False,
        pool: Optional["mp.pool.Pool"] = None,
        dry_run: bool = False
    ) -> Optional[Callable]:
        """Builds tile generator to extract of tiles across the slide.

        Args:
            shuffle (bool): Shuffle images during extraction.
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is  whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not
                perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            normalizer (str, optional): Normalization to use on image tiles.
                Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                If None, will use slideflow.slide.norm_tile.jpg
                Defaults to None.
            num_threads (int, optional): Number of threads for pool. Unused if
                `pool` is specified.
            pool (:obj:`multiprocessing.Pool`, optional): Multiprocessing pool
                to use. By using a shared pool, a slide no longer needs to spin
                up its own new pool for tile extraction, decreasing tile
                extraction time for large datasets. Defaults to None (create a
                new pool, using `num_threads`).
            img_format (str, optional): 'png', 'jpg', or 'numpy'. Format images
                should be returned in.
            full_core (bool, optional): Extract an entire detected core, rather
                than subdividing into image tiles. Defaults to False.
            show_progress (bool, optional): Show a progress bar for extraction.
        """
        import matplotlib.colors as mcol

        super().build_generator()
        if yolo:
            raise NotImplementedError(
                "Yolo annotation export not implemented for TMA slides"
            )
        if draw_roi:
            raise NotImplementedError(
                "ROI drawing not implemented for TMA slides"
            )
        if dry_run:
            raise NotImplementedError(
                "Dry running not enabled for TMA slides"
            )
        # Setup normalization
        norm = None if not normalizer else sf.norm.autoselect(
            method=normalizer,
            source=normalizer_source
        )
        # Detect CPU cores if num_threads not specified
        if num_threads is None:
            num_threads = os.cpu_count()
            if num_threads is None:
                num_threads = 8

        # Shuffle TMAs
        if shuffle:
            random.shuffle(self.object_rects)

        # Set whitespace / grayspace fraction to defaults if not provided
        if whitespace_fraction is None:
            whitespace_fraction = DEFAULT_WHITESPACE_FRACTION
        if whitespace_threshold is None:
            whitespace_threshold = DEFAULT_WHITESPACE_THRESHOLD
        if grayspace_fraction is None:
            grayspace_fraction = DEFAULT_GRAYSPACE_FRACTION
        if grayspace_threshold is None:
            grayspace_threshold = DEFAULT_GRAYSPACE_THRESHOLD

        # Establish extraction queues
        rectangle_queue = mp.Queue()  # type: mp.Queue
        extraction_queue = mp.Queue(self.QUEUE_SIZE)  # type: mp.Queue

        def section_extraction_worker(read_queue, write_queue):
            while True:
                tile_id, rect = read_queue.get(True)
                if rect == "DONE":
                    write_queue.put((tile_id, rect))
                    break
                else:
                    image_tile = self._get_sub_image(rect)
                    write_queue.put((tile_id, image_tile))

        def generator():
            if show_progress:
                pbar = Progress(transient=sf.getLoggingLevel()>20)
                task = pbar.add_task(
                    "Extracting...",
                    total=self.estimated_num_tiles
                )
                pbar.start()
            else:
                pbar = None
            with sf.util.cleanup_progress(pbar):
                ctx = mp.get_context('spawn')
                extraction_pool = ctx.Pool(
                    num_threads,
                    section_extraction_worker,
                    (rectangle_queue, extraction_queue,)
                )
                for rect in self.object_rects:
                    rectangle_queue.put(rect)
                rectangle_queue.put((-1, "DONE"))

                queue_progress = 0
                while True:
                    queue_progress += 1
                    tile_id, image_core = extraction_queue.get()
                    if type(image_core) == str and image_core == "DONE":
                        break
                    else:
                        if self.pb:
                            self.pb.advance(0)
                        if show_progress:
                            pbar.advance(task, 1)

                        resized_core = self._resize_to_target(image_core)

                        if full_core:
                            resized = cv2.resize(
                                image_core,
                                (self.tile_px, self.tile_px)
                            )
                            # Convert to final image format
                            if img_format != 'numpy':
                                resized = _convert_img_to_format(
                                    resized,
                                    img_format
                                )

                            yield {'image': resized, 'loc': [0, 0]}
                        else:
                            subtiles = self._split_core(resized_core)
                            for subtile in subtiles:
                                # Perform whitespace filtering
                                if whitespace_fraction < 1:
                                    frac = (np.mean(subtile, axis=2)
                                            > whitespace_threshold).sum()
                                    frac = frac / (self.tile_px**2)
                                    if frac > whitespace_fraction:
                                        continue

                                # Perform grayspace filtering
                                if grayspace_fraction < 1:
                                    hsv_image = mcol.rgb_to_hsv(subtile)
                                    frac = (hsv_image[:, :, 1]
                                            < grayspace_threshold).sum()
                                    frac = frac / (self.tile_px**2)
                                    if frac > grayspace_fraction:
                                        continue

                                # Apply normalization
                                if norm is not None:
                                    try:
                                        subtile = norm.rgb_to_rgb(subtile)
                                    except Exception:
                                        # The image could not be normalized, which
                                        # happens when a tile is primarily one
                                        # solid color (background)
                                        continue

                                # Convert to final image format
                                if img_format != 'numpy':
                                    subtile = _convert_img_to_format(
                                        subtile,
                                        img_format
                                    )
                                yield {'image': subtile, 'loc': [0, 0]}
                extraction_pool.close()

        return generator
