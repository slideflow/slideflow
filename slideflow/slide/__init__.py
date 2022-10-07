'''This module includes tools to convolutionally section whole slide images
into tiles. These tessellated tiles can be exported as PNG or JPG as raw
images or stored in the binary format TFRecords, with or without augmentation.

Requires: libvips (https://libvips.github.io/libvips/).'''

from __future__ import absolute_import, division, print_function

import re
import csv
import io
import json
import multiprocessing as mp
import os
import random
import time
import warnings
from functools import partial
from os.path import exists, join
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry as sg
import shapely.affinity as sa
import skimage
import skimage.filters
import slideflow as sf
from PIL import Image, ImageDraw, UnidentifiedImageError
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from slideflow import errors
from slideflow.slide.report import ExtractionPDF  # noqa F401
from slideflow.slide.report import ExtractionReport, SlideReport
from slideflow.util import SUPPORTED_FORMATS  # noqa F401
from slideflow.util import log, path_to_name  # noqa F401
from rich.progress import Progress

try:
    import pyvips as vips
except (ModuleNotFoundError, OSError) as e:
    log.error("Unable to load vips; slide processing will be unavailable. "
              f"Error raised: {e}")

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 100000000000
DEFAULT_JPG_MPP = 1
OPS_LEVEL_COUNT = 'openslide.level-count'
OPS_MPP_X = 'openslide.mpp-x'
OPS_VENDOR = 'openslide.vendor'
TIF_EXIF_KEY_MPP = 65326
OPS_WIDTH = 'width'
OPS_HEIGHT = 'height'
DEFAULT_WHITESPACE_THRESHOLD = 230
DEFAULT_WHITESPACE_FRACTION = 1.0
DEFAULT_GRAYSPACE_THRESHOLD = 0.05
DEFAULT_GRAYSPACE_FRACTION = 0.6
FORCE_CALCULATE_WHITESPACE = -1
FORCE_CALCULATE_GRAYSPACE = -1


def OPS_LEVEL_HEIGHT(level: int) -> str:
    return f'openslide.level[{level}].height'


def OPS_LEVEL_WIDTH(level: int) -> str:
    return f'openslide.level[{level}].width'


def OPS_LEVEL_DOWNSAMPLE(level: int) -> str:
    return f'openslide.level[{level}].downsample'


VIPS_FORMAT_TO_DTYPE = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


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


def _draw_roi(
    img: Union[np.ndarray, str],
    coords: List[int]
) -> np.ndarray:
    """Draw ROIs on image.

    Args:
        img (Union[np.ndarray, str]): Image.
        coords (List[List[int]]): ROI coordinates.

    Returns:
        np.ndarray: Image as numpy array.
    """
    annPolys = [sg.Polygon(b) for b in coords]
    if isinstance(img, np.ndarray):
        annotated_img = Image.fromarray(img)
    elif isinstance(img, str):
        annotated_img = Image.open(io.BytesIO(img))  # type: ignore
    draw = ImageDraw.Draw(annotated_img)
    for poly in annPolys:
        x, y = poly.exterior.coords.xy
        zipped = list(zip(x.tolist(), y.tolist()))
        draw.line(zipped, joint='curve', fill='red', width=5)
    return np.asarray(annotated_img)


def _roi_coords_from_image(
    c: List[int],
    args: SimpleNamespace
) -> Tuple[List[int], List[np.ndarray], List[List[int]]]:
    # Scale ROI according to downsample level
    extract_scale = (args.extract_px / args.full_extract_px)

    # Scale ROI according to image resizing
    resize_scale = (args.tile_px / args.extract_px)

    def proc_ann(ann):
        # Scale to full image size
        coord = ann.coordinates
        # Offset coordinates to extraction window
        coord = np.add(coord, np.array([-1 * c[0], -1 * c[1]]))
        # Rescale according to downsampling and resizing
        coord = np.multiply(coord, (extract_scale * resize_scale))
        return coord

    # Filter out ROIs not in this tile
    coords = []
    ll = np.array([0, 0])
    ur = np.array([args.tile_px, args.tile_px])
    for roi in args.rois:
        coord = proc_ann(roi)
        idx = np.all(np.logical_and(ll <= coord, coord <= ur), axis=1)
        coords_in_tile = coord[idx]
        if len(coords_in_tile) > 3:
            coords += [coords_in_tile]

    # Convert ROI to bounding box that fits within tile
    boxes = []
    yolo_anns = []
    for coord in coords:
        max_vals = np.max(coord, axis=0)
        min_vals = np.min(coord, axis=0)
        max_x = min(max_vals[0], args.tile_px)
        max_y = min(max_vals[1], args.tile_px)
        min_x = max(min_vals[0], 0)
        min_y = max(0, min_vals[1])
        width = (max_x - min_x) / args.tile_px
        height = (max_y - min_y) / args.tile_px
        x_center = ((max_x + min_x) / 2) / args.tile_px
        y_center = ((max_y + min_y) / 2) / args.tile_px
        yolo_anns += [[x_center, y_center, width, height]]
        boxes += [np.array([
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y]
        ])]
    return coords, boxes, yolo_anns


def _wsi_extraction_worker(
    c: List[int],
    args: SimpleNamespace
) -> Optional[Union[str, Dict]]:
    '''Multiprocessing worker for WSI. Extracts tile at given coordinates.'''

    x, y, grid_x, grid_y = c
    x_coord = int((x + args.full_extract_px / 2) / args.roi_scale)
    y_coord = int((y + args.full_extract_px / 2) / args.roi_scale)

    # If downsampling is enabled, read image from highest level
    # to perform filtering; otherwise filter from our target level
    slide = args.vips_wrapper(args.path, args.mpp_override, args.vips_cache)
    if args.whitespace_fraction < 1 or args.grayspace_fraction < 1:
        if args.filter_downsample_ratio > 1:
            filter_extract_px = args.extract_px // args.filter_downsample_ratio
            filter_region = slide.read_region(
                (x, y),
                args.filter_downsample_level,
                (filter_extract_px, filter_extract_px)
            )
        else:
            # Read the region and resize to target size
            filter_region = slide.read_region(
                (x, y),
                args.downsample_level,
                (args.extract_px, args.extract_px)
            )
        # Perform whitespace filtering [Libvips]
        if args.whitespace_fraction < 1:
            ws_fraction = filter_region.bandmean().relational_const(
                'more',
                args.whitespace_threshold
            ).avg() / 255
            if (ws_fraction > args.whitespace_fraction
               and args.whitespace_fraction != FORCE_CALCULATE_WHITESPACE):
                return None

        # Perform grayspace filtering [Libvips]
        if args.grayspace_fraction < 1:
            hsv_region = filter_region.sRGB2HSV()
            gs_fraction = hsv_region[1].relational_const(
                'less',
                args.grayspace_threshold*255
            ).avg() / 255
            if (gs_fraction > args.grayspace_fraction
               and args.whitespace_fraction != FORCE_CALCULATE_WHITESPACE):
                return None

    # Prepare return dict with WS/GS fraction
    return_dict = {'loc': [x_coord, y_coord]}  # type: Dict[str, Any]
    return_dict.update({'grid': [grid_x, grid_y]})
    if args.grayspace_fraction < 1:
        return_dict.update({'gs_fraction': gs_fraction})
    if args.whitespace_fraction < 1:
        return_dict.update({'ws_fraction': ws_fraction})

    # If dry run, return without the image
    if args.dry_run:
        return_dict.update({'loc': [x_coord, y_coord]})
        return return_dict

    # Normalizer
    if not args.normalizer:
        normalizer = None
    else:
        normalizer = sf.norm.autoselect(
            method=args.normalizer,
            source=args.normalizer_source
        )

    # Read the target downsample region now, if we were
    # filtering at a different level
    region = slide.read_region(
        (x, y),
        args.downsample_level,
        (args.extract_px, args.extract_px)
    )
    if region.bands == 4:
        region = region.flatten()  # removes alpha
    if int(args.tile_px) != int(args.extract_px):
        region = region.resize(args.tile_px/args.extract_px)
    assert(region.width == region.height == args.tile_px)

    if args.img_format != 'numpy':
        if args.img_format == 'png':
            image = region.pngsave_buffer()
        elif args.img_format in ('jpg', 'jpeg'):
            image = region.jpegsave_buffer()
        else:
            raise ValueError(f"Unknown image format {args.img_format}")

        # Apply normalization
        if normalizer:
            try:
                if args.img_format == 'png':
                    image = normalizer.png_to_png(image)
                elif args.img_format in ('jpg', 'jpeg'):
                    image = normalizer.jpeg_to_jpeg(image)
                else:
                    raise ValueError(f"Unknown image format {args.img_format}")
            except Exception:
                # The image could not be normalized,
                # which happens when a tile is primarily one solid color
                return None
    else:
        # Read regions into memory and convert to numpy arrays
        image = vips2numpy(region)

        # Apply normalization
        if normalizer:
            try:
                image = normalizer.rgb_to_rgb(image)
            except Exception:
                # The image could not be normalized,
                # which happens when a tile is primarily one solid color
                return None

    # Include ROI / bounding box processing.
    # Used to visualize ROIs on extracted tiles, or to generate YoloV5 labels.
    if args.yolo or args.draw_roi:
        coords, boxes, yolo_anns = _roi_coords_from_image(c, args)
    if args.draw_roi:
        image = _draw_roi(image, coords)

    return_dict.update({'image': image})
    if args.yolo:
        return_dict.update({'yolo': yolo_anns})
    return return_dict


def vips2numpy(vi: "vips.Image") -> np.ndarray:
    '''Converts a VIPS image into a numpy array'''
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def vips_resize(
    img: np.ndarray,
    crop_width: int,
    target_px: int
) -> np.ndarray:
    """Resizes and crops an image using libvips.resize()

    Args:
        img (np.ndarray): Image.
        crop_width (int): Height/width of image crop (before resize).
        target_px (int): Target size of final image after resizing.

    Returns:
        np.ndarray: Resized image.
    """
    img_data = np.ascontiguousarray(img).data
    vips_image = vips.Image.new_from_memory(
        img_data,
        crop_width,
        crop_width,
        bands=3,
        format="uchar"
    )
    vips_image = vips_image.resize(target_px/crop_width)
    return vips2numpy(vips_image)


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


class _VIPSWrapper:

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        cache_kw: Optional[Dict[str, Any]] = None
    ) -> None:
        '''Wrapper for VIPS to preserve openslide-like functions.'''
        self.path = path
        self.cache_kw = cache_kw if cache_kw else {}
        self.loaded_downsample_levels = {}  # type: Dict[int, "vips.Image"]
        loaded_image = self.load_downsample_level(0)

        # Load image properties
        self.properties = {}
        for field in loaded_image.get_fields():
            self.properties.update({field: loaded_image.get(field)})
        self.dimensions = (
            int(self.properties[OPS_WIDTH]),
            int(self.properties[OPS_HEIGHT])
        )
        # If Openslide MPP is not available, try reading from metadata
        if mpp is not None:
            log.debug(f"Setting MPP to {mpp}")
            self.properties[OPS_MPP_X] = mpp
        elif OPS_MPP_X not in self.properties.keys():
            log.debug(
                "Microns-Per-Pixel (MPP) not found, Searching EXIF"
            )
            try:
                with Image.open(path) as img:
                    if TIF_EXIF_KEY_MPP in img.tag.keys():
                        _mpp = img.tag[TIF_EXIF_KEY_MPP][0]
                        log.debug(
                            f"Using MPP {_mpp} per EXIF {TIF_EXIF_KEY_MPP}"
                        )
                        self.properties[OPS_MPP_X] = _mpp
                    elif (sf.util.path_to_ext(path).lower() == 'svs'
                          and 'image-description' in loaded_image.get_fields()):
                          img_des = loaded_image.get('image-description')
                          _mpp = re.findall(r'(?<=MPP\s\=\s)0\.\d+', img_des)
                          if _mpp is not None:
                            log.debug(
                                f"Using MPP {_mpp} from 'image-description' for SCN"
                                "-converted SVS format"
                            )
                            self.properties[OPS_MPP_X] = _mpp[0]
                    elif (sf.util.path_to_ext(path).lower() in ('tif', 'tiff')
                          and 'xres' in loaded_image.get_fields()):
                        xres = loaded_image.get('xres')  # 4000.0
                        if (xres == 4000.0
                           and loaded_image.get('resolution-unit') == 'cm'):
                            # xres = xres # though resolution from tiffinfo
                            # says 40000 pixels/cm, for some reason the xres
                            # val is 4000.0, so multipley by 10.
                            # Convert from pixels/cm to cm/pixels, then convert
                            # to microns by multiplying by 1000
                            mpp_x = (1/xres) * 1000
                            self.properties[OPS_MPP_X] = str(mpp_x)
                            log.debug(
                                f"Using MPP {mpp_x} per TIFF 'xres' field"
                                f" {loaded_image.get('xres')} and "
                                f"{loaded_image.get('resolution-unit')}"
                            )
                    else:
                        name = path_to_name(path)
                        log.warning(
                            f"Missing Microns-Per-Pixel (MPP) for {name}"
                        )
            except AttributeError:
                mpp = DEFAULT_JPG_MPP
                log.debug(f"Could not detect microns-per-pixel; using default {mpp}")
                self.properties[OPS_MPP_X] = mpp
            except UnidentifiedImageError:
                log.error(
                    f"PIL error; unable to read slide {path_to_name(path)}."
                )

        if OPS_LEVEL_COUNT in self.properties:
            self.level_count = int(self.properties[OPS_LEVEL_COUNT])
            # Calculate level metadata
            self.levels = []   # type: List[Dict[str, Any]]
            for lev in range(self.level_count):
                width = int(loaded_image.get(OPS_LEVEL_WIDTH(lev)))
                height = int(loaded_image.get(OPS_LEVEL_HEIGHT(lev)))
                downsample = float(loaded_image.get(OPS_LEVEL_DOWNSAMPLE(lev)))
                self.levels += [{
                    'dimensions': (width, height),
                    'width': width,
                    'height': height,
                    'downsample': downsample
                }]
        else:
            self.level_count = 1
            self.levels = [{
                    'dimensions': self.dimensions,
                    'width': self.dimensions[0],
                    'height': self.dimensions[1],
                    'downsample': 1
                }]
        self.level_downsamples = [lev['downsample'] for lev in self.levels]
        self.level_dimensions = [lev['dimensions'] for lev in self.levels]

    def best_level_for_downsample(
        self,
        downsample: float,
    ) -> int:
        '''Return lowest magnification level with a downsample level lower than
        the given target.

        Args:
            downsample (float): Ratio of target resolution to resolution
                at the highest magnification level. The downsample level of the
                highest magnification layer is equal to 1.
            levels (list(int), optional): Valid levels to search. Defaults to
                None (search all levels).

        Returns:
            int:    Optimal downsample level.'''
        max_downsample = 0
        for d in self.level_downsamples:
            if d < downsample:
                max_downsample = d
        try:
            max_level = self.level_downsamples.index(max_downsample)
        except Exception:
            log.debug(f"Error attempting to read level {max_downsample}")
            return 0
        return max_level

    def load_downsample_level(self, level: int) -> "vips.Image":
        downsampled_image = vips.Image.new_from_file(
            self.path,
            level=level,
            fail=True,
            access=vips.enums.Access.RANDOM
        )
        if self.cache_kw:
            downsampled_image = downsampled_image.tilecache(**self.cache_kw)
        self.loaded_downsample_levels.update({
            level: downsampled_image
        })
        return downsampled_image

    def get_downsampled_image(self, level: int) -> "vips.Image":
        '''Returns a VIPS image of a given downsample.'''
        if level in range(len(self.levels)):
            if level in self.loaded_downsample_levels:
                return self.loaded_downsample_levels[level]
            else:
                return self.load_downsample_level(level)
        else:
            return False

    def read_region(
        self,
        base_level_dim: Tuple[int, int],
        downsample_level: int,
        extract_size: Tuple[int, int]
    ) -> "vips.Image":
        """Extracts a region from the image at the given downsample level.

        Args:
            base_level_dim (Tuple[int, int]): Top-left location of the region
                to extract, using downsample layer coordinates (x, y)
            downsample_level (int): Downsample level to read.
            extract_size (Tuple[int, int]): Size of the region to read
                (width, height) using base layer coordinates.

        Returns:
            vips.Image: VIPS image.
        """
        base_level_x, base_level_y = base_level_dim
        extract_width, extract_height = extract_size
        downsample_factor = self.level_downsamples[downsample_level]
        downsample_x = int(base_level_x / downsample_factor)
        downsample_y = int(base_level_y / downsample_factor)
        image = self.get_downsampled_image(downsample_level)
        region = image.crop(
            downsample_x,
            downsample_y,
            extract_width,
            extract_height
        )
        return region

    def read_from_pyramid(
        self,
        top_left: Tuple[int, int],
        window_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None,
        target_downsample: Optional[float] = None,
    ) -> "vips.Image":
        """Reads a region from the image. Performance is accelerated by
        pyramid downsample layers, if available.

        Args:
            top_left (Tuple[int, int]): Top-left location of the region to
                extract, using base layer coordinates (x, y).
            window_size (Tuple[int, int]): Size of the region to read (width,
                height) using base layer coordinates.
            target_size (Tuple[int, int]): Resize the region to this target
                size (width, height).

        Returns:
            vips.Image: VIPS image. Dimensions will equal target_size unless
            the window includes an area of the image which is out of bounds.
            In this case, the returned image will be cropped.
        """
        if target_size is None and target_downsample is None:
            raise ValueError("Must supply either target_size or "
                             "target_downsample to read_from_pyramid()")
        if target_downsample is None:
            target_downsample = window_size[0] / target_size[0]

        ds_level = self.best_level_for_downsample(target_downsample)
        image = self.get_downsampled_image(ds_level)
        resize_factor = self.level_downsamples[ds_level] / target_downsample
        image = image.resize(resize_factor)

        if target_size is not None:
            return image.crop(
                int(top_left[0] / target_downsample),
                int(top_left[1] / target_downsample),
                min(target_size[0], image.width),
                min(target_size[1], image.height)
            )
        else:
            return image

class _JPGslideToVIPS(_VIPSWrapper):
    '''Wrapper for JPG files, which do not possess separate levels, to
    preserve openslide-like functions.'''

    def __init__(self, path: str, mpp: Optional[float] = None, cache_kw = None) -> None:
        self.path = path
        self.full_image = vips.Image.new_from_file(path)
        self.cache_kw = cache_kw if cache_kw else {}
        if not self.full_image.hasalpha():
            self.full_image = self.full_image.addalpha()
        self.properties = {}
        for field in self.full_image.get_fields():
            self.properties.update({field: self.full_image.get(field)})
        width = int(self.properties[OPS_WIDTH])
        height = int(self.properties[OPS_HEIGHT])
        self.dimensions = (width, height)
        self.level_count = 1
        self.loaded_downsample_levels = {
            0: self.full_image
        }
        # Calculate level metadata
        self.levels = [{
            'dimensions': (width, height),
            'width': width,
            'height': height,
            'downsample': 1,
        }]
        self.level_downsamples = [1]
        self.level_dimensions = [(width, height)]

        # MPP data
        if mpp is not None:
            log.debug(f"Setting MPP to {mpp}")
            self.properties[OPS_MPP_X] = mpp
        else:
            try:
                with Image.open(path) as img:
                    exif_data = img.getexif()
                    if TIF_EXIF_KEY_MPP in exif_data.keys():
                        _mpp = exif_data[TIF_EXIF_KEY_MPP]
                        log.debug(f"Using MPP {_mpp} per EXIF field {TIF_EXIF_KEY_MPP}")
                        self.properties[OPS_MPP_X] = _mpp
                    else:
                        raise AttributeError
            except AttributeError:
                mpp = DEFAULT_JPG_MPP
                log.debug(f"Could not detect microns-per-pixel; using default {mpp}")
                self.properties[OPS_MPP_X] = mpp
class ROI:
    '''Object container for ROI annotations.'''

    def __init__(self, name: str) -> None:
        self.name = name
        self.coordinates = []  # type: List[Tuple[int, int]]

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
        vips_cache: Optional[Dict[str, Any]] = None
    ) -> None:

        self.pb = pb
        self.name = path_to_name(path)
        self.shortname = sf.util._shortname(self.name)
        self.tile_px = tile_px
        self.enable_downsample = enable_downsample
        self.thumb_image = None  # type: Optional[Image.Image]
        self.stride_div = stride_div
        self.path = path
        self.qc_mask = None
        self.qc_mpp = None  # type: Optional[float]
        self.qc_method = None  # type: Optional[str]
        self.blur_burden = None  # type: Optional[float]
        self.roi_scale = 1  # type: float
        self.roi_method = None  # type: Optional[str]
        self.annPolys = []  # type: ignore
        self.filetype = sf.util.path_to_ext(path)
        self.__slide = None
        self._mpp_override = mpp
        self._vips_cache_kw = vips_cache

        # Initiate supported slide reader
        if not os.path.exists(path):
            raise errors.SlideNotFoundError(f"Could not find slide {path}.")
        if self.filetype.lower() not in sf.util.SUPPORTED_FORMATS:
            raise errors.SlideLoadError(
                f"{self.name}: unsupported filetype '{self.filetype}'"
            )

        # Collect basic slide information
        try:
            self.mpp = float(self.slide.properties[OPS_MPP_X])
        except KeyError:
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
        self.stride = self.extract_px // stride_div
        self.full_stride = self.full_extract_px // stride_div

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if '__slide' in state:
            state['__slide'] = None
        if '_BaseLoader__slide' in state:
            state['_BaseLoader__slide'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def _vips_wrapper(self) -> Any:
        if self.filetype.lower() in ('jpg', 'jpeg'):
            return _JPGslideToVIPS
        else:
            return _VIPSWrapper

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.slide.dimensions

    @property
    def properties(self) -> Dict:
        return self.slide.properties

    @property
    def vendor(self) -> Optional[str]:
        if OPS_VENDOR in self.slide.properties:
            return self.slide.properties[OPS_VENDOR]
        else:
            return None

    @property
    def slide(self) -> _VIPSWrapper:
        if self.__slide is not None:
            return self.__slide

        try:
            self.__slide = self._vips_wrapper(
                self.path,
                self._mpp_override,
                self._vips_cache_kw)
            return self.__slide  # type: ignore
        except vips.error.Error as e:
            raise errors.SlideLoadError(
                f"Error loading slide {self.shortname}: {e}"
            )

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
        self.qc_method = None
        log.debug(f'QC removed from slide {self.shortname}')

    def qc(
        self,
        method: str,
        blur_radius: int = 3,
        blur_threshold: float = 0.02,
        filter_threshold: float = 0.6,
        blur_mpp: float = 4
    ) -> Image.Image:
        """Applies quality control to a slide, performing filtering based on
        a whole-slide image thumbnail.

        'blur' method filters out blurry or out-of-focus slide sections.
        'otsu' method filters out background based on automatic saturation
        thresholding in the HSV colorspace.
        'both' applies both methods of filtering.

        Args:
            method (str): Quality control method, 'blur', 'otsu', or 'both'.
            blur_radius (int, optional): Blur radius.
            blur_threshold (float, optional): Blur threshold.
            filter_threshold (float, optional): Percent of a tile detected as
                background that will trigger a tile to be discarded.
                Defaults to 0.6.
            blur_mpp (float, optional): Size of WSI thumbnail on which to
                perform blur QC, in microns-per-pixel. Defaults to 4
                (equivalent magnification = 2.5 X).
        """

        if method not in ('blur', 'otsu', 'both'):
            raise errors.QCError(f"Unknown QC method {method}")
        starttime = time.time()

        self.qc_method = method

        # Blur QC must be performed at a set microns-per-pixel rather than
        # downsample level, as blur detection is much for sensitive to
        # effective magnification than Otsu's thresholding
        if method in ('blur', 'both'):
            thumb = self.thumb(mpp=blur_mpp)
            if thumb is None:
                raise errors.QCError(
                    f"Thumbnail error for slide {self.shortname}, QC failed"
                )
            thumb = np.array(thumb)
            if thumb.shape[-1] == 4:
                thumb = thumb[:, :, :3]
            gray = rgb2gray(thumb)
            img_laplace = np.abs(skimage.filters.laplace(gray))
            gaussian = skimage.filters.gaussian(img_laplace, sigma=blur_radius)
            blur_mask = gaussian <= blur_threshold
            lev = self.slide.level_count - 1
            self.qc_mask = blur_mask

        # Otsu's thresholding can be done on the lowest downsample level
        if method in ('otsu', 'both'):
            lev = self.slide.level_count - 1
            if self._vips_wrapper == _JPGslideToVIPS:
                otsu_thumb = vips.Image.new_from_file(self.path, fail=True)
            else:
                otsu_thumb = vips.Image.new_from_file(
                    self.path,
                    fail=True,
                    access=vips.enums.Access.RANDOM,
                    level=lev
                )
            try:
                otsu_thumb = vips2numpy(otsu_thumb)
            except vips.error.Error:
                raise errors.QCError(
                    f"Thumbnail error for slide {self.shortname}, QC failed"
                )
            if otsu_thumb.shape[-1] == 4:
                otsu_thumb = otsu_thumb[:, :, :3]

            # Only apply Otsu thresholding within ROI, if present
            if len(self.annPolys):
                ofact = self.roi_scale / self.slide.level_downsamples[lev]
                roi_mask = np.zeros((otsu_thumb.shape[0], otsu_thumb.shape[1]))
                scaled_polys = [
                    sa.scale(poly, xfact=ofact, yfact=ofact, origin=(0, 0))
                    for poly in self.annPolys
                ]
                roi_mask = rasterio.features.rasterize(
                    scaled_polys,
                    out_shape=otsu_thumb.shape[:2]
                )
                otsu_thumb = cv2.bitwise_or(
                    otsu_thumb,
                    otsu_thumb,
                    mask=roi_mask.astype(np.uint8)
                )
            hsv_img = cv2.cvtColor(otsu_thumb, cv2.COLOR_RGB2HSV)
            img_med = cv2.medianBlur(hsv_img[:, :, 1], 7)
            flags = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV
            _, otsu_mask = cv2.threshold(img_med, 0, 255, flags)
            otsu_mask = otsu_mask.astype(bool)
            self.qc_mask = otsu_mask

        # If performing both, ensure the mask sizes are equivalent (shrinks to
        # the size of the smaller mask - Otsu)
        if method == 'both':
            blur_mask = skimage.transform.resize(blur_mask, otsu_mask.shape)
            blur_mask = blur_mask.astype(bool)
            self.qc_mask = np.logical_or(blur_mask, otsu_mask)
            blur = np.count_nonzero(
                np.logical_and(
                    blur_mask,
                    np.logical_xor(blur_mask, otsu_mask)
                )
            )
            self.blur_burden = blur / (blur_mask.shape[0] * blur_mask.shape[1])
            log.debug(f"Blur burden: {self.blur_burden}")

        # Filter coordinates
        img = self.apply_qc_mask(self.qc_mask, filter_threshold=filter_threshold)
        dur = f'(time: {time.time()-starttime:.2f}s)'
        log.debug(f'QC ({method}) complete for slide {self.shortname} {dur}')
        return img

    def apply_qc_mask(
        self,
        mask: np.ndarray,
        filter_threshold: float = 0.6,
    ) -> Image:
        """Apply custom slide-level QC by filtering grid coordinates."""

        assert isinstance(mask, np.ndarray)
        assert len(mask.shape) == 2
        assert mask.dtype == bool

        downsample = self.dimensions[0] / mask.shape[1]
        qc_ratio = 1 / downsample
        qc_width = int(self.full_extract_px * qc_ratio)
        for i, (x, y, xi, yi) in enumerate(self.coord):  # type: ignore
            qc_x = int(x * qc_ratio)
            qc_y = int(y * qc_ratio)
            submask = mask[qc_y:(qc_y+qc_width), qc_x:(qc_x+qc_width)]
            if np.mean(submask) > filter_threshold:
                self.grid[xi, yi] = 0

        self.qc_mask = mask
        self.qc_mpp = self.mpp * downsample
        self.estimated_num_tiles = int(self.grid.sum())
        return Image.fromarray(img_as_ubyte(self.qc_mask))

    def square_thumb(self, width: int = 512) -> Image.Image:
        '''Returns a square thumbnail of the slide, with black bar borders.

        Args:
            width (int): Width/height of thumbnail in pixels.

        Returns:
            PIL image
        '''
        # Get thumbnail image and dimensions via fastest method available
        associated_images = self.slide.properties['slide-associated-images']
        if ('slide-associated-images' in self.slide.properties
           and 'thumbnail' in associated_images):
            vips_thumb = vips.Image.openslideload(
                self.slide.path,
                associated='thumbnail'
            )
        else:
            level = max(0, self.slide.level_count-2)
            vips_thumb = self.slide.get_downsampled_image(level)

        height = int(width / (vips_thumb.width / vips_thumb.height))
        np_thumb = vips2numpy(vips_thumb)
        thumb = Image.fromarray(np_thumb).resize((width, height))

        # Standardize to square with black borders as needed
        square_thumb = Image.new("RGB", (width, width))
        square_thumb.paste(thumb, (0, int((width-height)/2)))
        return square_thumb

    def thumb(
        self,
        mpp: Optional[float] = None,
        width: Optional[int] = None,
        coords: Optional[List[int]] = None,
        rois: bool = False,
        linewidth: int = 2,
        color: str = 'black'
    ) -> Image.Image:
        '''Returns PIL thumbnail of the slide.

        Args:
            mpp (float, optional): Microns-per-pixel, used to determine
                thumbnail size.
            width (int, optional): Alternatively, goal thumbnail width
                may be supplied.
            coords (list(int), optional): List of tile extraction coordinates
                to show as rectangles on the thumbnail, in [(x_center,
                y_center), ...] format. Defaults to None.

        Returns:
            PIL image
        '''

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

        # Get thumb via libvips & convert PIL Image
        if self.vendor and self.vendor == 'leica':
            # The libvips thumbnail function does not work appropriately
            # with Leica SCN images, so a downsample level must be
            # manually specified.
            thumbnail = vips.Image.new_from_file(
                self.path,
                fail=True,
                access=vips.enums.Access.RANDOM,
                level=self.slide.level_count-1
            )
        else:
            thumbnail = vips.Image.thumbnail(self.path, width)
        try:
            np_thumb = vips2numpy(thumbnail)
        except vips.error.Error as e:
            raise errors.SlideLoadError(f"Error loading slide thumbnail: {e}")
        image = Image.fromarray(np_thumb).resize((width, height))

        if coords:
            draw = ImageDraw.Draw(image)
            ratio = width / self.dimensions[0]
            wh = (self.full_extract_px * ratio) / 2
            for (x, y) in coords:  # type: ignore
                x, y = x * ratio * self.roi_scale, y * ratio * self.roi_scale  # type: ignore
                coords = (x-wh, y-wh, x+wh, y+wh)  # type: ignore
                draw.rectangle(coords, outline='black', width=2)
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

        # Assemble report DataFrame
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
        df = pd.DataFrame(df_dict)

        # Generate extraction report
        if report:
            report_data = dict(
                blur_burden=self.blur_burden,
                num_tiles=len(locations),
                qc_mask=self.qc_mask,
                locations=df
            )
            slide_report = SlideReport(
                sample_tiles,
                self.slide.path,
                data=report_data,
                thumb=self.thumb(
                    coords=locations,
                    rois=(self.roi_method != 'ignore')
                )
            )
            return slide_report
        else:
            return None

    def preview(self, rois: bool = True, **kwargs) -> Optional[Image.Image]:
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
            **kwargs
        )
        if generator is None:
            return self.thumb(rois=rois)
        locations = []
        for tile_dict in generator():
            locations += [tile_dict['loc']]
        return self.thumb(coords=locations, rois=rois)


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
        randomize_origin: bool = False,
        pb: Optional[Progress] = None,
        silent: Optional[bool] = None,
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
        """

        if silent is not None:
            warnings.warn("Argument `silent` is deprecated since 1.3 and will "
                          "be removed in 1.4. Please use `verbose` instead.")
            verbose = bool(not silent)

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
        self.rois = []  # type: List
        self.roi_method = roi_method
        self.randomize_origin = randomize_origin
        self.verbose = verbose

        # Look in ROI directory if available
        if roi_dir and exists(join(roi_dir, self.name + ".csv")):
            self.load_csv_roi(join(roi_dir, self.name + ".csv"))

        # Else try loading ROI from same folder as slide
        elif exists(self.name + ".csv"):
            self.load_csv_roi(path_to_name(path) + ".csv")
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
            self.load_csv_roi(mr)

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
            self.roi_method = 'ignore'
        elif len(self.rois) and roi_method == 'auto':
            log.debug(f"Slide {self.name}: extracting tiles from inside ROI.")
            self.roi_method = 'inside'

        # Build coordinate grid
        self._build_coord()

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
        image_dict = _wsi_extraction_worker(
            (x, y, grid_x, grid_y),
            SimpleNamespace(
                full_extract_px=self.full_extract_px,
                vips_wrapper=self._vips_wrapper,
                mpp_override=self._mpp_override,
                vips_cache=self._vips_cache_kw,
                roi_scale=self.roi_scale,
                rois=self.rois,
                grid=self.grid,
                downsample_level=self.downsample_level,
                path=self.path,
                extract_px=self.extract_px,
                tile_px=self.tile_px,
                full_stride=self.full_stride,
                normalizer=None,
                normalizer_source=None,
                whitespace_fraction=1,
                whitespace_threshold=1,
                grayspace_fraction=1,
                grayspace_threshold=1,
                img_format='numpy',
                yolo=False,
                draw_roi=False,
                dry_run=False,
            )
        )
        return image_dict['image']

    def _build_coord(self) -> None:
        '''Set up coordinate grid.'''

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
        if self.roi_method != 'ignore' and self.annPolys is not None:

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

            # Translate and scale the ROI polygons
            translated = [
                sa.translate(poly, x_offset/self.roi_scale, y_offset/self.roi_scale)
                for poly in self.annPolys
            ]
            scaled = [
                sa.scale(poly, xfact=xfact, yfact=yfact, origin=(0, 0))
                for poly in translated
            ]

            # Rasterize polygons to the size of the tile extraction grid
            self.roi_mask = rasterio.features.rasterize(
                scaled,
                out_shape=(self.grid.shape[1], self.grid.shape[0]),
                all_touched=False).astype(bool)
        else:
            self.roi_mask = None

        for yi, y in enumerate(y_range):
            for xi, x in enumerate(x_range):
                y = int(y)
                x = int(x)
                self.coord.append([x, y, xi, yi])

                # ROI filtering
                if self.roi_method != 'ignore' and self.annPolys is not None:
                    point_in_roi = self.roi_mask[yi, xi]
                    # If the extraction method is 'inside',
                    # skip the tile if it's not in an ROI
                    if (((self.roi_method == 'inside') and not point_in_roi)
                       or ((self.roi_method == 'outside') and point_in_roi)):
                        self.grid[xi, yi] = 0

        self.coord = np.array(self.coord)
        self.estimated_num_tiles = int(self.grid.sum())

    @property
    def shape(self):
        return self.grid.shape

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
        max_tiles: Optional[int] = None
    ) -> Optional[Callable]:
        """Builds tile generator to extract tiles from this slide.

        Args:
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

        w_args = SimpleNamespace(**{
            'full_extract_px': self.full_extract_px,
            'vips_wrapper': self._vips_wrapper,
            'mpp_override': self._mpp_override,
            'vips_cache': self._vips_cache_kw,
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
            'normalizer_source': normalizer_source,
            'whitespace_fraction': whitespace_fraction,
            'whitespace_threshold': whitespace_threshold,
            'grayspace_fraction': grayspace_fraction,
            'grayspace_threshold': grayspace_threshold,
            'img_format': img_format,
            'yolo': yolo,
            'draw_roi': draw_roi,
            'dry_run': dry_run
        })

        def generator():
            nonlocal pool
            should_close = False
            n_extracted = 0

            # Skip tiles filtered out with QC or ROI
            non_roi_coord = self.coord[
                self.grid[tuple(self.coord[:, 2:4].T)].astype(bool)
            ]
            if shard is not None:
                shard_idx, shard_count = shard
                sharded_coords = np.array_split(non_roi_coord, shard_count)
                non_roi_coord = sharded_coords[shard_idx]

            # Shuffle coordinates to randomize extraction order
            if shuffle:
                np.random.shuffle(non_roi_coord)

            # Set up worker pool
            if pool is None:
                if num_threads is None and num_processes is None:
                    # ThreadPool used by default due to escalating memory
                    # requirements when using multiprocessing
                    log.debug(f"Building generator ThreadPool({num_threads})")
                    _threads = os.cpu_count() if os.cpu_count() else 8
                    pool = mp.dummy.Pool(processes=_threads)
                    should_close = True
                elif num_threads is not None and num_threads > 1:
                    log.debug(f"Building generator ThreadPool({num_threads})")
                    _threads = os.cpu_count() if os.cpu_count() else 8
                    pool = mp.dummy.Pool(processes=num_threads)
                    should_close = True
                elif num_processes is not None and num_processes > 1:
                    log.debug(f"Building generator with Pool({num_processes})")
                    ctx = mp.get_context('spawn')
                    pool = ctx.Pool(processes=num_processes)
                    should_close = True
                else:
                    log.debug(f"Building generator without multithreading")
                    def _generator():
                        for c in non_roi_coord:
                            yield _wsi_extraction_worker(c, args=w_args)
                    i_mapped = _generator()
            else:
                log.debug("Building generator with a shared pool")
            if show_progress:
                pbar = Progress(transient=sf.getLoggingLevel() > 20)
                task = pbar.add_task('Extracting...', total=self.estimated_num_tiles)
                pbar.start()

            if pool is not None:
                if lazy_iter:
                    if max_tiles:
                        batch_size = min(pool._processes, max_tiles)
                    else:
                        batch_size = pool._processes
                    batched_coord = sf.util.batch(non_roi_coord, batch_size)
                    def _generator():
                        for batch in batched_coord:
                            yield from pool.imap(
                                partial(_wsi_extraction_worker, args=w_args),
                                batch
                            )
                    i_mapped = _generator()

                else:
                    i_mapped = pool.imap(
                        partial(_wsi_extraction_worker, args=w_args),
                        non_roi_coord,
                    )
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
            if show_progress:
                pbar.stop()
            if should_close:
                pool.close()
            name_msg = f'[green]{self.shortname}[/]'
            pos = len(non_roi_coord)
            num_msg = f'({n_extracted} tiles of {pos} possible)'
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
        color: str = 'black'
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

        Returns:
            PIL image
        """

        if rois:
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

        thumb = super().thumb(mpp=mpp, width=width, coords=coords)

        if rois:
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

    def load_csv_roi(self, path: str) -> int:
        '''Loads CSV ROI from a given path.'''

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

        # Load annotations as shapely.geometry objects
        if self.roi_method != 'ignore':
            self.annPolys = []
            for i, annotation in enumerate(self.rois):
                try:
                    poly = sg.Polygon(annotation.scaled_area(self.roi_scale))
                    self.annPolys += [poly]
                except ValueError:
                    log.warning(
                        f"Unable to use ROI {i} for [green]{self.name}[/]."
                        " At least 3 points required to create a shape."
                    )
            roi_area = sum([poly.area for poly in self.annPolys])
        else:
            roi_area = 1
        total_area = ((self.dimensions[0]/self.roi_scale)
                      * (self.dimensions[1]/self.roi_scale))
        self.roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

        return len(self.rois)

    def load_json_roi(self, path: str, scale: int = 10) -> int:
        '''Loads ROI from a JSON file.'''

        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], scale)
            self.rois.append(ROI(f"Object{len(self.rois)}"))
            self.rois[-1].add_shape(area_reduced)
        return len(self.rois)

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
            for image_dict in self.build_generator(
                shuffle=shuffle,
                show_progress=False,
                img_format=img_format,
                **kwargs
            )():
                if not (incl_slidenames or incl_loc):
                    yield torch.from_numpy(image_dict['image'])
                else:
                    to_return = {
                        'image_raw': torch.from_numpy(image_dict['image'])
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

        return tile_generator()


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
            (region_width, region_height)
        )
        extracted = vips2numpy(region)[:, :, :-1]
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
            if show_progress:
                pbar.stop()

        return generator
