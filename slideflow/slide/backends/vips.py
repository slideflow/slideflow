"""Libvips slide-reading backend.

Requires: libvips (https://libvips.github.io/libvips/)
"""

import re
import cv2
import numpy as np
import slideflow as sf
from types import SimpleNamespace
from PIL import Image, UnidentifiedImageError
from typing import (Any, Dict, List, Optional, Tuple, Union)
from slideflow import errors
from slideflow.util import log, path_to_name, path_to_ext  # noqa F401
from slideflow.slide.utils import *

try:
    import pyvips as vips
except (ModuleNotFoundError, OSError) as e:
    log.error("Unable to load vips; slide processing will be unavailable. "
              f"Error raised: {e}")


__vipsreader__ = None
__vipsreader_path__ = None
__vipsreader_args__ = None
__vipsreader_kwargs__ = None


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


def get_libvips_reader(path: str, *args, **kwargs):
    global __vipsreader__, __vipsreader_path__, __vipsreader_args__, __vipsreader_kwargs__

    # Return from buffer, if present.
    if (__vipsreader_path__ == path
       and __vipsreader_args__ == args
       and __vipsreader_kwargs__ == kwargs):
        return __vipsreader__

    # Read a JPEG image.
    if path_to_ext(path).lower() in ('jpg', 'jpeg', 'png'):
        reader = _JPGVIPSReader(path, *args, **kwargs)

    # Read a slide image.
    else:
        reader = _VIPSReader(path, *args, **kwargs)

    # Buffer args and return.
    __vipsreader_path__ = path
    __vipsreader_args__ = args
    __vipsreader_kwargs__ = kwargs
    __vipsreader__ = reader
    return reader


def vips2numpy(
    vi: "vips.Image",
) -> np.ndarray:
    '''Converts a VIPS image into a numpy array'''
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def vips2jpg(
    vi: "vips.Image",
) -> np.ndarray:
    '''Converts a VIPS image into a numpy array'''
    return vi.jpegsave_buffer(Q=95)


def vips2png(
    vi: "vips.Image",
) -> np.ndarray:
    '''Converts a VIPS image into a numpy array'''
    return vi.pngsave_buffer()


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


def tile_worker(
    c: List[int],
    args: SimpleNamespace
) -> Optional[Union[str, Dict]]:
    '''Multiprocessing worker for WSI. Extracts tile at given coordinates.'''

    if args.has_segmentation:
        c, tile_mask = c
        (x, y, grid_x), grid_y = c, 0
    else:
        tile_mask = None
        x, y, grid_x, grid_y = c

    x_coord = int(x + args.full_extract_px / 2)
    y_coord = int(y + args.full_extract_px / 2)

    # If downsampling is enabled, read image from highest level
    # to perform filtering; otherwise filter from our target level
    slide = get_libvips_reader(args.path, args.mpp_override, **args.reader_kwargs)
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

    # If using a segmentation mask, resize mask to match the tile size.
    if tile_mask is not None:
        tile_mask = cv2.resize(
            tile_mask,
            (args.tile_px, args.tile_px),
            interpolation=cv2.INTER_NEAREST)

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

    # Apply segmentation mask
    if tile_mask is not None:
        vips_mask = vips.Image.new_from_array(tile_mask)
        region = region.multiply(vips_mask)

    if args.img_format != 'numpy':
        if args.img_format == 'png':
            image = region.pngsave_buffer()
        elif args.img_format in ('jpg', 'jpeg'):
            image = region.jpegsave_buffer(Q=95)
        else:
            raise ValueError(f"Unknown image format {args.img_format}")

        # Apply normalization
        if args.normalizer:
            try:
                if args.img_format == 'png':
                    image = args.normalizer.png_to_png(image)
                elif args.img_format in ('jpg', 'jpeg'):
                    image = args.normalizer.jpeg_to_jpeg(image)
                else:
                    raise ValueError(f"Unknown image format {args.img_format}")
            except Exception as e:
                # The image could not be normalized,
                # which happens when a tile is primarily one solid color
                log.debug(f'Normalization error: {e}')
                return None
    else:
        # Read regions into memory and convert to numpy arrays
        image = vips2numpy(region).astype(np.uint8)

        # Apply normalization
        if args.normalizer:
            try:
                image = args.normalizer.rgb_to_rgb(image)
            except Exception:
                # The image could not be normalized,
                # which happens when a tile is primarily one solid color
                return None

    # Include ROI / bounding box processing.
    # Used to visualize ROIs on extracted tiles, or to generate YoloV5 labels.
    if args.yolo or args.draw_roi:
        coords, boxes, yolo_anns = roi_coords_from_image(c, args)
    if args.draw_roi:
        image = draw_roi(image, coords)

    return_dict.update({'image': image})
    if args.yolo:
        return_dict.update({'yolo': yolo_anns})
    return return_dict


class _VIPSReader:

    has_levels = True

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        cache_kw: Optional[Dict[str, Any]] = None,
        ignore_missing_mpp: bool = False
    ) -> None:
        '''Wrapper for Libvips to preserve cross-compatible functionality.'''

        self.path = path
        self.cache_kw = cache_kw if cache_kw else {}
        self.loaded_downsample_levels = {}  # type: Dict[int, "vips.Image"]
        loaded_image = self._load_downsample_level(0)

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
                if ignore_missing_mpp:
                    mpp = DEFAULT_JPG_MPP
                    log.debug(f"Could not detect microns-per-pixel; using default {mpp}")
                    self.properties[OPS_MPP_X] = mpp
                else:
                    raise errors.SlideMissingMPPError(
                        f'Could not detect microns-per-pixel for slide: {path}'
                    )
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

    @property
    def mpp(self):
        return self.properties[OPS_MPP_X]

    def _load_downsample_level(self, level: int) -> "vips.Image":
        image = self.read_level(level=level)
        if self.cache_kw:
            image = image.tilecache(**self.cache_kw)
        self.loaded_downsample_levels.update({
            level: image
        })
        return image

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

    def get_downsampled_image(self, level: int) -> "vips.Image":
        '''Returns a VIPS image of a given downsample.'''
        if level in range(len(self.levels)):
            if level in self.loaded_downsample_levels:
                return self.loaded_downsample_levels[level]
            else:
                return self._load_downsample_level(level)
        else:
            return False

    def read_level(
        self,
        fail: bool = True,
        access=vips.enums.Access.RANDOM,
        to_numpy: bool = False,
        **kwargs
    ) -> Union[vips.Image, np.ndarray]:
        """Read a pyramid level."""
        image = vips.Image.new_from_file(self.path, fail=fail, access=access, **kwargs)
        if to_numpy:
            return vips2numpy(image)
        else:
            return image

    def read_region(
        self,
        base_level_dim: Tuple[int, int],
        downsample_level: int,
        extract_size: Tuple[int, int],
        convert: Optional[str] = None,
        flatten: bool = False,
        resize_factor: Optional[float] = None
    ) -> "vips.Image":
        """Extracts a region from the image at the given downsample level.

        Args:
            base_level_dim (Tuple[int, int]): Top-left location of the region
                to extract, using base layer coordinates (x, y)
            downsample_level (int): Downsample level to read.
            extract_size (Tuple[int, int]): Size of the region to read
                (width, height) using downsample layer coordinates.

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
        # Final conversions
        if flatten and region.bands == 4:
            region = region.flatten()
        if resize_factor is not None:
            region = region.resize(resize_factor)
        if convert and convert.lower() in ('jpg', 'jpeg'):
            return vips2jpg(region)
        elif convert and convert.lower() == 'png':
            return vips2png(region)
        elif convert == 'numpy':
            return vips2numpy(region)
        else:
            return region

    def read_from_pyramid(
        self,
        top_left: Tuple[int, int],
        window_size: Tuple[int, int],
        target_size: Tuple[int, int],
        convert: Optional[str] = None,
        flatten: bool = False,
    ) -> "vips.Image":
        """Reads a region from the image using base layer coordinates.
        Performance is accelerated by pyramid downsample layers, if available.

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
        target_downsample = window_size[0] / target_size[0]
        ds_level = self.best_level_for_downsample(target_downsample)
        image = self.get_downsampled_image(ds_level)
        resize_factor = self.level_downsamples[ds_level] / target_downsample
        image = image.resize(resize_factor)
        image = image.crop(
            int(top_left[0] / target_downsample),
            int(top_left[1] / target_downsample),
            min(target_size[0], image.width),
            min(target_size[1], image.height)
        )
        # Final conversions
        if flatten and image.bands == 4:
            image = image.flatten()
        if convert and convert.lower() in ('jpg', 'jpeg'):
            return vips2jpg(image)
        elif convert and convert.lower() == 'png':
            return vips2png(image)
        elif convert == 'numpy':
            return vips2numpy(image)
        else:
            return image

    def thumbnail(
        self,
        width: int = 512,
        fail: bool = True,
        access = vips.enums.Access.RANDOM,
        **kwargs
    ) -> np.ndarray:
        """Return thumbnail of slide as numpy array."""

        if (OPS_VENDOR in self.properties and self.properties[OPS_VENDOR] == 'leica'):
            thumbnail = self.read_level(fail=fail, access=access, **kwargs)
        else:
            thumbnail = vips.Image.thumbnail(self.path, width)
        try:
            return vips2numpy(thumbnail)
        except vips.error.Error as e:
            raise sf.errors.SlideLoadError(f"Error loading slide thumbnail: {e}")

class _JPGVIPSReader(_VIPSReader):
    '''Wrapper for JPG files, which do not possess separate levels, to
    preserve openslide-like functions.'''

    has_levels = False

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        cache_kw = None,
        ignore_missing_mpp: bool = True
    ) -> None:
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
                if ignore_missing_mpp:
                    mpp = DEFAULT_JPG_MPP
                    log.debug(f"Could not detect microns-per-pixel; using default {mpp}")
                    self.properties[OPS_MPP_X] = mpp
                else:
                    raise errors.SlideMissingMPPError(
                        f'Could not detect microns-per-pixel for slide: {path}'
                    )
