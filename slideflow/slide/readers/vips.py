"""Libvips slide-reading backend.

Requires: libvips (https://libvips.github.io/libvips/)
"""

import re
from typing import (Any, Dict, List, Optional, Tuple)

import numpy as np
import slideflow as sf
from PIL import Image, UnidentifiedImageError
from slideflow.util import log, path_to_name  # noqa F401
from slideflow.slide.utils import *

from rich.progress import Progress


try:
    import pyvips as vips
except (ModuleNotFoundError, OSError) as e:
    log.error("Unable to load vips; slide processing will be unavailable. "
              f"Error raised: {e}")


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


def vips_thumbnail(
    path: str,
    width: int = 512,
    fail: bool = True,
    leica_scn: bool = False,
    access = vips.enums.Access.RANDOM,
    **kwargs
) -> np.ndarray:

    if leica_scn:
        thumbnail = vips.Image.new_from_file(path, fail=fail, access=access, **kwargs)
    else:
        thumbnail = vips.Image.thumbnail(path, width)
    try:
        return vips2numpy(thumbnail)
    except vips.error.Error as e:
        raise sf.errors.SlideLoadError(f"Error loading slide thumbnail: {e}")


class _VIPSReader:

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


class _JPGVIPSReader(_VIPSReader):
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