"""Libvips slide-reading backend.

Requires: libvips (https://libvips.github.io/libvips/)
"""

import re
import cv2
import numpy as np
import slideflow as sf
import xml.etree.ElementTree as ET
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

# -----------------------------------------------------------------------------

SUPPORTED_BACKEND_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs',
                             'tiff', 'svslide', 'bif', 'jpg', 'jpeg', 'png',
                             'ome.tiff', 'ome.tif']

# -----------------------------------------------------------------------------

__vipsreader__ = None
__vipsreader_path__ = None
__vipsreader_args__ = None
__vipsreader_kwargs__ = None

# -----------------------------------------------------------------------------

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

    # Read a JPEG/PNG/TIFF image.
    if path_to_ext(path).lower() in ('jpg', 'jpeg', 'png'):
        reader = _SingleLevelVIPSReader(path, *args, **kwargs)

    # Read an OME-TIFF image.
    elif path.endswith('.ome.tif') or path.endswith('.ome.tiff'):
        reader = _OmeTiffVIPSReader(path, *args, **kwargs)

    # Read any other slide image.
    else:
        vips_image = vips.Image.new_from_file(path)
        if (vips_image.get('vips-loader') == 'tiffload'
            and 'n-pages' in vips_image.get_fields()
            and 'image-description' in vips_image.get_fields()
            and vips_image.get('image-description').startswith('Versa')):
            reader = _VersaVIPSReader(path, *args, **kwargs)
        elif (vips_image.get('vips-loader') == 'tiffload'
              and 'n-pages' in vips_image.get_fields()):
            reader = _MultiPageVIPSReader(path, *args, **kwargs)
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


def vips_padded_crop(image, x, y, width, height):
    bg = [255]
    if x+width <= image.width and y+height <= image.height:
        return image.crop(x, y, width, height)
    elif x+width > image.width and y+height <= image.height:
        cropped = image.crop(x, y, image.width-x, height)
        return cropped.gravity('west', width, height, background=bg)
    elif x+width <= image.width and y+height > image.height:
        cropped = image.crop(x, y, width, image.height-y)
        return cropped.gravity('north', width, height, background=bg)
    elif x+width > image.width and y+height > image.height:
        cropped = image.crop(x, y, image.width-x, image.height-y)
        return cropped.gravity('north-west', width, height, background=bg)
    else:
        raise errors.SlideError(
            "Unable to interpret padded crop for image {} at location {}, {} "
            "and width/height {}, {}.".format(
                image, x, y, width, height
            ))

def detect_mpp(
    path: str,
    loaded_image: Optional["vips.image.Image"] = None,
) -> Optional[float]:

    # --- Search VIPS fields ------------------------------------------

    # Load the image with Vips, if not already loaded
    if loaded_image is None:
        loaded_image = vips.Image.new_from_file(path)

    vips_fields = loaded_image.get_fields()

    if OPS_MPP_X in vips_fields:
        return float(loaded_image.get(OPS_MPP_X))

    # Search for MPP using SCN format
    if (sf.util.path_to_ext(path).lower() == 'svs'
            and 'image-description' in vips_fields):
        img_des = loaded_image.get('image-description')
        _mpp_matches = re.findall(r'(?<=MPP\s\=\s)0\.\d+', img_des)
        if len(_mpp_matches) and _mpp_matches[0] is not None:
            _mpp = _mpp_matches[0]
            log.debug(
                f"Using MPP {_mpp} from 'image-description' for SCN"
                "-converted SVS format"
            )
            return float(_mpp)

    # Search for MPP via TIFF EXIF field
    if (sf.util.path_to_ext(path).lower() in ('tif', 'tiff')
            and 'xres' in vips_fields):
        xres = loaded_image.get('xres')  # 4000.0
        if (xres == 4000.0
            and 'resolution-unit' in vips_fields
            and loaded_image.get('resolution-unit') == 'cm'):
            # xres = xres # though resolution from tiffinfo
            # says 40000 pixels/cm, for some reason the xres
            # val is 4000.0, so multiply by 10.
            # Convert from pixels/cm to cm/pixels, then convert
            # to microns by multiplying by 1000
            mpp_x = (1/xres) * 1000
            log.debug(
                f"Using MPP {mpp_x} per TIFF 'xres' field"
                f" {loaded_image.get('xres')} and "
                f"{loaded_image.get('resolution-unit')}"
            )
            return mpp_x

    # Search for MPP within OME-TIFF format
    if path.endswith('.ome.tif') or path.endswith('.ome.tiff'):
        xml_str = loaded_image.get('image-description')
        root = ET.fromstring(xml_str)
        try:
            root_ids = [i for i in range(len(root)) if 'Name' in root[i].attrib and root[i].attrib['Name'].endswith('_01')]
            assert len(root_ids) == 1
            root_id = root_ids[0]
            assert root[root_id].attrib['Name'].endswith('_01')
            assert root[root_id][3].tag.endswith('Pixels')
            mpp_x = float(root[root_id][3].attrib['PhysicalSizeX'])
            log.debug(
                f"Using MPP {mpp_x} per OME-TIFF PhysicalSizeX field"
            )
            return mpp_x
        except Exception as e:
            log.warning(f"Unable to read OME-TIFF PhysicalSizeX field. Error: {e}")
            pass

    # --- Search EXIF & tags ------------------------------------------
    try:
        with Image.open(path) as img:
            # Search exif data
            exif_data = img.getexif()
            if exif_data and TIF_EXIF_KEY_MPP in exif_data.keys():
                _mpp = exif_data[TIF_EXIF_KEY_MPP]
                log.debug(f"Using MPP {_mpp} per EXIF field {TIF_EXIF_KEY_MPP}")
                return float(_mpp)

            # Search image tags
            if hasattr(img, 'tag') and TIF_EXIF_KEY_MPP in img.tag.keys():
                _mpp = img.tag[TIF_EXIF_KEY_MPP][0]
                log.debug(
                    f"Using MPP {_mpp} per EXIF {TIF_EXIF_KEY_MPP}"
                )
                return float(_mpp)
    except UnidentifiedImageError:
        pass

    return None



# -----------------------------------------------------------------------------

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
        try:
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
        except vips.error.Error as e:
            log.warning(f"Error reading region at ({x}, {y}): {e}")
            return None
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
    try:
        region = slide.read_region(
            (x, y),
            args.downsample_level,
            (args.extract_px, args.extract_px)
        )
    except vips.error.Error as e:
        log.warning(f"Error reading region at ({x}, {y}): {e}")
        return None
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
        try:
            image = vips2numpy(region).astype(np.uint8)
        except vips.error.Error as e:
            log.warning('Error reading tile at ({}, {}): {}'.format(
                x_coord, y_coord, e
            ))
            return None

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

# -----------------------------------------------------------------------------


class _VIPSReader:

    has_levels = True

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        *,
        cache_kw: Optional[Dict[str, Any]] = None,
        ignore_missing_mpp: bool = False,
        pad_missing: bool = True,
        loaded_image: Optional["vips.Image"] = None,
        use_bounds: bool = False,
        transforms: Optional[List[int]] = None,
    ) -> None:
        """Libvips slide reader.

        Args:
            path (str): Path to slide.
            mpp (float, optional): Forcibly set microns-per-pixel.

        Keyword args:
            cache_kw (Dict, Optional): Optional keyword arguments for setting
                up a libvips cache. Keyword arguments are passed to
                ``pyvips.image.tilecache(**cache_kw)``. If not specified,
                tile cache is not used. Defaults to None.
            ignore_missing_mpp (bool): If MPP information cannot be found,
                do not raise an error. Defaults to False.
            pad_missing (bool): If an image crop is out-of-bounds for a slide
                (e.g., an edge tile), pad the image with black. If False,
                will raise an error if an out-of-bounds area is requested.
                Defaults to False.
            use_bounds (bool): If True, use the slide bounds to determine
                the slide dimensions. This will crop out unscanned white space.
                If a tuple of int, interprets the bounds as ``(top_left_x,
                top_left_y, width, height)``. If False, use the full slide
                dimensions. Defaults to False.
            transforms (list(int), optional): List of transforms to apply to
                the slide before establishing coordinate grid. Options include
                any combination of ``ROTATE_90_CLOCKWISE``,
                ``ROTATE_180_CLOCKWISE``, ``ROTATE_270_CLOCKWISE``,
                ``FLIP_HORIZONTAL``, and ``FLIP_VERTICAL``. Defaults to None.

        """
        self.path = path
        self.pad_missing = pad_missing
        self.cache_kw = cache_kw if cache_kw else {}
        self.loaded_downsample_levels = {}  # type: Dict[int, "vips.Image"]
        if loaded_image is None:
            loaded_image = vips.Image.new_from_file(path)
        self.vips_loader = loaded_image.get('vips-loader')
        if isinstance(transforms, int):
            transforms = [transforms]
        self.transforms = transforms

        # Load image properties
        self.properties = {}
        for field in loaded_image.get_fields():
            self.properties.update({field: loaded_image.get(field)})
        self.dimensions = self._detect_dimensions()
        # If Openslide MPP is not available, try reading from metadata
        if mpp is not None:
            log.debug(f"Setting MPP to {mpp}")
            self.properties[OPS_MPP_X] = mpp
        elif OPS_MPP_X not in self.properties.keys():
            log.debug("Microns-Per-Pixel (MPP) not found, Searching EXIF")
            mpp = detect_mpp(path, loaded_image)
            if mpp is not None:
                self.properties[OPS_MPP_X] = mpp
            elif ignore_missing_mpp:
                self.properties[OPS_MPP_X] = DEFAULT_JPG_MPP
                log.debug(f"Could not detect microns-per-pixel; using default "
                          f"{DEFAULT_JPG_MPP}")
            else:
                raise errors.SlideMissingMPPError(
                    f'Could not detect microns-per-pixel for slide: {path}'
                )

        # Check for bounding box
        if isinstance(use_bounds, (list, tuple, np.ndarray)):
            self.bounds = tuple(use_bounds)
        elif use_bounds and OPS_BOUNDS_X in self.properties:
            self.bounds = (
                int(self.properties[OPS_BOUNDS_X]),
                int(self.properties[OPS_BOUNDS_Y]),
                int(self.properties[OPS_BOUNDS_WIDTH]),
                int(self.properties[OPS_BOUNDS_HEIGHT])
            )
        else:
            self.bounds = None
        if self.bounds is not None:
            self.dimensions = (
                self.bounds[2],
                self.bounds[3]
            )

        # Load levels
        self._load_levels(loaded_image)

    @property
    def mpp(self):
        return self.properties[OPS_MPP_X]

    def has_mpp(self):
        return OPS_MPP_X in self.properties and self.properties[OPS_MPP_X] is not None

    def _detect_dimensions(self) -> Tuple[int, int]:
        return (
            int(self.properties[OPS_WIDTH]),
            int(self.properties[OPS_HEIGHT])
        )

    def _load_levels(self, vips_image: Optional["vips.Image"]):
        """Load downsample levels."""

        if vips_image is None:
            vips_image = vips.Image.new_from_file(self.path)

        if OPS_LEVEL_COUNT in self.properties:
            self.level_count = int(self.properties[OPS_LEVEL_COUNT])
            # Calculate level metadata
            self.levels = []   # type: List[Dict[str, Any]]
            for lev in range(self.level_count):
                width = int(vips_image.get(OPS_LEVEL_WIDTH(lev)))
                height = int(vips_image.get(OPS_LEVEL_HEIGHT(lev)))
                downsample = float(vips_image.get(OPS_LEVEL_DOWNSAMPLE(lev)))
                self.levels += [{
                    'dimensions': (width, height),
                    'width': width,
                    'height': height,
                    'downsample': downsample,
                    'level': lev
                }]
        elif 'n-pages' in self.properties and OPS_LEVEL_COUNT not in self.properties:
            log.debug("Attempting to read non-standard multi-page TIFF")
            # This is a multipage tiff without openslide metadata.
            # Ignore the last 2 pages, which per our experimentation,
            # are likely to be the slide label and image thumbnail.
            self.level_count = min(int(self.properties['n-pages']) - 3, 1)
            # Calculate level metadata
            self.levels = []
            for lev in range(self.level_count):
                temp_img = vips.Image.new_from_file(self.path, page=lev)
                width = int(temp_img.get('width'))
                height = int(temp_img.get('height'))
                downsample = float(int(self.properties[OPS_WIDTH]) / width)
                self.levels += [{
                    'dimensions': (width, height),
                    'width': width,
                    'height': height,
                    'downsample': downsample,
                    'level': lev
                }]
            self.levels = sorted(self.levels, key=lambda x: x['width'], reverse=True)

        else:
            self.level_count = 1
            self.levels = [{
                    'dimensions': self.dimensions,
                    'width': int(self.properties[OPS_WIDTH]),
                    'height': int(self.properties[OPS_HEIGHT]),
                    'downsample': 1,
                    'level': 0
                }]

        # Adjust for bounding boxes, if present
        if self.bounds is not None:
            for lev in range(self.level_count):
                self.levels[lev]['width'] = int(np.round(self.bounds[2] / self.levels[lev]['downsample']))
                self.levels[lev]['height'] = int(np.round(self.bounds[3] / self.levels[lev]['downsample']))
                self.levels[lev]['dimensions'] = (self.levels[lev]['width'], self.levels[lev]['height'])

        # Adjust for transforms, if present
        if self.transforms is not None:
            for transform in self.transforms:
                if transform in (ROTATE_90_CLOCKWISE, ROTATE_270_CLOCKWISE):
                    for lev in range(self.level_count):
                        self.levels[lev]['width'], self.levels[lev]['height'] = \
                            self.levels[lev]['height'], self.levels[lev]['width']
                        self.levels[lev]['dimensions'] = (self.levels[lev]['width'], self.levels[lev]['height'])
                    self.dimensions = (self.dimensions[1], self.dimensions[0])

        self.level_downsamples = [lev['downsample'] for lev in self.levels]
        self.level_dimensions = [lev['dimensions'] for lev in self.levels]

    def _load_downsample_level(self, level: int) -> "vips.Image":
        image = self.read_level(level=level)
        if self.cache_kw:
            image = image.tilecache(**self.cache_kw)  # type: ignore
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

    def coord_to_raw(self, x, y):
        """Convert coordinates from base layer to untransformed base layer.

        Convert base layer coordinates (after bounding box crop (``self.bounds``)
        and rotation/transforms (self.``transforms``)) to coordinates in the
        raw, untransformed base layer.

        """
        # Since bounds are applied before transforms, the first step
        # is to reverse the transforms to get the raw coordinates.
        if self.transforms is not None:
            for transform in self.transforms[::-1]:
                if transform == ROTATE_90_CLOCKWISE:
                    x, y = y, self.dimensions[0] - x
                if transform == ROTATE_180_CLOCKWISE:
                    x, y = self.dimensions[0] - x, self.dimensions[1] - y
                if transform == ROTATE_270_CLOCKWISE:
                    x, y = self.dimensions[1] - y, x
                if transform == FLIP_HORIZONTAL:
                    x = self.dimensions[0] - x
                if transform == FLIP_VERTICAL:
                    y = self.dimensions[1] - y

        # Then, apply the bounds
        if self.bounds is not None:
            x += self.bounds[0]
            y += self.bounds[1]

        return x, y

    def raw_to_coord(self, x, y):
        """Convert coordinates from untransformed base layer to base layer.

        Convert coordinates in the raw, untransformed base layer to
        base layer coordinates (after bounding box crop (``self.bounds``)
        and rotation/transforms (self.``transforms``)).

        """
        # First, apply the bounds
        if self.bounds is not None:
            x -= self.bounds[0]
            y -= self.bounds[1]

        # Then, apply the transforms
        if self.transforms is not None:
            for transform in self.transforms:
                if transform == ROTATE_90_CLOCKWISE:
                    x, y = self.dimensions[0] - y, x
                if transform == ROTATE_180_CLOCKWISE:
                    x, y = self.dimensions[0] - x, self.dimensions[1] - y
                if transform == ROTATE_270_CLOCKWISE:
                    x, y = y, self.dimensions[1] - x
                if transform == FLIP_HORIZONTAL:
                    x = self.dimensions[0] - x
                if transform == FLIP_VERTICAL:
                    y = self.dimensions[1] - y

        return x, y

    def bound_and_transform(self, image: vips.Image, level: int) -> vips.Image:
        """Apply bounding box crop and transforms to a VIPS image."""
        if self.bounds is not None:
            ds = self.level_downsamples[level]
            crop_bounds = (
                int(np.round(self.bounds[0] / ds)),
                int(np.round(self.bounds[1] / ds)),
                int(np.round(self.bounds[2] / ds)),
                int(np.round(self.bounds[3] / ds))
            )
            image = image.crop(*crop_bounds)
        if self.transforms is not None:
            for transform in self.transforms:
                if transform == ROTATE_90_CLOCKWISE:
                    image = image.rot90()
                if transform == ROTATE_180_CLOCKWISE:
                    image = image.rot180()
                if transform == ROTATE_270_CLOCKWISE:
                    image = image.rot270()
                if transform == FLIP_HORIZONTAL:
                    image = image.fliphor()
                if transform == FLIP_VERTICAL:
                    image = image.flipver()
        return image

    def read_level(
        self,
        fail: bool = True,
        access=vips.enums.Access.RANDOM,
        to_numpy: bool = False,
        level: Optional[int] = None,
        **kwargs
    ) -> Union[vips.Image, np.ndarray]:
        """Read a pyramid level."""

        if self.properties['vips-loader'] == 'tiffload' and level is not None:
            kwargs['page'] = self.levels[level]['level']
        elif level is not None:
            kwargs['level'] = level
        image = vips.Image.new_from_file(self.path, fail=fail, access=access, **kwargs)
        image = self.bound_and_transform(image, level=level)
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
        resize_factor: Optional[float] = None,
        pad_missing: Optional[bool] = None
    ) -> "vips.Image":
        """Extracts a region from the image at the given downsample level.

        Args:
            base_level_dim (Tuple[int, int]): Top-left location of the region
                to extract, using base layer coordinates (x, y)
            downsample_level (int): Downsample level to read.
            extract_size (Tuple[int, int]): Size of the region to read
                (width, height) using downsample layer coordinates.
            pad_missing (bool, optional): Pad missing regions with black.
                If None, uses the value of the `pad_missing` attribute.
                Defaults to None.

        Returns:
            vips.Image: VIPS image.
        """
        base_level_x, base_level_y = base_level_dim
        extract_width, extract_height = extract_size
        downsample_factor = self.level_downsamples[downsample_level]
        downsample_x = int(base_level_x / downsample_factor)
        downsample_y = int(base_level_y / downsample_factor)
        image = self.get_downsampled_image(downsample_level)
        crop_args = (
            downsample_x,
            downsample_y,
            extract_width,
            extract_height
        )
        if ((pad_missing is not None and pad_missing)
           or (pad_missing is None and self.pad_missing)):
            region = vips_padded_crop(image, *crop_args)
        else:
            region = image.crop(*crop_args)
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
        pad_missing: Optional[bool] = None
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
            pad_missing (bool, optional): Pad missing regions with black.
                If None, uses the value of the `pad_missing` attribute.
                Defaults to None.

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
        crop_args = (
            int(top_left[0] / target_downsample),
            int(top_left[1] / target_downsample),
            min(target_size[0], image.width),
            min(target_size[1], image.height)
        )
        if ((pad_missing is not None and pad_missing)
           or (pad_missing is None and self.pad_missing)):
            image = vips_padded_crop(image, *crop_args)
        else:
            image = image.crop(*crop_args)
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

        if ((OPS_VENDOR in self.properties and self.properties[OPS_VENDOR] == 'leica')
           or (self.vips_loader == 'tiffload') or self.bounds or self.transforms):
            thumbnail = self.read_level(fail=fail, access=access, **kwargs)
        else:
            thumbnail = vips.Image.thumbnail(self.path, width)
        try:
            return vips2numpy(thumbnail)
        except vips.error.Error as e:
            raise sf.errors.SlideLoadError(f"Error loading slide thumbnail: {e}")


class _MultiPageVIPSReader(_VIPSReader):

    def _load_levels(self, vips_image: Optional["vips.Image"]):
        """Load downsample levels."""
        log.debug("Attempting to read levels from non-standard multi-page TIFF")
        # This is a multipage tiff without openslide metadata.
        # Ignore the last 2 pages, which per our experimentation,
        # are likely to be the slide label and image thumbnail.
        self.level_count = int(self.properties['n-pages'])
        # Calculate level metadata
        self.levels = []
        for lev in range(self.level_count):
            temp_img = vips.Image.new_from_file(self.path, page=lev)
            width = int(temp_img.get('width'))
            height = int(temp_img.get('height'))
            downsample = float(int(self.properties[OPS_WIDTH]) / width)
            self.levels += [{
                'dimensions': (width, height),
                'width': width,
                'height': height,
                'downsample': downsample,
                'level': lev
            }]
        self.levels = sorted(self.levels, key=lambda x: x['width'], reverse=True)
        log.debug(f"Read {self.level_count} levels.")
        self.level_downsamples = [lev['downsample'] for lev in self.levels]
        self.level_dimensions = [lev['dimensions'] for lev in self.levels]


class _OmeTiffVIPSReader(_VIPSReader):

    def __init__(self, *args, **kwargs):
        self.page_labels = None
        self._num_pyramid_levels = None
        super().__init__(*args, **kwargs)

    @property
    def num_pyramid_levels(self):
        if self._num_pyramid_levels is not None:
            return self._num_pyramid_levels
        else:
            # Determine the number of pyramid levels from XML
            str_xml = self.properties['image-description']
            root = ET.fromstring(str_xml)
            possible_anns = [child for child in root if child.tag.endswith('StructuredAnnotations')]
            if len(possible_anns) == 0:
                raise errors.SlideError("Could not find pyramid levels in OME-TIFF XML.")
            if len(possible_anns) > 1:
                raise errors.SlideError("Could not interpret OME-TIFF XML; found multiple 'StructuredAnnotations' fields.")
            ann = possible_anns[0]
            page_id = self.get_page_by_label('main')
            resolution_pyramid = [child for child in ann if child.tag.endswith('MapAnnotation') and child.attrib['ID'] == f'Annotation:Resolution:{page_id}']
            if len(resolution_pyramid) == 0:
                raise errors.SlideError("Could not find pyramid levels in OME-TIFF XML.")
            if len(resolution_pyramid) > 1:
                raise errors.SlideError("Could not interpret OME-TIFF XML; found multiple resolution pyramids.")
            self._num_pyramid_levels = len(resolution_pyramid[0][0])
            return self._num_pyramid_levels

    def build_page_labels(self):
        """Build page labels from OME-TIFF XML."""
        xml_str = self.properties['image-description']
        root = ET.fromstring(xml_str)
        self.page_labels = {
            (child.attrib['Name'] if not child.attrib['Name'].endswith('_01') else 'main'): int(child.attrib['ID'].split(':')[-1])
            for child in root
            if 'ID' in child.attrib and 'Image' in child.attrib['ID']
        }

    def get_page_by_label(self, label: str) -> int:
        """Return page number by label."""
        if self.page_labels is None:
            self.build_page_labels()
        if label in self.page_labels:
            return self.page_labels[label]
        else:
            raise ValueError(f"Unknown page label {label}")

    def _load_levels(self, vips_image: Optional["vips.Image"]):
        """Load downsample levels."""
        log.debug("Attempting to read levels from OME-TIFF")

        # This is a multipage tiff split into RGB channels.
        # The first (3) page(s) are the slide label (RGB channels).
        if not self.properties['n-pages'] % 3 == 0:
            raise errors.SlideError(
                "Unexpected number of pages in OME-TIFF. Expected a multiple "
                f"of 3, but found {self.properties['n-pages']}."
            )
        self.level_count = self.num_pyramid_levels
        main_page = self.get_page_by_label('main')
        # Calculate level metadata
        self.levels = []
        for lev in range(self.level_count):
            try:
                temp_img = vips.Image.new_from_file(self.path, page=main_page*3, subifd=lev-1)
            except vips.error.Error as e:
                if 'subifd' in str(e) and 'out of range' in str(e):
                    # XML may have more levels than the actual image
                    self.level_count = lev
                    break
            width = int(temp_img.get('width'))
            height = int(temp_img.get('height'))
            downsample = float(int(self.properties[OPS_WIDTH]) / width)
            self.levels += [{
                'dimensions': (width, height),
                'width': width,
                'height': height,
                'downsample': downsample,
                'level': lev
            }]
        self.levels = sorted(self.levels, key=lambda x: x['width'], reverse=True)
        log.debug(f"Read {self.level_count} levels.")
        self.level_downsamples = [lev['downsample'] for lev in self.levels]
        self.level_dimensions = [lev['dimensions'] for lev in self.levels]

        # Update width and height
        width, height = self.levels[0]['dimensions']
        self.properties[OPS_WIDTH] = width
        self.properties[OPS_HEIGHT] = height
        self.dimensions = (width, height)

        # Update downsamples
        for lev in range(self.level_count):
            self.levels[lev]['downsample'] = float(int(self.properties[OPS_WIDTH]) / self.levels[lev]['width'])
            self.level_downsamples[lev] = self.levels[lev]['downsample']

    def thumbnail(
        self,
        width: int = 512,
        fail: bool = True,
        access = vips.enums.Access.RANDOM,
        level: Optional[int] = 2,
        **kwargs
    ) -> np.ndarray:
        """Return thumbnail of slide as numpy array."""
        thumbnail = self.read_level(fail=fail, access=access, level=level, **kwargs)
        try:
            thumb = vips2numpy(thumbnail)
            return thumb
        except vips.error.Error as e:
            raise sf.errors.SlideLoadError(f"Error loading slide thumbnail: {e}")

    def read_level(
        self,
        fail: bool = True,
        access=vips.enums.Access.RANDOM,
        to_numpy: bool = False,
        level: int = 0,
        **kwargs
    ) -> Union[vips.Image, np.ndarray]:
        """Read a pyramid level

        Stacks RGB pages in the OME-TIFF file format into a single image.
        """
        main_page = self.get_page_by_label('main')
        r, g, b = [
            vips.Image.new_from_file(
                self.path,
                fail=fail,
                access=access,
                page=(main_page * 3) + n,
                subifd=level-1,
                **kwargs)
            for n in range(3)
        ]
        image = r.bandjoin([g, b])
        image = self.bound_and_transform(image, level=level)
        if to_numpy:
            return vips2numpy(image)
        else:
            return image


class _VersaVIPSReader(_VIPSReader):

    def _load_levels(self, vips_image: Optional["vips.Image"]):
        """Load downsample levels."""
        log.debug("Attempting to read levels from Versa multi-page image")
        # This is a multipage tiff without openslide metadata.
        # Ignore the last 2 pages, which per our experimentation,
        # are likely to be the slide label and image thumbnail.
        all_lev = self.level_count = max(int(self.properties['n-pages']) - 2, 1)
        # Calculate level metadata
        self.levels = []
        for lev in range(self.level_count):
            temp_img = vips.Image.new_from_file(self.path, page=lev)
            width = int(temp_img.get('width'))
            height = int(temp_img.get('height'))
            downsample = float(int(self.properties[OPS_WIDTH]) / width)
            self.levels += [{
                'dimensions': (width, height),
                'width': width,
                'height': height,
                'downsample': downsample,
                'level': lev
            }]
        self.levels = sorted(self.levels, key=lambda x: x['width'], reverse=True)
        log.debug(f"Read {self.level_count} of {all_lev} levels.")
        self.level_downsamples = [lev['downsample'] for lev in self.levels]
        self.level_dimensions = [lev['dimensions'] for lev in self.levels]

    def thumbnail(self, width: int = 512, *args, **kwargs) -> np.ndarray:
        """Return thumbnail of slide as numpy array."""
        vips_image = vips.Image.new_from_file(self.path, page=1)
        np_image = vips2numpy(vips_image)
        width_height_ratio = np_image.shape[1] / np_image.shape[0]
        height = int(width / width_height_ratio)
        return cv2.resize(np_image, (width, height))


class _SingleLevelVIPSReader(_VIPSReader):
    '''Wrapper for JPG files, which do not possess separate levels, to
    preserve openslide-like functions.'''

    has_levels = False

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        cache_kw = None,
        ignore_missing_mpp: bool = True,
        pad_missing: bool = True,
        loaded_image: Optional["vips.Image"] = None,
        use_bounds: bool = False,           # Not used for JPEG images.
        transforms: Optional[Any] = None,   # Not used for JPEG images.
    ) -> None:
        self.bounds = None
        self.transforms = transforms
        self.path = path
        self.pad_missing = pad_missing
        if loaded_image is None:
            loaded_image = vips.Image.new_from_file(path)
        self.cache_kw = cache_kw if cache_kw else {}
        if not loaded_image.hasalpha():
            loaded_image = loaded_image.addalpha()
        self.properties = {}
        for field in loaded_image.get_fields():
            self.properties.update({field: loaded_image.get(field)})
        width = int(self.properties[OPS_WIDTH])
        height = int(self.properties[OPS_HEIGHT])
        self.dimensions = (width, height)
        self.vips_loader = loaded_image.get('vips-loader')
        self.level_count = 1
        self.loaded_downsample_levels = {
            0: loaded_image
        }
        # Calculate level metadata
        self.levels = [{
            'dimensions': (width, height),
            'width': width,
            'height': height,
            'downsample': 1,
            'level': 0
        }]
        self.level_downsamples = [1]
        self.level_dimensions = [(width, height)]

        # If MPP is not provided, try reading from metadata
        if mpp is not None:
            log.debug(f"Setting MPP to {mpp}")
            self.properties[OPS_MPP_X] = mpp
        elif OPS_MPP_X not in self.properties.keys():
            log.debug("Microns-Per-Pixel (MPP) not found, Searching EXIF")
            mpp = detect_mpp(path, loaded_image)
            if mpp is not None:
                self.properties[OPS_MPP_X] = mpp
            elif ignore_missing_mpp:
                self.properties[OPS_MPP_X] = DEFAULT_JPG_MPP
                log.debug(f"Could not detect microns-per-pixel; using default "
                          f"{DEFAULT_JPG_MPP}")
            else:
                raise errors.SlideMissingMPPError(
                    f'Could not detect microns-per-pixel for slide: {path}'
                )

    def _load_downsample_level(self, level: int = 0) -> "vips.Image":
        if level:
            raise ValueError(f"_SingleLevelVipsReader does not support levels")
        image = self.read_level()
        if self.cache_kw:
            image = image.tilecache(**self.cache_kw)  # type: ignore
        self.loaded_downsample_levels.update({
            level: image
        })
        return image

    def read_level(
        self,
        fail: bool = True,
        access=vips.enums.Access.RANDOM,
        to_numpy: bool = False,
        level: Optional[int] = None,
        **kwargs
    ) -> Union[vips.Image, np.ndarray]:
        """Read a pyramid level."""
        if level:
            raise ValueError(f"_SingleLevelVipsReader does not support levels")
        return super().read_level(
            fail=fail,
            access=access,
            to_numpy=to_numpy,
            **kwargs
        )
