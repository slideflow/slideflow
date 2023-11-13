"""cuCIM slide-reading backend.

Requires: cuCIM (...)
"""

import cv2
import numpy as np

from types import SimpleNamespace
from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING
from slideflow.util import log
from skimage.transform import resize
from skimage.util import img_as_float32
from skimage.color import rgb2hsv
from slideflow.slide.utils import *

if TYPE_CHECKING:
    from cucim import CuImage

# -----------------------------------------------------------------------------

SUPPORTED_BACKEND_FORMATS = ['svs', 'tif', 'tiff']

# -----------------------------------------------------------------------------

__cv2_resize__ = True
__cuimage__ = None
__cuimage_path__ = None

# -----------------------------------------------------------------------------

def get_cucim_reader(path: str, *args, **kwargs):
    return _cuCIMReader(path, *args, **kwargs)


def cucim2numpy(img: "CuImage") -> np.ndarray:
    return ((img_as_float32(np.asarray(img))) * 255).astype(np.uint8)


def numpy2jpg(img: np.ndarray) -> str:
    if img.shape[-1] == 4:
        img = img[:, :, 0:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".jpg", img)[1].tobytes()   # Default quality = 95%


def numpy2png(img: np.ndarray) -> str:
    if img.shape[-1] == 4:
        img = img[:, :, 0:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", img)[1].tobytes()


def cucim2jpg(img: "CuImage") -> str:
    img = cucim2numpy(img)
    return numpy2jpg(img)


def cucim2png(img: "CuImage") -> str:
    img = cucim2numpy(img)
    return numpy2png(img)


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
    slide = get_cucim_reader(args.path, args.mpp_override, **args.reader_kwargs)
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
            ws_fraction = np.mean((np.mean(cucim2numpy(filter_region), axis=-1) > args.whitespace_threshold))
            if (ws_fraction > args.whitespace_fraction
               and args.whitespace_fraction != FORCE_CALCULATE_WHITESPACE):
                return None

        # Perform grayspace filtering [Libvips]
        if args.grayspace_fraction < 1:
            hsv_region = rgb2hsv(np.asarray(filter_region))
            gs_fraction = np.mean(hsv_region[:, :, 1] < args.grayspace_threshold)
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

    # cuCIM resize
    if not __cv2_resize__:
        if int(args.tile_px) != int(args.extract_px):
            region = resize(np.asarray(region), (args.tile_px, args.tile_px))

    region = cucim2numpy(region)

    # cv2 resize
    if __cv2_resize__:
        if int(args.tile_px) != int(args.extract_px):
            region = cv2.resize(region, (args.tile_px, args.tile_px))

    assert(region.shape[0] == region.shape[1] == args.tile_px)

    # Remove the alpha channel and convert to RGB
    if region.shape[-1] == 4:
        region = region[:, :, 0:3]

    # Apply segmentation mask
    if tile_mask is not None:
        region[tile_mask == 0] = (0, 0, 0)

    # Apply normalization
    if args.normalizer:
        try:
            region = args.normalizer.rgb_to_rgb(region)
        except Exception:
            # The image could not be normalized,
            # which happens when a tile is primarily one solid color
            return None

    if args.img_format != 'numpy':
        image = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
        # Default image quality for JPEG is 95%
        image = cv2.imencode("."+args.img_format, image)[1].tobytes()
    else:
        image = region

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


class _cuCIMReader:

    has_levels = True

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        cache_kw: Optional[Dict[str, Any]] = None,
        num_workers: int = 0,
        ignore_missing_mpp: bool = True,
        use_bounds: bool = False,  #TODO: Not yet implemented
    ):
        '''Wrapper for cuCIM reader to preserve cross-compatible functionality.'''
        global __cuimage__, __cuimage_path__

        from cucim import CuImage

        self.path = path
        self.cache_kw = cache_kw if cache_kw else {}
        self.loaded_downsample_levels = {}  # type: Dict[int, "CuImage"]
        if path == __cuimage_path__:
            self.reader = __cuimage__
        else:
            __cuimage__ = self.reader = CuImage(path)
            __cuimage_path__ = path
        self.num_workers = num_workers
        self._mpp = None

        # Check for Microns-per-pixel (MPP)
        if mpp is not None:
            log.debug(f"Manually setting MPP to {mpp}")
            self._mpp = mpp
        for prop_key in self.metadata:
            if 'MPP' in self.metadata[prop_key]:
                self._mpp = self.metadata[prop_key]['MPP']
            elif 'DICOM_PIXEL_SPACING' in self.metadata[prop_key]:
                ps = self.metadata[prop_key]['DICOM_PIXEL_SPACING'][0]
                self._mpp = ps * 1000  # Convert from millimeters -> microns
        if not self.mpp:
            log.warn("Unable to auto-detect microns-per-pixel (MPP).")

        # Pyramid layers
        self.dimensions = tuple(self.properties['shape'][0:2][::-1])
        self.levels = []
        for lev in range(self.level_count):
            self.levels.append({
                'dimensions': self.level_dimensions[lev],
                'width': self.level_dimensions[lev][0],
                'height': self.level_dimensions[lev][1],
                'downsample': self.level_downsamples[lev],
                'level': lev
            })

    @property
    def mpp(self):
        return self._mpp

    @property
    def metadata(self):
        return self.reader.metadata

    @property
    def properties(self):
        return self.reader.metadata['cucim']

    @property
    def resolutions(self):
        return self.properties['resolutions']

    @property
    def level_count(self):
        return self.resolutions['level_count']

    @property
    def level_dimensions(self):
        return self.resolutions['level_dimensions']

    @property
    def level_downsamples(self):
        return self.resolutions['level_downsamples']

    @property
    def level_tile_sizes(self):
        return self.resolutions['level_tile_sizes']

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
            int:    Optimal downsample level.
        '''
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

    def read_level(self, level: int, to_numpy: bool = False):
        """Read a pyramid level."""
        image = self.reader.read_region(level=level)
        if to_numpy:
            return cucim2numpy(image)
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
    ) -> "CuImage":
        region = self.reader.read_region(
            location=base_level_dim,
            size=(int(extract_size[0]), int(extract_size[1])),
            level=downsample_level,
            num_workers=self.num_workers,
        )
        if resize_factor:
            target_size = (int(np.round(extract_size[0] * resize_factor)),
                           int(np.round(extract_size[1] * resize_factor)))
            if not __cv2_resize__:
                region = resize(np.asarray(region), target_size)
        # Final conversions
        if flatten and region.shape[-1] == 4:
            region = region[:, :, 0:3]
        if convert and convert.lower() in ('jpg', 'jpeg', 'png', 'numpy'):
            region = cucim2numpy(region)
        if resize_factor and __cv2_resize__:
            region = cv2.resize(region, target_size)
        if convert and convert.lower() in ('jpg', 'jpeg'):
            return numpy2jpg(region)
        elif convert and convert.lower() == 'png':
            return numpy2png(region)
        return region

    def read_from_pyramid(
        self,
        top_left: Tuple[int, int],
        window_size: Tuple[int, int],
        target_size: Tuple[int, int],
        convert: Optional[str] = None,
        flatten: bool = False,
    ) -> "CuImage":
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
            CuImage: Image. Dimensions will equal target_size unless
            the window includes an area of the image which is out of bounds.
            In this case, the returned image will be cropped.
        """
        target_downsample = window_size[0] / target_size[0]
        ds_level = self.best_level_for_downsample(target_downsample)

        # Use a lower downsample level if the window size is too small
        ds = self.level_downsamples[ds_level]
        if not int(window_size[0] / ds) or not int(window_size[1] / ds):
            ds_level = max(0, ds_level-1)
            ds = self.level_downsamples[ds_level]

        region = self.read_region(
            top_left,
            ds_level,
            (int(window_size[0] / ds), int(window_size[1] / ds))
        )
        if not __cv2_resize__:
            region = resize(np.asarray(region), (target_size[1], target_size[0]))
        # Final conversions
        if flatten and region.shape[-1] == 4:
            region = region[:, :, 0:3]

        if convert and convert.lower() in ('jpg', 'jpeg', 'png', 'numpy'):
            region = cucim2numpy(region)
        if __cv2_resize__:
            region = cv2.resize(region, target_size)
        if convert and convert.lower() in ('jpg', 'jpeg'):
            return numpy2jpg(region)
        elif convert and convert.lower() == 'png':
            return numpy2png(region)
        return region

    def thumbnail(
        self,
        width: int = 512,
        level: Optional[int] = None,
        associated: bool = False
    ) -> np.ndarray:
        """Return thumbnail of slide as numpy array."""
        if associated:
            log.debug("associated=True not implemented for cucim() thumbnail,"
                      "reading from lowest-magnification layer.")
        if level is None:
            level = self.level_count - 1
        w, h = self.dimensions
        height = int((width * h) / w)
        img = self.read_level(level=level)
        if __cv2_resize__:
            img = cucim2numpy(img)
            return cv2.resize(img, (width, height))
        else:
            img = resize(np.asarray(img), (width, height))
            return cucim2numpy(img)
