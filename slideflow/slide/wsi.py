'''This module includes tools to convolutionally section whole slide images
into tiles. These tessellated tiles can be exported as PNG or JPG as raw
images or stored in the binary format TFRecords, with or without augmentation.'''

from __future__ import absolute_import, division, print_function


import time
import os
import csv
import json
import multiprocessing as mp
import random
import warnings
import cv2
import numpy as np
import pandas as pd
import rasterio.features
import shapely.affinity as sa
import skimage
import skimage.filters
from shapely import __version__ as shapely_version
from shapely.errors import ShapelyDeprecationWarning
from packaging import version
from PIL import Image, ImageDraw
from rich.progress import Progress
from skimage import img_as_ubyte
from slideflow import errors
from functools import partial
from os.path import exists, join, abspath
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import slideflow as sf
import slideflow.slide.qc
from slideflow.util import log, path_to_name  # noqa F401
from .report import SlideReport
from .utils import *
from .backends import tile_worker, backend_formats, wsi_reader


warnings.simplefilter('ignore', Image.DecompressionBombWarning)
warnings.simplefilter("ignore", ShapelyDeprecationWarning)
Image.MAX_IMAGE_PIXELS = 100000000000

# -----------------------------------------------------------------------

class WSI:
    '''Loads a slide and its annotated region of interest (ROI).'''

    def __init__(
        self,
        path: str,
        tile_px: int,
        tile_um: Union[int, str],
        stride_div: int = 1,
        *,
        enable_downsample: bool = True,
        roi_dir: Optional[str] = None,
        rois: Optional[List[str]] = None,
        roi_method: str = 'auto',
        roi_filter_method: Union[str, float] = 'center',
        origin: Union[str, Tuple[int, int]] = (0, 0),
        pb: Optional[Progress] = None,
        verbose: bool = True,
        use_edge_tiles: bool = False,
        mpp: Optional[float] = None,
        simplify_roi_tolerance: Optional[float] = None,
        artifact_rois: Optional[List[str]] = list(),
        **reader_kwargs: Any
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
            origin (str or tuple(int, int)): Offset the starting grid (x, y).
                Either a tuple of ints or 'random'. Defaults to (0, 0).
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
            use_bounds (bool): If True, use the slide bounds to determine
                the slide dimensions. This will crop out unscanned white space.
                If a tuple of int, interprets the bounds as ``(top_left_x,
                top_left_y, width, height)``. If False, use the full slide
                dimensions. **Only available when using Libvips**
                (``SF_SLIDE_BACKEND=libvips``). Defaults to False.
            transforms (list(int), optional): List of transforms to apply to
                the slide before establishing coordinate grid. Options include
                any combination of ``ROTATE_90_CLOCKWISE``,
                ``ROTATE_180_CLOCKWISE``, ``ROTATE_270_CLOCKWISE``,
                ``FLIP_HORIZONTAL``, and ``FLIP_VERTICAL``. **Only available
                when using Libvips** (``SF_SLIDE_BACKEND=libvips``).
                Defaults to None.
            artifact_rois (list(str), optional): List of ROI issue labels
                to treat as artifacts. Whenever this is not None, all the ROIs with
                referred label will be inverted with ROI.invert_roi().
                Defaults to an empty list.

        """
        # Initialize calculated variables
        self.pb = pb
        self.name = path_to_name(path)
        self.shortname = sf.util._shortname(self.name)
        self.tile_px = tile_px
        self.enable_downsample = enable_downsample
        self.thumb_image = None  # type: Optional[Image.Image]
        self.stride_div = stride_div
        self.path = path
        self.filetype = sf.util.path_to_ext(path)
        self.blur_burden = None  # type: Optional[float]
        self.roi_method = None  # type: Optional[str]
        self.extracted_x_size = 0  # type: int
        self.extracted_y_size = 0  # type: int
        self.estimated_num_tiles = 0  # type: int
        self.rois = []  # type: List[ROI]  # List of individual ROI annotations
        self.roi_method = roi_method
        self.roi_grid = None  # type: Optional[np.ndarray]
        self.roi_filter_method = roi_filter_method
        self.qc_masks = []  # type: List[QCMask]
        self.alignment = None  # type: Optional[Alignment]
        self.verbose = verbose
        self.segmentation = None
        self.use_edge_tiles = use_edge_tiles
        self.__slide = None
        self._mpp_override = mpp
        self._reader_kwargs = reader_kwargs
        self.artifact_rois = artifact_rois # type: Optional[List[str]]
        self.grid: np.ndarray

        if isinstance(origin, str) and origin != 'random':
            raise ValueError(
                "Unrecognized value for argument 'origin': {} ."
                "Expected either 'random' or a tuple of ints.".format(origin)
            )
        if isinstance(origin, tuple) and len(origin) != 2:
            raise ValueError(
                "If 'origin' is a tuple, it must be of length 2."
            )
        self.origin = origin

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

        # Initiate supported slide reader
        if not os.path.exists(path):
            raise errors.SlideNotFoundError(f"Could not find slide {path}.")
        if self.filetype.lower() not in sf.util.SUPPORTED_FORMATS:
            raise errors.SlideLoadError(
                f"{self.name}: unsupported filetype '{self.filetype}'"
            )
        if self.filetype.lower() not in backend_formats():
            raise errors.IncompatibleBackendError(
                f"{self.name}: filetype '{self.filetype}' is not supported "
                f"by the current backend, {sf.slide_backend()}"
            )

        # Collect basic slide information
        if not self.slide.has_mpp:
            raise errors.SlideMissingMPPError(
                f"Slide {self.path} missing MPP ({OPS_MPP_X})"
            )
        try:
            self.mpp = float(self.slide.mpp)
        except Exception as e:
            raise errors.SlideMissingMPPError(
                f"Unable to parse MPP for slide {self.path} ({OPS_MPP_X}). "
                f"Error raised: {e}"
            )

        # Configure downsample information
        self._configure_downsample(tile_um)

        # Look in ROI directory if available
        if roi_dir and exists(join(roi_dir, self.name + ".csv")):
            self.load_csv_roi(
                join(roi_dir, self.name + ".csv"),
                process=False,
                simplify_tolerance=simplify_roi_tolerance
            )
        elif rois and self.name in [path_to_name(r) for r in rois]:
            matching_rois = []
            for rp in rois:
                rn = path_to_name(rp)
                if rn == self.name:
                    matching_rois += [rp]
            matching = matching_rois[0]
            if len(matching_rois) > 1:
                log.warning(
                    f"Multiple ROIs found for {self.name}; using {matching}"
                )
            self.load_csv_roi(
                matching,
                process=False,
                simplify_tolerance=simplify_roi_tolerance
            )

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

        # Summarize slide information
        self._log_slide_summary()

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

    def __getitem__(self, index) -> Optional[np.ndarray]:
        """Returns a tile at the given index.

        Args:
            index (tuple): (x, y) grid coordinates of tile to extract.

        Returns:
            Optional[numpy.ndarray]: Image tile, or None if tile is filtered.

        """
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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if '__slide' in state:
            state['__slide'] = None
        if '_WSI__slide' in state:
            state['_WSI__slide'] = None
        if 'pb' in state:
            state['pb'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _build_coord(self) -> None:
        """Set up coordinate grid for image tiles.

        The coordinate grid, stored in ``self.coord``, is a list of lists,
        where each sublist contains the following information:

        - 0: **x**: x-coordinate of the top-left corner of the tile.
        - 1: **y**: y-coordinate of the top-left corner of the tile.
        - 2: **grid_x**: x-coordinate of the tile in self.grid.
        - 3: **grid_y**: y-coordinate of the tile in self.grid.

        """

        # First, remove any existing ROI QC Masks, as these will be recalculated
        # when the coordinate grid is rebuilt.
        self.remove_roi_qc()

        # Calculate window sizes, strides, and coordinates for windows
        self.extracted_x_size = self.dimensions[0] - self.full_extract_px
        self.extracted_y_size = self.dimensions[1] - self.full_extract_px

        # Randomize origin, if desired
        if self.origin == 'random':
            start_x = random.randint(0, self.full_stride-1)
            start_y = random.randint(0, self.full_stride-1)
        else:
            assert isinstance(self.origin, tuple)
            start_x, start_y = self.origin
        log.debug("Slide origin: ({}, {})".format(start_x, start_y))

        # Coordinates must be in level 0 (full) format
        # for the read_region function.
        # Coordinates correspond to top-left corner of the tile.
        self.coord = []  # type: Union[List, np.ndarray]
        edge_buffer = 0 if self.use_edge_tiles else self.full_extract_px
        y_range = np.arange(
            start_y,
            (self.dimensions[1]+1) - edge_buffer,
            self.full_stride
        )
        x_range = np.arange(
            start_x,
            (self.dimensions[0]+1) - edge_buffer,
            self.full_stride
        )

        self.grid = np.ones((len(x_range), len(y_range)), dtype=bool)

        # For any indexes in y_range or x_range corresponding to a negative value,
        # set the corresponding index in self.grid to False.
        # This may occur after slide alignment.
        self.grid[np.argwhere(x_range < 0), :] = False
        self.grid[:, np.argwhere(y_range < 0)] = False

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
            xfact = self.grid.shape[0] / xtrim
            yfact = self.grid.shape[1] / ytrim

            # Offset to align the ROI polygons with the image tile grid
            x_offset = - (full_extract/2 - stride/2)
            y_offset = - (full_extract/2 - stride/2)

            # Translate ROI polygons
            translated = [
                sa.translate(roi.poly, x_offset, y_offset)
                for roi in self.rois if roi.label not in self.artifact_rois
            ]

            # Set scale to 50 times greater than grid size
            # if filtering by float
            o = 1 if roi_by_center else 50

            if len(translated) > 0:
                # Scale ROI polygons
                scaled = [
                    sa.scale(poly, xfact=xfact * o, yfact=yfact * o, origin=(0, 0))
                    for poly in translated
                ]

                # Rasterize polygons to the size of the tile extraction grid.
                # Rasterize polygons for ROIs individually, to keep track of
                # which ROI each tile belongs to, then merge.
                self.roi_grid = np.stack([
                    rasterio.features.rasterize(
                        [scaled_roi],
                        out_shape=(self.grid.shape[1] * o, self.grid.shape[0] * o),
                        all_touched=False).astype(bool).astype(int) * (i + 1)
                    for i, scaled_roi in enumerate(scaled)
                ], axis=0).max(axis=0).T # max means union of the ROIs

            # If self.artifact_rois is not an empty list, calculate the translated_issues, scaled_isses and roi_grid_issues
            if self.artifact_rois:
                # Translate ROI issues polygons
                translated_issues = [
                    sa.translate(roi.invert_roi(self.dimensions).poly, x_offset, y_offset)
                    for roi in self.rois if roi.label in self.artifact_rois
                ]

                if len(translated_issues) > 0:
                    # Scale ROI issues polygons (ROI issues are already inverted)
                    scaled_issues = [
                        sa.scale(poly, xfact=xfact * o, yfact=yfact * o, origin=(0, 0))
                        for poly in translated_issues
                    ]

                    # Rasterize ROI issues polygons, these are the intersection of all ROI issues
                    roi_grid_issues = np.stack([
                        rasterio.features.rasterize(
                            [scaled_roi],
                            out_shape=(self.grid.shape[1] * o, self.grid.shape[0] * o),
                            all_touched=False).astype(bool).astype(int) * (i + 1)
                        for i, scaled_roi in enumerate(scaled_issues)
                    ], axis=0).min(axis=0).T # min means intersection of the ROI issues

                    # Merge Raseterized ROI tissues and issues grid to self.roi_grid (treating as one big ROI)
                    if self.roi_grid is None:
                        self.roi_grid = roi_grid_issues
                    else:
                        # roi_grid_tissues is not None and roi_grid_issues is not None
                        # there is no case in which both are None
                        self.roi_grid = np.minimum(roi_grid_issues, self.roi_grid) # min means intersection of the ROIs

            # Create a merged boolean mask.
            self.roi_mask = self.roi_grid.T.astype(bool)  # type: ignore
        else:
            self.roi_mask = None

        for yi, y in enumerate(y_range):
            for xi, x in enumerate(x_range):
                y = int(y)
                x = int(x)

                # Skip the slide if the coordinate has a negative value.
                # This may happen after slide alignment.
                if x < 0 or y < 0:
                    continue

                self.coord.append([x, y, xi, yi])

                # ROI filtering
                if self.has_rois() and roi_by_center:
                    point_in_roi = self.roi_mask[yi, xi]
                    # If the extraction method is 'inside',
                    # skip the tile if it's not in an ROI
                    if (((self.roi_method in ('inside', 'auto')) and not point_in_roi)
                       or ((self.roi_method == 'outside') and point_in_roi)):
                        self.grid[xi, yi] = 0

        # If roi_filter_method is a float, then perform tile selection
        # based on what proportion of the tile is in an ROI,
        # rather than choosing a tile by centroid (roi_filter_method='center')
        if self.roi_method != 'ignore' and self.has_rois() and not roi_by_center:
            self.apply_qc_mask(
                (~self.roi_mask if self.roi_method == 'inside' else self.roi_mask),
                filter_threshold=(1-self.roi_filter_method),  # type: ignore
                is_roi=True
            )

        self.coord = np.array(self.coord)
        # Handle the case where there is only one tile
        if self.coord.ndim == 1 and self.coord.shape[0] > 0:
            self.coord = self.coord[np.newaxis, :]
        self.estimated_num_tiles = int(self.grid.sum())
        log.debug(f"Set up coordinate grid, shape={self.grid.shape}")

    def _configure_downsample(
        self,
        tile_um: Union[str, int],
        enable_downsample: bool = True
    ) -> None:
        """Configure downsample level for tile extraction.

        Args:
            tile_um (int or str): Size of tiles to extract, in microns (int) or
                magnification (str, e.g. "20x").
            enable_downsample (bool, optional): Allow use of downsampled
                intermediate layers in the slide image pyramid, which greatly
                improves tile extraction speed. May result in artifacts for
                slides with incompletely generated intermediates pyramids.
                Defaults to True.

        """
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
            self.extract_px = self.tile_px
            self.full_extract_px = int(self.downsample_factor * self.tile_px)
            self.tile_um = int(self.downsample_factor * self.mpp * self.tile_px)
            log.debug(f"Using magnification {closest_mag:.1f}x (level="
                      f"{ds_level}, tile_um={self.tile_um})")

        # Calculate downsample level by tile micron size
        else:
            assert isinstance(tile_um, int)
            self.tile_um = tile_um
            self.full_extract_px = int(tile_um / self.mpp)
            ds = self.full_extract_px / self.tile_px
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
        self.stride = int(np.round(self.extract_px / self.stride_div))
        self.full_stride = int(np.round(self.full_extract_px / self.stride_div))

    def _log_slide_summary(self) -> None:
        """Log slide information (MPP, ROIs, grid shape, number of tiles)."""
        mpp_roi_msg = f'{self.mpp} um/px | {len(self.rois)} ROI(s)'
        size_msg = f'Size: {self.dimensions[0]} x {self.dimensions[1]}'
        log.debug(f"{self.shortname}: Slide info: {mpp_roi_msg} | {size_msg}")
        grid_msg = f"{self.shortname}: Grid shape: {self.grid.shape} "
        grid_msg += f"| Tiles to extract: {self.estimated_num_tiles}"
        log.debug(grid_msg)

    def _log_tile_extraction(self) -> None:
        """Log tile extraction parameters."""
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
    def shape(self):
        """Returns the shape of the tile grid."""
        return self.grid.shape

    @property
    def slide(self) -> Any:
        """Backend-specific slide object."""
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
        return self.get_qc_mask()

    # --- Alignment --------------------------------------------------------

    def align_to(
        self,
        slide: "WSI",
        apply: bool = True,
        *,
        finetune_depth: Optional[Sequence[float]] = None,
        normalizer: Optional[str] = 'reinhard_mask',
        allow_errors: bool = False
    ) -> Tuple[Tuple[int, int], float]:
        """Align this slide to another slide.

        Alignment is performed by first aligning thumbnails at low magnification
        (mpp = 8), then progressively fine-tuning alignment at increasing
        magnification (mpp = 1, 0.5, 0.25), focused on a dense tissue region.
        The densest tissue region is identified using the QC mask, if available,
        otherwise via Otsu thresholding.

        Args:
            slide (:class:`slideflow.WSI`): Slide to align to.
            apply (bool): Whether to apply the alignment to the slide.

        Keyword Args:
            finetune_depth (Optional[List[int]]): List of magnifications at
                which to fine-tune alignment. Defaults to [1, 0.5, 0.25].
            normalizer (str, optional): Stain normalization method to use.
                Defaults to 'reinhard_mask'.
            allow_errors (bool): Whether to allow and ignore alignment errors
                when finetuning at higher magnification. Defaults to False.

        Returns:
            Tuple of (x, y) offset and MSE of initial alignment.

        Raises:
            TypeError: If ``slide`` is not a :class:`slideflow.WSI` object.

            AlignmentError: If initial, thumbnail-based alignment fails, or
                if finetuning alignment fails at any magnification and
                ``allow_errors`` is False.

        """
        from scipy import ndimage

        if not isinstance(slide, WSI):
            raise TypeError("Can only align to another slide.")

        if finetune_depth is None:
            finetune_depth = [1, 0.5, 0.25]

        # Steps:
        # 1. Identify tissue region as target for alignment.
        # 2. Rough align with low-mag thumbnails (mpp = 8).
        # 3. Fine-tune alignment at a dense tissue region (mpp = 1, 0.5, 0.25).

        # --- 1. Identify tissue regions as targets for alignment. ------------

        # Use QC mask (.qc_mask) if available, otherwise calculate one.
        # Target should be the centroid of unmasked tissue regions, but
        # there may be multiple distinct tissue regions.

        # First, grab the QC mask, or make one if it is not available.
        if self.qc_mask is not None:
            mask = self.qc_mask
        else:
            log.debug("Applying Otsu thresholding to identify tissue regions.")
            mask = sf.slide.qc.Otsu()(self)

        # Next, fill holes and remove small peaks through gaussian blur,
        # thresholding, and morphological closing.
        log.debug("Filling holes and removing small peaks in tissue mask.")
        mask = skimage.morphology.binary_closing(
            skimage.filters.gaussian(mask, sigma=5) > 0.5,
            skimage.morphology.disk(5)
        )

        # For each pixel in the mask, calculate the nearest distance to an
        # unmasked pixel. This will assist us with finding the densest areas
        # of tissue.
        log.debug("Calculating distance transform of tissue mask.")
        distances = ndimage.distance_transform_edt(~mask)

        # Find the coordinates of the pixel with the highest average distance.
        # This is the center of the densest tissue region.
        log.debug("Identifying target for alignment.")
        target = np.unravel_index(np.argmax(distances), distances.shape)

        # Convert from mask coordinates to slide coordinates.
        target = (
            int(target[1] * (self.dimensions[0] / mask.shape[1])),
            int(target[0] * (self.dimensions[1] / mask.shape[0]))
        )
        target_them = (
            int(np.round(target[0] * (self.mpp / slide.mpp))),
            int(np.round(target[1] * (self.mpp / slide.mpp)))
        )
        log.debug("Low-mag alignment complete.")
        log.debug("Target for alignment (us): {}".format(target))
        log.debug("Target for alignment (them, pre-alignment): {}".format(target_them))

        # --- 2. Align low-mag thumbnails. ------------------------------------

        # Calculate thumbnails for alignment.
        log.debug("Calculating low-mag thumbnails for alignment.")
        our_thumb = np.array(self.thumb(mpp=8))
        their_thumb = np.array(slide.thumb(mpp=8))

        # Stain normalization
        if normalizer is not None:
            log.debug("Aligning with stain normalization: {}".format(normalizer))
            if isinstance(normalizer, str):
                norm = sf.norm.autoselect(normalizer, backend='opencv')
            elif isinstance(normalizer, sf.norm.StainNormalizer):
                norm = normalizer
            else:
                raise ValueError("normalizer must be a str or instance of StainNormalizer")
            our_thumb = norm.transform(our_thumb[:, :, 0:3])
            their_thumb = norm.transform(their_thumb[:, :, 0:3])

        # Align thumbnails and adjust for scale.
        try:
            log.debug("Aligning low-mag thumbnails (mpp=8)...")
            alignment_raw, mse = align_by_translation(
                their_thumb, our_thumb, round=True, calculate_mse=True
            )
        except errors.AlignmentError:
            raise errors.AlignmentError("Alignment failed at thumbnail (mpp=8)")
        alignment = (int(np.round(alignment_raw[0] * (8 / self.mpp))),
                     int(np.round(alignment_raw[1] * (8 / self.mpp))))
        alignment_them = (-int(np.round(alignment_raw[0] * (8 / slide.mpp))),
                          -int(np.round(alignment_raw[1] * (8 / slide.mpp))))

        log.debug("Low-mag alignment (us): {}".format(alignment))
        log.debug("Low-mag alignment (them): {}".format(alignment_them))

        # --- 3. Fine-tune alignment at tissue regions. -----------------------

        # Get the coordinates of the tissue region in both slides.
        for finetune_mpp in finetune_depth:
            if (finetune_mpp < self.mpp) or (finetune_mpp < slide.mpp):
                log.debug("Skipping finetune at mpp={}".format(finetune_mpp))
                continue
            # Us
            our_window_size = (
                int(np.round(512 * (finetune_mpp/self.mpp))),
                int(np.round(512 * (finetune_mpp/self.mpp)))
            )
            our_top_left = (
                int(np.round(target[0] - (our_window_size[0]/2))),
                int(np.round(target[1] - (our_window_size[1]/2)))
            )
            log.debug("Extracting mpp={} alignment window (ours) at window_size={}, top_left={}".format(
                finetune_mpp, our_window_size, our_top_left)
            )
            our_region = self.slide.read_from_pyramid(
                top_left=our_top_left,
                window_size=our_window_size,
                target_size=(512, 512),
                convert='numpy',
                flatten=True,
                pad_missing=True
            )
            # Them
            their_window_size = (
                int(np.round(512 * (finetune_mpp/slide.mpp))),
                int(np.round(512 * (finetune_mpp/slide.mpp)))
            )
            their_top_left = (
                int(np.round(target_them[0] - (their_window_size[0]/2))) + alignment_them[0],
                int(np.round(target_them[1] - (their_window_size[1]/2))) + alignment_them[1]
            )
            log.debug("Extracting mpp={} alignment window (theirs) at window_size={}, top_left={}".format(
                finetune_mpp, their_window_size, their_top_left)
            )
            their_region = slide.slide.read_from_pyramid(
                top_left=their_top_left,
                window_size=their_window_size,
                target_size=(512, 512),
                convert='numpy',
                flatten=True,
                pad_missing=True
            )

            if normalizer is not None:
                our_region = norm.transform(our_region[:, :, 0:3])
                their_region = norm.transform(their_region[:, :, 0:3])

            try:
                rough_alignment = sf.slide.utils._find_translation_matrix(their_region, our_region, h=50, search_window=53)
            except cv2.error:
                rough_alignment = None
                log.debug("Initial rough alignment failed at mpp={}".format(finetune_mpp))
            else:
                log.debug("Initial rough alignment complete at mpp={}".format(finetune_mpp))

            # Finetune alignment on this region.
            try:
                alignment_fine = align_by_translation(their_region, our_region, round=True, warp_matrix=rough_alignment)
            except errors.AlignmentError:
                msg = "Alignment failed at finetuning (mpp={})".format(finetune_mpp)
                if allow_errors:
                    log.error(msg)
                else:
                    raise errors.AlignmentError(msg)
            else:
                alignment = (
                    alignment[0] + int(np.round(alignment_fine[0] * (finetune_mpp/self.mpp))),
                    alignment[1] + int(np.round(alignment_fine[1] * (finetune_mpp/self.mpp)))
                )
                alignment_them = (
                    alignment_them[0] - int(np.round(alignment_fine[0] * (finetune_mpp/slide.mpp))),
                    alignment_them[1] - int(np.round(alignment_fine[1] * (finetune_mpp/slide.mpp)))
                )
                log.debug("Finetune alignment complete at mpp={}.".format(finetune_mpp))
                log.debug("Finetuned alignment (us) at mpp={}: {}".format(finetune_mpp, alignment))
                log.debug("Finetuned alignment (them) at mpp={}: {}".format(finetune_mpp, alignment_them))

        # If not applying alignment, return the base alignment and MSE.
        if not apply:
            log.info("Slide aligned with MSE {:.2f}".format(mse))
            return alignment, mse  # type: ignore

        # Apply alignment.
        self.origin = alignment
        self.alignment = Alignment.from_translation(
            origin=self.slide.coord_to_raw(*alignment),
            scale=(slide.mpp / self.mpp),
        )
        log.info("Slide aligned with MSE {:.2f}. Origin set to {}".format(
                mse, self.origin
        ))

        # Rebuild coordinates and reapply QC, if present.
        self._build_coord()
        if self.has_non_roi_qc():
            self.apply_qc_mask()

        return alignment, mse  # type: ignore

    def align_tiles_to(
        self,
        slide: "WSI",
        normalizer: Optional[str] = 'reinhard_mask',
        *,
        allow_errors: bool = True,
        mask_on_fail: bool = True,
        align_by: str = 'fit',
        ignore_outliers = True,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Align tiles to another slide.

        Differs from :meth:`slideflow.WSI.align_to` in that it aligns each
        tile individually, rather than the slide as a whole. This is useful
        when aligning slides with distortion, whose alignment may drift across
        the slide.

        Args:
            slide (:class:`slideflow.WSI`): Slide to align to.
            normalizer (str, optional): Stain normalization method to use.

        Keyword Args:
            allow_errors (bool): Whether to allow and ignore alignment errors
                when finetuning alignment fails at any magnification and
                ``allow_errors`` is False. Defaults to True.
            mask_on_fail (bool): Whether to mask tiles that fail alignment.
                Defaults to True.
            align_by (str): Either 'tile' or 'fit'. If 'tile', tiles are
                aligned individually. If 'fit', tiles are aligned by fitting
                a plane to the alignment of all tiles. Defaults to 'tile'.
            ignore_outliers (bool): Whether to ignore outliers when fitting
                a plane to tile alignment. Defaults to True.
            **kwargs: Keyword arguments passed to :meth:`slideflow.WSI.align_to`.

        Raises:
            ValueError: If ``align_by`` is not 'tile' or 'fit'.

        Returns:
            np.ndarray: Alignment grid, with shape = (grid_x, grid_y, 2).

        """
        if align_by not in ('tile', 'fit'):
            raise ValueError("align_by must be 'tile' or 'median'")

        # Stain normalizer.
        if normalizer is not None:
            if isinstance(normalizer, str):
                normalizer = sf.norm.autoselect(normalizer, backend='opencv')
            elif not isinstance(normalizer, sf.norm.StainNormalizer):
                raise ValueError("normalizer must be a str or instance of StainNormalizer")

        # Perform coarse alignment.
        self.align_to(
            slide, apply=True, normalizer=normalizer, allow_errors=allow_errors, **kwargs
        )

        # Finetune alignment at each tile location.
        from tqdm import tqdm

        ctx = mp.get_context('spawn') if sf.slide_backend() == 'libvips' else mp.get_context('fork')
        pool = ctx.Pool(num_workers or sf.util.num_cpu())

        alignment_coords = np.zeros((self.coord.shape[0], 2))
        half_extract_px = int(np.round(self.full_extract_px/2))
        idx_to_remove = []
        for tile_alignment, c in tqdm(pool.imap_unordered(
                                        partial(calc_alignment,
                                                us=self,
                                                them=slide,
                                                n=normalizer),
                                        enumerate(self.coord)),
                                      desc="Aligning tiles...",
                                      total=len(self.coord)):
            idx, (x, y, xi, yi) = c
            if tile_alignment == 'error':
                msg = "Tile alignment failed at x={}, y={} (grid {}, {})".format(
                    x, y, xi, yi
                )
                if allow_errors:
                    log.debug(msg)
                    tile_alignment = None
                else:
                    raise errors.AlignmentError(msg)
            if tile_alignment is None and mask_on_fail and align_by == 'tile':
                self.grid[xi, yi] = False
                idx_to_remove += [idx]
            elif tile_alignment is None:
                idx_to_remove += [idx]
            if tile_alignment is not None:
                pixel_ratio = (self.full_extract_px / self.tile_px)
                x_adjust = int(np.round(tile_alignment[0] * pixel_ratio))
                y_adjust = int(np.round(tile_alignment[1] * pixel_ratio))
                x_base, y_base = self.slide.coord_to_raw(
                    x + half_extract_px,
                    y + half_extract_px
                )
                x_base_adjusted, y_base_adjusted = self.slide.coord_to_raw(
                    x + half_extract_px + x_adjust,
                    y + half_extract_px + y_adjust
                )
                x_base_adjustment = x_base_adjusted - x_base
                y_base_adjustment = y_base_adjusted - y_base
                alignment_coords[idx] = np.array([x_base_adjustment, y_base_adjustment])
                log.debug("Tile alignment complete at x={}, y={} (grid {}, {}): adjust by {}, {}".format(
                    x, y, xi, yi, x_adjust, y_adjust
                ))

        pool.close()

        coord_mask = np.any(self.get_masked_coord().mask, 1)
        coord_mask[np.array(idx_to_remove).astype(int)] = True
        mask = np.repeat(coord_mask[:, None], 2, axis=1)
        all_alignment_coords = np.ma.masked_array(alignment_coords, mask=mask)  # type: ignore
        coord_raw = self.slide.coord_to_raw(
            self.coord[~coord_mask][:, 0] + half_extract_px,
            self.coord[~coord_mask][:, 1] + half_extract_px
        )
        log.debug("Removing {} indices with failed alignment. Max coord size: {}".format(len(idx_to_remove), len(self.coord)))

        if align_by == 'fit':
            log.debug("Fitting to {} coordinates.".format((~coord_mask).sum()))
            x_adjustment_coordinates = np.column_stack((
                coord_raw[0],
                coord_raw[1],
                all_alignment_coords[~coord_mask][:, 0],
            ))
            y_adjustment_coordinates = np.column_stack((
                coord_raw[0],
                coord_raw[1],
                all_alignment_coords[~coord_mask][:, 1],
            ))

            def build_aligned_coords(x_centroid, x_normal, y_centroid, y_normal):
                coord_on_plane = np.zeros((len(self.coord), 2), dtype=int)
                coord_on_plane = np.ma.masked_array(coord_on_plane, mask=mask)
                for idx, (x, y, xi, yi) in enumerate(self.coord):
                    # Convert coordinates to raw base layer coordinates
                    bx, by = self.slide.coord_to_raw(
                        x + half_extract_px,
                        y + half_extract_px
                    )
                    # Align to raw base layer coordinates
                    coord_on_plane[idx] = (
                        int(np.round(z_on_plane(bx, by, x_centroid, x_normal))),
                        int(np.round(z_on_plane(bx, by, y_centroid, y_normal)))
                    )
                return coord_on_plane

            x_centroid, x_normal = best_fit_plane(x_adjustment_coordinates)
            y_centroid, y_normal = best_fit_plane(y_adjustment_coordinates)
            fit_alignment = build_aligned_coords(x_centroid, x_normal, y_centroid, y_normal)

            if ignore_outliers:
                # Calculate outlier threshold (90th percentile)
                diff = np.abs(all_alignment_coords - fit_alignment)
                diff = np.max(diff, axis=-1)
                threshold = np.percentile(diff[~diff.mask].data, 90)
                all_alignment_coords.mask[diff > threshold] = True
                coord_mask[diff > threshold] = True
                fit_alignment.mask = all_alignment_coords.mask
                log.debug("Re-fitting to {} coordinates, ignoring outliers.".format((~coord_mask).sum()))

                coord_raw = self.slide.coord_to_raw(
                    self.coord[~coord_mask][:, 0] + half_extract_px,
                    self.coord[~coord_mask][:, 1] + half_extract_px
                )

                # Recalculate fit without outliers
                x_adjustment_coordinates = np.column_stack((
                    coord_raw[0],
                    coord_raw[1],
                    all_alignment_coords[~coord_mask][:, 0],
                ))
                y_adjustment_coordinates = np.column_stack((
                    coord_raw[0],
                    coord_raw[1],
                    all_alignment_coords[~coord_mask][:, 1],
                ))

                x_centroid, x_normal = best_fit_plane(x_adjustment_coordinates)
                y_centroid, y_normal = best_fit_plane(y_adjustment_coordinates)

                all_alignment_coords = build_aligned_coords(x_centroid, x_normal, y_centroid, y_normal)
            else:
                all_alignment_coords = fit_alignment

            self.alignment = Alignment.from_fit(
                origin=self.slide.coord_to_raw(*self.origin),
                scale=(slide.mpp / self.mpp),
                centroid=(x_centroid, y_centroid),
                normal=(x_normal, y_normal)
            )

        for idx, (x, y, xi, yi) in enumerate(self.coord):
            if np.ma.is_masked(all_alignment_coords[idx][0]):
                continue

            bx, by = self.slide.coord_to_raw(
                x + half_extract_px,
                y + half_extract_px
            )
            x, y = self.slide.raw_to_coord(
                bx + all_alignment_coords[idx][0],
                by + all_alignment_coords[idx][1]
            )
            self.coord[idx, 0] = x - half_extract_px
            self.coord[idx, 1] = y - half_extract_px

        # Delete tiles that failed to align.
        if idx_to_remove and align_by == 'tile':
            log.warning("Removing {} tiles that failed to align.".format(len(idx_to_remove)))
            self.coord = np.delete(self.coord, idx_to_remove, axis=0)

        if align_by != 'fit':
            self.alignment = Alignment.from_coord(
                origin=self.slide.coord_to_raw(*self.origin),
                scale=(slide.mpp / self.mpp),
                coord=self.coord
            )

        log.info("Slide alignment complete and finetuned at each unmasked tile location.")

        return all_alignment_coords

    def apply_alignment(self, alignment: Alignment) -> None:
        """Apply alignment to the slide.

        Args:
            alignment (slideflow.slide.Alignment): Alignment object.

        """
        self.alignment = alignment
        self.origin = self.slide.raw_to_coord(*alignment.origin)
        if alignment.coord is not None:
            self.coord = alignment.coord
        elif alignment.centroid is None:
            self._build_coord()
            if self.qc_mask is not None:
                self.apply_qc_mask()
        else:
            self._build_coord()
            if self.qc_mask is not None:
                self.apply_qc_mask()
            if alignment.centroid is not None:
                x_centroid, y_centroid = alignment.centroid
                x_normal, y_normal = alignment.normal
                half_extract_px = int(np.round(self.full_extract_px/2))
                for idx, (x, y, xi, yi) in enumerate(self.coord):
                    x = (xi * int(np.round(self.full_stride/alignment.scale))) * alignment.scale
                    y = (yi * int(np.round(self.full_stride/alignment.scale))) * alignment.scale
                    x += self.origin[0]
                    y += self.origin[1]
                    bx, by = self.slide.coord_to_raw(
                        x + half_extract_px,
                        y + half_extract_px
                    )
                    adjust_x = int(np.round(z_on_plane(bx, by, x_centroid, x_normal)))
                    adjust_y = int(np.round(z_on_plane(bx, by, y_centroid, y_normal)))
                    x, y = self.slide.raw_to_coord(bx + adjust_x, by + adjust_y)
                    self.coord[idx, 0] = x - half_extract_px
                    self.coord[idx, 1] = y - half_extract_px

    def load_alignment(self, path: str) -> None:
        """Load alignment from a file.

        Args:
            path (str): Path to alignment file.

        """
        self.apply_alignment(Alignment.load(path))

    # --- All other functions -----------------------------------------------

    def apply_qc_mask(
        self,
        mask: Optional[Union[np.ndarray, QCMask]] = None,
        filter_threshold: Optional[float] = None,
        *,
        is_roi: bool = False
    ) -> "Image":
        """Apply custom slide-level QC by filtering grid coordinates.

        The mask should have a shape (height, width) proportional to the
        slide's dimensions.

        If the mask is numerical, the mask is thresholded at filter_threshold,
        with values above the threshold indicating a region to discard.

        If the mask is a boolean array, True indicates a region to
        discard and False indicates a region to keep.

        If the mask is a QCMask, the filter_threshold is ignored.

        Args:
            mask (np.ndarray or :class:`slideflow.slide.QCMask`, optional):
                Boolean QC mask array or ``QCMask`` object. If None, will
                re-apply the current masks. Defaults to None.
            filter_threshold (float): Percent of a tile detected as
                background that will trigger a tile to be discarded.
                Only used if ``mask`` is an np.ndarray.
                Defaults to 0.6.

        Keyword Args:
            is_roi (bool): Whether the mask is an ROI mask. Only used if ``mask``
                is an ``np.ndarray``. Defaults to False.

        Returns:
            Image: Image of applied QC mask.
        """
        # If no mask is provided and none has been previously applied,
        # raise an error.
        if mask is None and not len(self.qc_masks):
            raise errors.QCError("No QC mask available")

        # If no mask provided, re-apply the current masks.
        if mask is None:
            for qc_mask in self.qc_masks:
                self.apply_qc_mask(qc_mask)
            return Image.fromarray(img_as_ubyte(self.qc_mask))

        # Verify that the mask is a np.ndarray or QCMask.
        if not isinstance(mask, (np.ndarray, QCMask)):
            raise TypeError("mask must be a np.ndarray or QCMask")

        # Set the filter threshold if not provided.
        # If mask is a QCMask, use its filter_threshold.
        # Otherwise, default to 0.6.
        if not isinstance(mask, QCMask) and filter_threshold is None:
            filter_threshold = 0.6
        elif filter_threshold is not None and isinstance(mask, QCMask):
            raise ValueError(
                "filter_threshold cannot be provided if mask is a QCMask"
            )
        elif filter_threshold is None:
            filter_threshold = mask.filter_threshold  # type: ignore

        # If the provided mask is an np.ndarray, convert it to a QCMask.
        if not isinstance(mask, QCMask):
            mask = QCMask(mask, filter_threshold=filter_threshold, is_roi=is_roi)  # type: ignore
            self.qc_masks.append(mask)

        # Apply the mask to the grid.
        downsample = self.dimensions[0] / mask.shape[1]
        qc_ratio = 1 / downsample
        qc_width = int(np.round(self.full_extract_px * qc_ratio))
        for x, y, xi, yi in self.coord:  # type: ignore
            # x and y are top-left coordinates for the tile.
            qc_x = int(np.round(x * qc_ratio))
            qc_y = int(np.round(y * qc_ratio))
            submask = mask.mask[qc_y:(qc_y+qc_width), qc_x:(qc_x+qc_width)]
            if (submask.size > 0) and (np.mean(submask) > filter_threshold):
                    self.grid[xi, yi] = 0

        # Update the estimated number of tiles.
        self.estimated_num_tiles = int(self.grid.sum())

        # Return an image of the applied mask.
        return Image.fromarray(img_as_ubyte(self.qc_mask))

    def apply_segmentation(self, segmentation: "sf.cellseg.Segmentation") -> None:
        """Apply cell segmentation to the slide.

        This sets the coordinates to the centroids of the segmentation.

        Args:
            segmentation (slideflow.cellseg.Segmentation): Segmentation object
                to apply.

        """
        # Filter out masks outside of ROIs, if present.
        if self.has_rois():
            log.debug(f"Applying {len(self.rois)} ROIs to segmentation.")
            segmentation.apply_rois(1, [r.poly for r in self.rois])

        if segmentation.slide is None:
            segmentation.slide = self
        self.segmentation = segmentation
        centroids = segmentation.centroids(wsi_dim=True)
        self.seg_coord = np.concatenate(
            (centroids, np.expand_dims(np.arange(centroids.shape[0]), axis=-1)),
            axis=-1)
        nonzero = self.seg_coord[:, 0] > 0
        self.seg_coord[:, 0:2][nonzero] -= int(self.full_extract_px/2)
        self.estimated_num_tiles = centroids.shape[0]

    def area(self) -> float:
        """Calculate area (mm^2) of slide that passes QC masking."""
        dim_x, dim_y = self.dimensions[0], self.dimensions[1]
        total_area_in_sq_microns = (dim_x * self.mpp) * (dim_y * self.mpp)
        if self.qc_mask is not None:
            s = self.qc_mask.shape
            p = 1 - (self.qc_mask.sum() / (s[0] * s[1]))
            area_in_sq_microns = p * total_area_in_sq_microns
        else:
            area_in_sq_microns = total_area_in_sq_microns
        area_in_sq_mm = area_in_sq_microns * 1e-6
        return area_in_sq_mm

    def build_generator(
        self,
        *,
        shuffle: bool = True,
        whitespace_fraction: float = None,
        whitespace_threshold: float = None,
        grayspace_fraction: float = None,
        grayspace_threshold: float = None,
        normalizer: Optional[Union[str, "slideflow.norm.StainNormalizer"]] = None,
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
        """Builds a tile generator to extract tiles from this slide.

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
            normalizer_source (str, optional): Stain normalization preset or
                path to a source image. Valid presets include 'v1', 'v2', and
                'v3'. If None, will use the default present ('v3').
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
            shard (tuple(int, int), optional): If provided, will only extract
                tiles from the shard with index `shard[0]` out of `shard[1]`
                shards. Defaults to None.

        Returns:
            A generator that yields a dictionary with the keys:

                - ``"image"``: image data.
                - ``"yolo"``: yolo-formatted annotations, (x_center, y_center, width, height), optional.
                - ``"grid"``: (x, y) grid coordinates of the tile.
                - ``"loc"``: (x, y) coordinates of tile center, in base (level=0) dimension.

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

        self._log_tile_extraction()
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
                    #   to reduce memory utilization.
                    # In the Libvips backend, a multiprocessing pool is default
                    #   to significantly improve performance.
                    n_cores = sf.util.num_cpu(default=8)
                    if sf.slide_backend() == 'libvips':
                        num_processes = max(int(n_cores/2), 1)
                    else:
                        num_threads = n_cores
                if num_threads is not None and num_threads > 1:
                    log.debug(f"Building generator ThreadPool({num_threads})")
                    pool = mp.dummy.Pool(processes=num_threads)
                    should_close = True
                elif num_processes is not None and num_processes > 1:
                    ptype = 'spawn' if sf.slide_backend() == 'libvips' else 'fork'
                    log.debug(f"Building generator with Pool({num_processes}), "
                              f"type={ptype}")
                    ctx = mp.get_context(ptype)
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

    def coord_to_grid(
        self,
        x: int,
        y: int,
        *,
        anchor: str = 'center'
    ) -> Tuple[int, int]:
        """Find the grid index of a tile by its base-level coordinates.

        Args:
            x (int): x-coordinate of the tile, in base (level=0) dimension.
            y (int): y-coordinate of the tile, in base (level=0) dimension.

        Keyword args:
            anchor (str): Anchor point for the coordinates. Either 'topleft'
                or 'center'. Defaults to 'center'.

        Returns:
            Tuple[int, int]: Grid index of the tile.

        Raises:
            ValueError: If anchor is not 'topleft' or 'center'.
            IndexError: If tile is not found at the given coordinates.

        """
        if anchor not in ('topleft', 'center'):
            raise ValueError("anchor must be 'topleft' or 'center'")
        if anchor == 'center':
            x -= int(self.full_extract_px/2)
            y -= int(self.full_extract_px/2)
        coord_idx, = np.where((
            (self.coord[:, 0] == x)
            & (self.coord[:, 1] == y)
        ))
        if not len(coord_idx):
            raise IndexError(f"Tile at coord=({x}, {y}) not found")
        assert len(coord_idx) == 1
        x, y, grid_x, grid_y = self.coord[coord_idx[0]]
        return grid_x, grid_y

    def dim_to_mpp(self, dimensions: Tuple[float, float]) -> float:
        return (self.dimensions[0] * self.mpp) / dimensions[0]

    def export_rois(self, dest: Optional[str] = None) -> str:
        """Export loaded ROIs to a given destination, in CSV format.

        ROIs are exported with the columns 'roi_name', 'x_base', and 'y_base'.
        Coordinates are in base dimension (level 0) of the slide.

        Args:
            dest (str): Path to destination folder. If not provided, will
                export ROIs in the current folder. Defaults to None.

        Returns:
            None

        """
        names, labels, x, y = [], [], [], []

        def append_roi(roi):
            nonlocal names, labels, x, y
            c = np.array(roi.coordinates)
            assert len(c.shape) == 2
            names += [roi.name] * c.shape[0]
            labels += [roi.label] * c.shape[0]
            x += list(c[:, 0])
            y += list(c[:, 1])

        for roi in self.rois:
            append_roi(roi)
            for hole in roi.holes.values():
                append_roi(hole)

        df = pd.DataFrame({
            'roi_name': names,
            'label': labels,
            'x_base': x,
            'y_base': y
        })
        if dest is None:
            dest = f'{self.name}.csv'
        df.to_csv(dest, index=False)
        log.info(f"{len(self.rois)} ROIs exported to {abspath(dest)}")
        return abspath(dest)

    def get_qc_mask(self, roi: bool = True) -> Optional[np.ndarray]:
        """Return the combined QC mask for the slide.

        Args:
            roi (bool): Whether to include ROI masks. Defaults to True.

        """
        _all_masks = [m for m in self.qc_masks if (roi or (not m.is_roi))]
        if not _all_masks:
            return None
        elif len(_all_masks) == 1:
            return _all_masks[0].mask
        else:
            _, smallest = min((m.shape[0], idx)
                               for (idx, m) in enumerate(_all_masks))
            shape = _all_masks[smallest].shape
            mask = skimage.transform.resize(_all_masks[0].mask, shape).astype(bool)
            for _next in _all_masks[1:]:
                _next_m = skimage.transform.resize(_next.mask, shape).astype(bool)
                mask = np.logical_or(mask, _next_m)
            return mask

    def get_masked_coord(self) -> np.ma.core.MaskedArray:
        """Get a masked array of the coordinate grid, masked by QC.

        The returned masked array is of shape (n, 4), where n is the number of tiles.
        The columns are (x, y, grid_x, grid_y), where x and y are the
        top-left coordinates of the tile, and grid_x and grid_y are the
        grid indices of the tile.

        """
        true_grid_indices = np.flatnonzero(self.grid)
        linear_indices_of_coord = np.ravel_multi_index(
            self.coord[:, 2:4].T,
            dims=self.grid.shape
        )
        unmasked_coord_indices = np.in1d(
            linear_indices_of_coord,
            true_grid_indices
        )
        return np.ma.masked_array(
            self.coord,
            mask=~np.repeat(unmasked_coord_indices[:, None], 4, axis=1)
        )

    def get_roi_by_name(self, name: str) -> Optional[ROI]:
        """Get an ROI by its name.

        Args:
            name (str): Name of the ROI.

        Returns:
            ROI: ROI object.

        """
        for roi in self.rois:
            if roi.name == name:
                return roi
        return None

    def get_tile_coord(self, anchor='topleft') -> np.ndarray:
        """Get a coordinate grid of all tiles, restricted to those that pass QC
        and any ROI filtering.

        The returned array is of shape (n, 4), where n is the number of tiles.
        The columns are (x, y, grid_x, grid_y), where x and y are the
        top-left coordinates of the tile, and grid_x and grid_y are the
        grid indices of the tile.

        """
        if anchor not in ('center', 'topleft'):
            raise ValueError("Expected `anchor` to be 'center' or 'topleft'")
        c = self.coord[
            self.grid[tuple(self.coord[:, 2:4].T)].astype(bool)
        ].copy()
        if anchor == 'center':
            c[:, 0] += int(self.full_extract_px/2)
            c[:, 1] += int(self.full_extract_px/2)
        return c

    def get_tile_dataframe(self) -> pd.DataFrame:
        """Build a dataframe of tiles and associated ROI labels.

        Returns:
            Pandas dataframe of all tiles, with the following columns:
            - ``loc_x``: X-coordinate of tile center
            - ``loc_y``: Y-coordinate of tile center
            - ``grid_x``: X grid index of the tile
            - ``grid_y``: Y grid index of the tile
            - ``roi_name``: Name of the ROI if tile is in an ROI, else None
            - ``roi_desc``: Description of the ROI if tile is in ROI, else None
            - ``label``: ROI label, if present.

        """
        roi_names = []
        roi_desc = []
        labels = []
        index = []
        loc = []
        grid = []
        for x, y, xi, yi in self.coord:
            if not self.grid[xi, yi]:
                continue
            _, roi = self.get_tile_roi(grid=(xi, yi))

            # Convert from top-left to center coordinates
            x += int(self.full_extract_px/2)
            y += int(self.full_extract_px/2)

            loc.append([x, y])
            grid.append([xi, yi])
            roi_names.append(None if not roi else roi.name)
            roi_desc.append(None if not roi else roi.description)
            labels.append(None if not roi else roi.label)
            index.append(f'{self.name}-{x}-{y}')
        loc = np.array(loc)
        grid = np.array(grid)
        df = pd.DataFrame({
            'loc_x': loc[:, 0],
            'loc_y': loc[:, 1],
            'grid_x': grid[:, 0],
            'grid_y': grid[:, 1],
            'roi_name': roi_names,
            'roi_desc': roi_desc,
            'label': labels
        }, index=index)
        return df

    def get_tile_roi_mask(
        self,
        *,
        grid: Optional[Tuple[int, int]] = None,
        loc: Optional[Tuple[int, int]] = None,
        mode: str = 'binary',
        roi_labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Get the ROI mask for a tile at the given location.

        Keyword Args:
            grid (tuple[int, int], optional): Grid indices of the tile.
                Must supply either ``grid`` or ``loc``. Defaults to None.
            loc (tuple[int, int], optional): Location of the tile center.
                Must supply either ``grid`` or ``loc``. Defaults to None.
            mode (str, optional): 'binary', 'multiclass', or 'multilabel'.
                Defaults to 'binary'.
            roi_labels (list[str], optional): List of ROI labels to include.
                Defaults to None.

        Returns:
            np.ndarray: ROI mask for the tile, with dtype int and shape
                (n, tile_px, tile_px), where n is the number of ROI labels.

        """
        if grid is None and loc is None:
            raise ValueError("Either grid or loc must be provided.")

        # Definitions.
        fe = self.full_extract_px
        fs = self.full_stride
        scale = self.tile_px / fe

        # Get the polygon vertices for the tile.
        if grid is not None:
            # Convert from grid to top-left coordinates
            gx, gy = grid
            topleft = (gx * fs, gy * fs)
            bottomleft = (gx * fs, (gy * fs) + fe)
            bottomright = ((gx * fs) + fe, (gy * fs) + fe)
            topright = ((gx * fs) + fe, gy * fs)
        else:
            # Convert from center to top-left coordinates
            cx, cy = loc
            cx -= int(fe / 2)
            cy -= int(fe / 2)
            topleft = (cx, cy)
            bottomleft = (cx, cy + fe)
            bottomright = (cx + fe, cy + fe)
            topright = (cx + fe, cy)

        # Get a polygon for the tile, used for determining overlapping ROIs.
        tile = sg.Polygon([topleft, bottomleft, bottomright, topright])

        # Compute the mask from ROIs.
        if len(self.rois) == 0:
            if roi_labels:
                mask = np.zeros((len(roi_labels), self.tile_px, self.tile_px), dtype=int)
            else:
                mask = np.zeros((1, self.tile_px, self.tile_px), dtype=int)

        # Handle ROIs with labels (multilabel or multiclass)
        elif roi_labels:
            labeled_masks = []
            for label in roi_labels:
                wsi_polys = [p.poly for p in self.rois if p.label == label]
                if len(wsi_polys) == 0:
                    mask = np.zeros((self.tile_px, self.tile_px), dtype=int)
                    labeled_masks.append(mask)
                else:
                    all_polys = unary_union(wsi_polys)
                    polys = get_scaled_and_intersecting_polys(
                        all_polys, tile, scale, topleft
                    )
                    if isinstance(polys, sg.Polygon) and polys.is_empty:
                        mask = np.zeros((self.tile_px, self.tile_px), dtype=int)
                    else:
                        # Rasterize to an int mask.
                        mask = rasterio.features.rasterize(
                            [polys],
                            out_shape=[self.tile_px, self.tile_px]
                        )
                        mask = mask.astype(int)
                    labeled_masks.append(mask)
            mask = np.stack(labeled_masks, axis=0)

        # Handle ROIs without labels (binary)
        else:
            # Determine the intersection at the given tile location.
            all_polys = unary_union([p.poly for p in self.rois])
            polys = get_scaled_and_intersecting_polys(
                all_polys, tile, scale, topleft
            )

            if isinstance(polys, sg.Polygon) and polys.is_empty:
                mask = np.zeros((self.tile_px, self.tile_px), dtype=int)
            else:
                # Rasterize to an int mask.
                try:
                    mask = rasterio.features.rasterize(
                        [polys],
                        out_shape=[self.tile_px, self.tile_px]
                    )
                    mask = mask.astype(bool).astype(np.int32)
                except ValueError:
                    mask = np.zeros((self.tile_px, self.tile_px), dtype=int)

            # Add a dummy channel dimension.
            mask = mask[None, :, :]

        # Process according to the mode.
        if mode == 'multiclass':
            mask = mask * np.arange(1, mask.shape[0]+1)[:, None, None]
            mask = mask.max(axis=0)
        elif mode == 'binary' and mask.ndim == 3:
            mask = np.any(mask, axis=0)[None, :, :].astype(int)

        return mask

    def has_non_roi_qc(self) -> bool:
        """Check if the slide has any non-ROI QC masks."""
        return any(not m.is_roi for m in self.qc_masks)

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
            normalizer_source (str, optional): Stain normalization preset or
                path to a source image. Valid presets include 'v1', 'v2', and
                'v3'. If None, will use the default present ('v3').
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
        slide_bytes = bytes(self.name, 'utf-8')

        for index, tile_dict in enumerate(generator_iterator):
            x, y = location = tile_dict['loc']
            locations += [location]
            grid_locations += [tile_dict['grid']]
            if 'ws_fraction' in tile_dict:
                ws_fractions += [tile_dict['ws_fraction']]
            if 'gs_fraction' in tile_dict:
                gs_fractions += [tile_dict['gs_fraction']]

            if dry_run:
                continue

            img_str = tile_dict['image']
            if len(sample_tiles) < 10:
                sample_tiles += [img_str]
            elif (not tiles_dir and not tfrecord_dir) and not dry_run:
                break
            if tiles_dir:
                img_f = join(
                    tiles_dir,
                    f'{self.shortname}-{x}-{y}.{img_format}'
                )
                with open(img_f, 'wb') as outfile:
                    outfile.write(img_str)
                if 'yolo' in tile_dict and len(tile_dict['yolo']):
                    yolo_f = join(tiles_dir, f'{self.shortname}-{x}-{y}.txt')
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
                record = sf.io.serialized_record(slide_bytes, img_str, x, y)
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
        return self.extract_tiles(
            tfrecord_dir,
            tiles_dir,
            img_format,
            report,
            apply_masks=apply_masks,
            from_centroids=True,
            **kwargs
        )

    def get_tile_roi(
        self,
        coord: Optional[Tuple[int, int]] = None,
        grid: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[int], Optional[str]]:
        """Find the ROI that contains a given tile.

        Args:
            coord (Tuple[int, int], optional): Base-level coordinates of the
                tile. Cannot supply both ``coord`` and ``grid``. Defaults to None.
            grid (Tuple[int, int], optional): Grid index of the tile.
                Cannot supply both ``coord`` and ``grid``. Defaults to None.

        Returns:
            Tuple[int, ROI]: ROI index (index of WSI.rois) and
                the :class:`slideflow.slide.ROI` that contains the tile.
                If no ROI contains the tile, returns (None, None).

        """
        if coord is not None and grid is not None:
            raise ValueError("Cannot specify both coord and grid")
        if coord is not None:
            grid = self.coord_to_grid(*coord)
        elif grid is None:
            raise ValueError("Must specify either coord or grid")
        if self.roi_grid is None:
            return None, None
        grid_x, grid_y = grid
        roi_idx = self.roi_grid[grid_x, grid_y] - 1
        if roi_idx == -1:
            return None, None
        else:
            return roi_idx, self.rois[roi_idx]

    def grid_to_coord(
        self,
        grid_x: int,
        grid_y: int,
        *,
        anchor: str = 'center'
    ) -> Tuple[int, int]:
        """Find the base-level coordinates of a tile by its grid index.

        Args:
            grid_x (int): x-index of the tile in the grid.
            grid_y (int): y-index of the tile in the grid.

        Keyword args:
            anchor (str): Anchor point for the coordinates. Either 'topleft'
                or 'center'. Defaults to 'center'.

        Returns:
            Tuple[int, int]: Base-level coordinates of the tile.

        Raises:
            ValueError: If anchor is not 'topleft' or 'center'.
            IndexError: If tile is not found at the given coordinates.

        """
        if anchor not in ('topleft', 'center'):
            raise ValueError("anchor must be 'topleft' or 'center'")
        grid_idx, = np.where((
            (self.coord[:, 2] == grid_x)
            & (self.coord[:, 3] == grid_y)
        ))
        if not len(grid_idx):
            raise IndexError(f"Tile at grid=({grid_x}, {grid_y}) not found")
        assert len(grid_idx) == 1
        x, y, grid_x, grid_y = self.coord[grid_idx[0]]
        if anchor == 'center':
            x += int(self.full_extract_px/2)
            y += int(self.full_extract_px/2)
        return x, y

    def get_tile_mask(self, index, sparse_mask) -> np.ndarray:
        """Get a mask for a tile, given a sparse mask.

        Examples
            Get a mask for a tile, given a sparse mask.

                >>> from slideflow.cellseg import seg_utils, Segmentation
                >>> segmentation = Segmentation(...)
                >>> wsi = sf.WSI(...)
                >>> wsi.apply_segmentation(segmentation)
                >>> sparse_mask = seg_utils.sparse_mask(segmentation.masks)
                >>> wsi.get_tile_mask(0, sparse_mask)
                <numpy.ndarray>

        Args:
            index (int): Index of tile.
            sparse_mask (scipy.sparse.csr_matrix): Sparse mask.

        Returns:
            numpy.ndarray: Mask for tile.

        """
        # Get the corresponding segmentation mask, reading from the sparse matrix
        seg = self.segmentation
        if seg is None:
            raise ValueError("Segmentation not yet applied to slide.")
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

    def has_rois(self) -> bool:
        """Checks if the slide has loaded ROIs and they are not being ignored."""
        return (self.roi_method != 'ignore'
                and len(self.rois))

    def get_next_roi_name(self) -> str:
        """Get the next available name for an ROI."""
        existing = [
            int(r.name[4:]) for r in self.rois
            if r.name.startswith('ROI_') and r.name[4:].isnumeric()
        ]
        roi_id = list(set(list(range(len(existing)+1))) - set(existing))[0]
        name = f'ROI_{roi_id}'
        return name

    def load_roi_array(
        self,
        array: np.ndarray,
        *,
        process: bool = True,
        label: Optional[str] = None,
        name: Optional[str] = None,
        allow_errors: bool = False,
        simplify_tolerance: Optional[float] = None
    ) -> int:
        """Load an ROI from a numpy array.

        Args:
            array (np.ndarray): Array of shape (n_points, 2) containing
                the coordinates of the ROI shape, in base (level=0) dimension.

        Keyword Args:
            process (bool): Process ROIs after loading. Defaults to True.

        """
        name = name or self.get_next_roi_name()
        try:
            roi = ROI(name, array, label=label)
        except errors.InvalidROIError as e:
            if allow_errors:
                log.warn("Unable to load ROI: {}".format(e))
                return
            else:
                raise
        if simplify_tolerance is not None:
            roi.simplify(simplify_tolerance)
        self.rois.append(roi)
        if self.roi_method == 'auto':
            self.roi_method = 'inside'
        if process:
            self.process_rois()
        for i, _roi in enumerate(self.rois):
            if _roi == roi:
                return i
            for hole in _roi.holes.values():
                if hole == roi:
                    return i
        return None

    def load_csv_roi(
        self,
        path: str,
        *,
        process: bool = True,
        scale: int = 1,
        skip_invalid: bool = True,
        simplify_tolerance: Optional[float] = None
    ) -> int:
        """Load ROIs from a CSV file.

        CSV file must contain headers 'ROI_name', 'X_base', and 'Y_base'.

        Any previously loaded ROIs are cleared prior to loading.

        Args:
            path (str): Path to CSV file.

        Keyword Args:
            process (bool): Process ROIs after loading. Defaults to True.
            scale (int): Scale factor to apply to ROI coordinates. Defaults to 1.

        """
        # Clear any previously loaded ROIs.
        self.rois = []

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
            index_label = None if not "label" in headers else headers.index("label")
            for row in reader:
                roi_name = row[index_name]
                x_coord = int(float(row[index_x]) * scale)
                y_coord = int(float(row[index_y]) * scale)
                label = None if index_label is None else row[index_label]

                if roi_name not in roi_dict:
                    roi_dict[roi_name] = {
                        'coords': [],
                        'label': label
                    }
                roi_dict[roi_name]['coords'].append((x_coord, y_coord))

            for roi_name in roi_dict:
                try:
                    roi = ROI(
                        roi_name,
                        np.array(roi_dict[roi_name]['coords']),
                        label=roi_dict[roi_name]['label']
                    )
                except errors.InvalidROIError as e:
                    if skip_invalid:
                        log.warn("Skipping invalid ROI ({}): {}".format(roi_name, e))
                        continue
                    else:
                        raise
                else:
                    if simplify_tolerance is not None:
                        roi.simplify(simplify_tolerance)
                    self.rois.append(roi)
        if process:
            self.process_rois()
        log.debug(f"Loaded ROIs from {path}")
        return len(self.rois)

    def load_json_roi(
        self,
        path: str,
        *,
        scale: int = 1,
        process: bool = True,
        skip_invalid: bool = True
    ) -> int:
        """Load ROIs from a JSON file.

        JSON file must contain a 'shapes' key, with a list of dictionaries
        containing a 'points' key, whose value is a list of (x, y) coordinates.

        Args:
            path (str): Path to JSON file.
            scale (int): Scale factor to apply to ROI coordinates. Defaults to 1.
            process (bool): Process ROIs after loading. Defaults to True.

        """
        # Clear any previously loaded ROIs.
        self.rois = []

        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], scale).astype(np.int64)
            roi_name = self.get_next_roi_name()
            try:
                self.rois.append(ROI(roi_name, area_reduced))
            except errors.InvalidROIError as e:
                if skip_invalid:
                    log.warn("Skipping invalid ROI ({}): {}".format(roi_name, e))

        if process:
            self.process_rois()
        if self.roi_method == 'auto':
            self.roi_method = 'inside'
        return len(self.rois)

    def masked_thumb(self, background: str = 'white', **kwargs) -> np.ndarray:
        """Return a masked thumbnail of a slide, using QC and/or ROI masks.

        Args:
            background (str, optional): Background color. Defaults to 'white'.

        Keyword args:
            **kwargs: Keyword arguments passed to :meth:`WSI.thumb()`.

        Returns:
            np.ndarray: Masked thumbnail image.

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
            from slideflow.slide.qc import Otsu, GaussianV2
            sf.log.debug(
                "Applying Otsu's thresholding & Gaussian blur filter "
                "to stain norm context"
            )
            _blur_mask = GaussianV2()(image)
            qc_mask = Otsu()(image, mask=_blur_mask)
        # Mask by ROI and QC, if applied.
        # Use white as background for masked areas.
        if qc_mask is not None:
            qc_img = img_as_ubyte(qc_mask)
            mask = ~cv2.resize(qc_img, (image.shape[1], image.shape[0]))
        if roi_mask is not None:
            roi_img = img_as_ubyte(roi_mask)
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

    def mpp_to_dim(self, mpp: float) -> Tuple[int, int]:
        width = int((self.mpp * self.dimensions[0]) / mpp)
        height = int((self.mpp * self.dimensions[1]) / mpp)
        return (width, height)

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
        _compatible = sf.util.is_tile_size_compatible(
            config['tile_px'],
            config['tile_um'],
            self.tile_px,
            self.tile_um
        )
        if not _compatible:
            raise ValueError(
                "Slide tile size (tile_px={}, tile_um={}) does not match the "
                "model (tile_px={}, tile_um={}).".format(
                    self.tile_px, self.tile_um,
                    config['tile_px'], config['tile_um']
            ))
        log.info("Calculating whole-slide prediction...")
        heatmap = Heatmap(self, model, generate=True, **kwargs)
        preds = heatmap.predictions.reshape(-1, heatmap.predictions.shape[-1])
        preds = np.nanmean(np.ma.masked_where(preds == sf.heatmap.MASK, preds), axis=0).filled()
        if heatmap.uncertainty is not None:
            unc = heatmap.uncertainty.reshape(-1, heatmap.uncertainty.shape[-1])
            unc = np.nanmean(np.ma.masked_where(unc == sf.heatmap.MASK, unc), axis=0).filled()
            return preds, unc
        else:
            return preds

    def preview(
        self,
        rois: bool = True,
        thumb_kwargs: Optional[Dict] = None,
        low_res: bool = True,
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
            thumb_kwargs (Optional[Dict], optional): Keyword arguments to pass
                to the thumb method. Defaults to None.
            low_res (bool, optional): Use low resolution thumbnail. Defaults to
                True.
        """
        if 'show_progress' not in kwargs:
            kwargs['show_progress'] = (self.pb is None)
        generator = self.build_generator(
            dry_run=True,
            deterministic=False,
            **kwargs
        )
        if thumb_kwargs is None:
            thumb_kwargs = dict(low_res=low_res)
        if generator is None:
            return self.thumb(rois=rois,  **thumb_kwargs)
        locations = []
        for tile_dict in generator():
            locations += [tile_dict['loc']]
        log.debug(f"Previewing with {len(locations)} extracted tile locations.")
        return self.thumb(
            coords=locations, rois=rois, **thumb_kwargs
        )

    def process_rois(self):
        """Process loaded ROIs and apply to the slide grid.

        Returns:
            int: Number of ROIs processed.

        """
        # Load annotations as shapely.geometry objects.
        if self.roi_method != 'ignore':
            self._find_and_process_holes()

        # Regenerate the grid to reflect the newly-loaded ROIs.
        self._build_coord()

        # Re-apply any existing QC mask, now that the coordinates have changed.
        if self.has_non_roi_qc():
            self.apply_qc_mask()

        return len(self.rois)

    def _find_and_process_holes(self):
        """Find and process holes in ROIs."""

        from shapely.strtree import STRtree

        self.rois.sort(key=lambda x: x.poly.area, reverse=True)

        outer_rois = []

        labels = list(set([roi.label for roi in self.rois]))

        for label in labels:

            rois = [roi for roi in self.rois if roi.label == label]
            polygons = [roi.poly for roi in self.rois if roi.label == label]
            strtree = STRtree(polygons)

            for roi, poly in zip(rois, polygons):

                if version.parse(shapely_version) < version.parse('2.0.0'):
                    possible_containers = strtree.query(poly)
                else:
                    possible_containers_idx = strtree.query(poly)
                    possible_containers = [polygons[i] for i in possible_containers_idx]

                # Filter out the polygon itself
                possible_containers = [p for p in possible_containers if p != poly]

                # Check if the polygon is contained by another
                contained_by = [p for p in possible_containers if p.contains(poly)]

                if not contained_by:
                    # Polygon is an outer polygon
                    outer_rois.append(roi)
                else:
                    # Polygon is a hole, find its immediate outer polygon
                    # Sort by area (smallest to largest) to find the closets outer.
                    contained_by.sort(key=lambda x: x.area)
                    immediate_outer_poly = contained_by[0]
                    immediate_outer_roi = rois[polygons.index(immediate_outer_poly)]

                    # If the immediate outer is not already listed as an outer,
                    # then the immediate outer is a hole and this polygon is a nested
                    # polygon within a hole and should be treated as an outer.
                    if immediate_outer_roi not in outer_rois:
                        outer_rois.append(roi)
                    else:
                        # Otherwise, add the polygon to the immediate outer as a hole
                        immediate_outer_roi.add_hole(roi)

        # Restrict the ROIs to only outer polygons, which have now had the holes applied.
        self.rois = outer_rois

    def qc(
        self,
        method: Union[str, Callable, List[Callable]],
        *,
        blur_radius: int = 3,
        blur_threshold: float = 0.02,
        filter_threshold: float = 0.6,
        blur_mpp: Optional[float] = None,
        pool: Optional["mp.pool.Pool"] = None
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
            method.insert(idx, sf.slide.qc.GaussianV2(mpp=blur_mpp,
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
            if pool is not None:
                try:
                    qc.pool = pool  # type: ignore
                except Exception as e:
                    log.debug(f"Unable to set pool for QC method {qc}")
            mask = qc(self)
            if mask is not None:
                img = self.apply_qc_mask(mask, filter_threshold=filter_threshold)
        dur = f'(time: {time.time()-starttime:.2f}s)'
        log.debug(f'QC ({method}) complete for slide {self.shortname} {dur}')
        return img

    def remove_qc(self) -> None:
        self.qc_masks = [m for m in self.qc_masks if m.is_roi]
        self._build_coord()
        log.debug(f'QC removed from slide {self.shortname}')

    def remove_roi_qc(self) -> None:
        """Remove ROI-based QC from the slide."""
        self.qc_masks = [m for m in self.qc_masks if not m.is_roi]
        if len(self.qc_masks):
            self.apply_qc_mask()

    def remove_roi(
        self,
        idx: Union[int, List[int]],
        *,
        process: bool = True
    ) -> None:
        """Remove an ROI from the slide.

        Args:
            idx (int, list(int)): Index or indices of the ROI(s) to remove.

        Keyword Args:
            process (bool): Process ROIs after removing. Defaults to True.

        """
        if isinstance(idx, int):
            idx = [idx]
        for i in sorted(idx, reverse=True):
            del self.rois[i]
        if process:
            self.process_rois()

    def show_alignment(
        self,
        slide: "WSI",
        mpp: float = 4
    ) -> Image.Image:
        """Show aligned thumbnail of another slide."""
        if not isinstance(slide, WSI):
            raise TypeError("Can only align to another slide.")

        # Calculate thumbnails for alignment.
        our_thumb = np.array(self.thumb(mpp=mpp))
        their_thumb = np.array(slide.thumb(mpp=mpp))

        # Return an image of a thumbnail of the given slide,
        # aligned to this slide.
        return Image.fromarray(align_image(their_thumb, our_thumb))

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
        *,
        coords: Optional[List[int]] = None,
        rect_linewidth: int = 2,
        rect_color: str = 'black',
        rois: bool = False,
        linewidth: int = 2,
        color: str = 'black',
        use_associated_image: bool = False,
        low_res: bool = False,
    ) -> Image.Image:
        """Generate a PIL Image of the slide thumbnail, with ROI overlay.

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
        thumb = Image.fromarray(np_thumb).resize((width, height))

        if coords:
            draw = ImageDraw.Draw(thumb)
            ratio = width / self.dimensions[0]
            wh = (self.full_extract_px * ratio) / 2
            for (x, y) in coords:  # type: ignore
                x, y = x * ratio, y * ratio  # type: ignore
                coords = (x-wh, y-wh, x+wh, y+wh)  # type: ignore
                draw.rectangle(coords, outline=rect_color, width=rect_linewidth)

        if rois and len(self.rois):
            draw = ImageDraw.Draw(thumb)
            roi_polys = [r.scaled_poly(roi_scale) for r in self.rois]
            for roi in self.rois:
                for hole in roi.holes.values():
                    roi_polys.append(hole.scaled_poly(roi_scale))
            for i, poly in enumerate(roi_polys):
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.coords.xy
                    zipped = list(zip(x.tolist(), y.tolist()))
                    draw.line(zipped, joint='curve', fill=color, width=linewidth)
                elif poly.geom_type in ('MultiPolygon', 'GeometryCollection'):
                    for part in poly.geoms:
                        if part.is_empty or part.geom_type != 'Polygon':
                            continue
                        x, y = part.exterior.coords.xy
                        zipped = list(zip(x.tolist(), y.tolist()))
                        draw.line(zipped, joint='curve', fill=color, width=linewidth)
                else:
                    sf.log.error(f"Unable to plot ROI {i}, unknown geometry type: {poly.geom_type}")
            return thumb
        else:
            return thumb

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
                If True or 'coord', will return X/Y coordinates of the tile center
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
            'loc_x'         Image tile center x location (if ``incl_loc`` provided)
            'loc_y'         Image tile center y location (if ``incl_loc`` provided)
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
                If True or 'coord', will return X/Y coordinates of the tile center
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
            'loc_x'         Image tile center x location (if ``incl_loc`` provided)
            'loc_y'         Image tile center y location (if ``incl_loc`` provided)
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

    def verify_alignment(
        self,
        slide: "WSI",
        mpp: float = 4
    ) -> float:
        """Verify alignment to another slide by calculating MSE."""
        if not isinstance(slide, WSI):
            raise TypeError("Can only align to another slide.")

        # Calculate thumbnails for alignment.
        our_thumb = np.array(self.thumb(mpp=mpp))
        their_thumb = np.array(slide.thumb(mpp=mpp))

        aligned_theirs = align_image(their_thumb, our_thumb)

        theirs_gray = cv2.cvtColor(aligned_theirs, cv2.COLOR_BGR2GRAY)
        ours_gray = cv2.cvtColor(our_thumb, cv2.COLOR_BGR2GRAY)

        return compute_alignment_mse(theirs_gray, ours_gray)

    def view(self):
        """Open the slide in Slideflow Studio for interactive display.

        See :ref:`studio` for more information.

        """
        from slideflow.studio import Studio

        studio = Studio()
        studio.load_slide(self.path, stride=self.stride_div, tile_px=self.tile_px, tile_um=self.tile_um)
        studio.run()
