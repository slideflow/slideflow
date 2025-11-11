'''Functions for slide extraction reports (PDF).'''

from __future__ import absolute_import, division, print_function

import io
import os
import tempfile
import pandas as pd
import numpy as np
import cv2

from fpdf import FPDF, XPos, YPos
from PIL import Image, ImageDraw, UnidentifiedImageError
from datetime import datetime
from os.path import join, exists
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Tuple

import slideflow as sf
from slideflow.util import log, path_to_name  # noqa F401

if TYPE_CHECKING:
    import pandas as pd

# -----------------------------------------------------------------------------

def render_thumbnail(report: "SlideReport") -> Optional["Image.Image"]:
    return report.thumb


def render_image_row(report: "SlideReport") -> Optional[bytes]:
    return report.image_row()

# -----------------------------------------------------------------------------

class SlideReport:
    '''Report to summarize tile extraction from a slide, including
    example images of extracted tiles.
    '''

    def __init__(
        self,
        images: List[bytes],
        path: str,
        tile_px: int,
        tile_um: Union[int, str],
        *,
        thumb: Optional[Image.Image] = None,
        thumb_coords: Optional[np.ndarray] = None,
        data: Optional[Dict[str, Any]] = None,
        compress: bool = True,
        ignore_thumb_errors: bool = False
    ) -> None:
        """Creates a slide report summarizing tile extraction, with some example
        extracted images.

        Args:
            images (list(str)): List of JPEG image strings (example tiles).
            path (str): Path to slide.
            data (dict, optional): Dictionary of slide extraction report
                metadata. Expected keys may include 'blur_burden', 'num_tiles',
                'locations', and 'qc_mask'. Defaults to None.
            compress (bool, optional): Compresses images to reduce image sizes.
                Defaults to True.
            thumb (PIL.Image): Thumbnail of slide. Defaults to None.
            thumb_coords (np.ndarray): Array of (x, y) tile extraction
                coordinates, for display on the thumbnail. Defaults to None.
            ignore_thumb_errors (bool): Ignore errors raised when attempting
                to create a slide thumbnail.


        """
        self.data = data
        self.path = path
        self.tile_px = tile_px
        self.tile_um = tile_um
        if data is not None:
            self.has_rois = 'num_rois' in data and data['num_rois'] > 0
        else:
            self.has_rois = False
        self.timestamp = str(datetime.now())

        # Thumbnail
        self.ignore_thumb_errors = ignore_thumb_errors
        self.thumb_coords = thumb_coords
        if thumb is not None:
            self._thumb = Image.fromarray(np.array(thumb)[:, :, 0:3])
        else:
            self._thumb = None

        if not compress:
            self.images = images  # type: List[bytes]
        else:
            self.images = [self._compress(img) for img in images]

    @property
    def thumb(self):
        if self._thumb is None:
            try:
                self.calc_thumb()
            except Exception:
                if self.ignore_thumb_errors:
                    return None
                else:
                    raise
        return self._thumb

    @property
    def blur_burden(self) -> Optional[float]:
        """Metric defined as the proportion of non-background slide
        with high blur. Only calculated if both Otsu and Blur QC is used.

        Returns:
            float
        """
        if self.data is None:
            return None
        if 'blur_burden' in self.data:
            return self.data['blur_burden']
        else:
            return None

    @property
    def num_tiles(self) -> Optional[int]:
        """Number of tiles extracted.

        Returns:
            int
        """
        if self.data is None:
            return None
        if 'num_tiles' in self.data:
            return self.data['num_tiles']
        else:
            return None

    @property
    def locations(self) -> Optional["pd.DataFrame"]:
        """DataFrame with locations of extracted tiles, with the following
        columns:

        ``loc_x``: Extracted tile x coordinates (as saved in TFRecords).
        Calculated as the full coordinate value / 10.

        ``loc_y``: Extracted tile y coordinates (as saved in TFRecords).
        Calculated as the full coordinate value / 10.

        ``grid_x``: First dimension index of the tile extraction grid.

        ``grid_y``: Second dimension index of the tile extraction grid.

        ``gs_fraction``: Grayspace fraction. Only included if grayspace
        filtering is used.

        ``ws_fraction``: Whitespace fraction. Only included if whitespace
        filtering is used.

        Returns:
            pandas.DataFrame

        """
        if self.data is None:
            return None
        if 'locations' in self.data:
            return self.data['locations']
        else:
            return None

    @property
    def qc_mask(self) -> Optional[np.ndarray]:
        """Numpy array with the QC mask, of shape WSI.grid and type bool
        (True = include tile, False = discard tile)

        Returns:
            np.ndarray
        """
        if self.data is None:
            return None
        if 'qc_mask' in self.data:
            return self.data['qc_mask']
        else:
            return None

    def calc_thumb(self) -> None:
        try:
            wsi = sf.WSI(
                self.path,
                tile_px=self.tile_px,
                tile_um=self.tile_um,
                verbose=False,
            )
        except sf.errors.SlideMissingMPPError:
            wsi = sf.WSI(
                self.path,
                tile_px=self.tile_px,
                tile_um=self.tile_um,
                verbose=False,
                mpp=1   # Force MPP to 1 to add support for slides missing MPP.
                        # The MPP does not need to be accurate for thumbnail generation.
            )
        self._thumb = wsi.thumb(
            coords=self.thumb_coords,
            rois=self.has_rois,
            low_res=True,
            width=512,
            rect_linewidth=1,
        )
        self._thumb = Image.fromarray(np.array(self._thumb)[:, :, 0:3])

    def _compress(self, img: bytes) -> bytes:
        with io.BytesIO() as output:
            pil_img = Image.open(io.BytesIO(img)).convert('RGB')
            if pil_img.height > 256:
                pil_img = Image.fromarray(
                    cv2.resize(np.array(pil_img), [256, 256])
                )
            pil_img.save(output, format="JPEG", quality=75)
            return output.getvalue()

    def image_row(self) -> Optional[bytes]:
        '''Merges images into a single row of images'''
        if not self.images:
            return None
        pil_images = [Image.open(io.BytesIO(i)) for i in self.images]
        widths, heights = zip(*(pi.size for pi in pil_images))
        total_width = sum(widths)
        max_height = max(heights)
        row_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for image in pil_images:
            row_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]
        with io.BytesIO() as output:
            row_image.save(output, format="JPEG", quality=75)
            return output.getvalue()

# -----------------------------------------------------------------------------

class LowerMagSlideReport(SlideReport):
    """Report for lower magnification tile extraction (e.g., 40x → 10x)."""

    def __init__(
        self,
        images: List[bytes],
        path: str,
        tile_px: int,
        tile_um: Union[int, str],
        source_tile_um: Union[int, str],
        mag_ratio: int,
        *,
        thumb: Optional[Image.Image] = None,
        thumb_coords: Optional[np.ndarray] = None,
        source_thumb_coords: Optional[np.ndarray] = None,
        data: Optional[Dict[str, Any]] = None,
        compress: bool = True,
        ignore_thumb_errors: bool = False
    ) -> None:
        super().__init__(
            images=images,
            path=path,
            tile_px=tile_px,
            tile_um=tile_um,
            thumb=thumb,
            thumb_coords=thumb_coords,
            data=data,
            compress=compress,
            ignore_thumb_errors=ignore_thumb_errors
        )
        self.source_tile_um = source_tile_um
        self.mag_ratio = int(mag_ratio)
        self.source_thumb_coords = source_thumb_coords

    @property
    def combination_efficiency(self) -> Optional[float]:
        """% of source tiles successfully combined into target tiles."""
        total_source = self.data.get('total_source_tiles', 0)
        used_source = self.data.get('source_tiles_used', 0)
        if not total_source:
            return None
        return (used_source / total_source) * 100.0

    @property
    def spatial_coverage_reduction(self) -> Optional[float]:
        """% reduction in number of tiles (spatial coverage)."""
        total_source = self.data.get('total_source_tiles', 0)
        num_combined = self.data.get('num_tiles', 0)
        if not total_source:
            return None
        return (1.0 - (num_combined / total_source)) * 100.0

    @property
    def incomplete_groups_percent(self) -> Optional[float]:
        """% of tile groups that were incomplete."""
        incomplete = self.data.get('incomplete_groups', 0)
        combined = self.data.get('num_tiles', 0)
        total_groups = incomplete + combined
        if not total_groups:
            return None
        return (incomplete / total_groups) * 100.0

    def _get_target_tfrecord_path(self) -> Optional[str]:
        """Get the path to the target TFRecord file for this slide."""
        try:
            import slideflow as sf
            from os.path import exists, join
            
            # Build target TFRecord path based on slide name and magnifications
            slide_name = sf.util.path_to_name(self.path)
            
            # Construct target directory name: "{target_px}px_{target_mag}_from_{source_mag}"
            target_px = getattr(self, 'target_tile_px', self.tile_px)  # Default to current tile_px if not set
            source_mag = getattr(self, 'source_tile_um')
            target_mag = getattr(self, 'tile_um')
            
            target_dir_name = f"{target_px}px_{target_mag}_from_{source_mag}"
            
            # Look for TFRecord in project structure
            # Assuming we can get to the project root from self.path
            project_root = self._find_project_root()
            if project_root:
                target_path = join(project_root, 'tfrecords', target_dir_name, f"{slide_name}.tfrecords")
                if exists(target_path):
                    return target_path
                    
        except Exception as e:
            import slideflow as sf
            log.debug(f"Error finding target TFRecord path: {e}")
            
        return None
    
    def _find_project_root(self) -> Optional[str]:
        """Find the project root directory."""
        from os.path import dirname, exists, join
        
        # Start from slide path and work up to find project root
        current = dirname(self.path)
        for _ in range(5):  # Limit search depth
            if exists(join(current, 'tfrecords')):
                return current
            current = dirname(current)
        return None

    def _extract_target_locations(self) -> List[Tuple[int, int]]:
        """Return target grid locations as list of (grid_x, grid_y)."""
        # Try to read target coordinates directly from the target TFRecord
        if hasattr(self, 'source_tile_um') and hasattr(self, 'tile_um'):
            target_tfrecord_path = self._get_target_tfrecord_path()
            if target_tfrecord_path:
                try:
                    target_coords = sf.io.get_locations_from_tfrecord(target_tfrecord_path)
                    return target_coords
                except Exception as e:
                    log.debug(f"Could not read target TFRecord {target_tfrecord_path}: {e}")
        
        # First check if we have the combined_grid_coords attribute set directly  
        if hasattr(self, 'combined_grid_coords') and self.combined_grid_coords:
            return list(self.combined_grid_coords)
        
        # Otherwise look in the data dict
        locs = self.data.get('locations')
        if locs is None:
            return []
        # If DataFrame, prefer grid_x/grid_y; fall back to loc_x/loc_y if those are grids
        if isinstance(locs, pd.DataFrame):
            if {'grid_x', 'grid_y'}.issubset(locs.columns):
                gx = locs['grid_x'].astype(int).tolist()
                gy = locs['grid_y'].astype(int).tolist()
                return list(zip(gx, gy))
            elif {'loc_x', 'loc_y'}.issubset(locs.columns):
                # assume loc_* are already grid indices at lower mag
                gx = locs['loc_x'].astype(int).tolist()
                gy = locs['loc_y'].astype(int).tolist()
                return list(zip(gx, gy))
            else:
                return []
        # Else assume iterable of tuples
        try:
            out = []
            for t in locs:
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    out.append((int(t[0]), int(t[1])))
            return out
        except Exception:
            return []

    def calc_thumb(self) -> None:
        """Draw source (black) and target (red) overlays on the WSI thumbnail.
        Magnification-agnostic: no hardcoded DS/512/10x. Uses slide metadata + report fields."""
        import numpy as np
        from PIL import Image, ImageDraw
        import slideflow as sf

        # ----- 0) Open WSI (params here don't affect geometry; we map via level-0 size) -----
        try:
            wsi = sf.WSI(
                self.path,
                tile_px=getattr(self, 'tile_px'),
                tile_um=getattr(self, 'tile_um'),
                verbose=False,
            )
        except Exception as e:
            # If slideflow exposes specific error types, include them; otherwise catch generic Exception.
            try:
                SlideLoadError = sf.errors.SlideLoadError
                SlideMissingMPPError = sf.errors.SlideMissingMPPError
            except Exception:
                SlideLoadError = None
                SlideMissingMPPError = None

            is_slide_error = (
                (SlideLoadError and isinstance(e, SlideLoadError)) or
                (SlideMissingMPPError and isinstance(e, SlideMissingMPPError))
            )

            # Log and gracefully skip this slide's overlay drawing.
            log.warning(f"Skipping thumbnail overlay for {self.path} due to WSI open error: {e}")
            # Mark as skipped so higher-level reports can count this.
            self.data = self.data or {}
            self.data['skipped'] = True

            # Try to generate a basic thumbnail safely; if that fails, create a blank placeholder.
            try:
                base_thumb = sf.WSI(self.path, tile_px=getattr(self, 'tile_px', 512), tile_um=getattr(self, 'tile_um')).thumb(
                    coords=None, rois=getattr(self, 'has_rois', False), low_res=True, width=1024, rect_linewidth=1
                )
                thumb = Image.fromarray(np.asarray(base_thumb)[:, :, :3])
            except Exception as e_thumb:
                log.debug(f"Couldn't create fallback thumbnail for {self.path}: {e_thumb}; using blank placeholder.")
                thumb = Image.new('RGB', (1024, 1024), (255, 255, 255))

            # Save placeholder and exit early — don't attempt any overlay drawing.
            self._thumb = thumb
            return

        # ----- 1) Build thumbnail & get level-0 geometry -----
        base_thumb = wsi.thumb(coords=None, rois=self.has_rois, low_res=True, width=1024, rect_linewidth=1)
        thumb = Image.fromarray(np.asarray(base_thumb)[:, :, :3])
        draw = ImageDraw.Draw(thumb)

        try:
            W0, H0 = wsi.slide.level_dimensions[0]   # level-0 size in px
        except Exception as e:
            log.warning(f"Could not read level-0 dimensions for {self.path}: {e}")
            self._thumb = thumb
            return

        # ----- 2) Compute coordinate→THUMB scale -----
        # For LowerMagSlideReport, coordinates are stored in level-0 WSI pixel space,
        # so we use a simple scale: level-0 pixels → thumbnail pixels.
        # For regular SlideReport, coordinates may be in a "DRAW" space requiring transformation.

        # Simple scale for level-0 coordinates → thumbnail
        scale_level0_to_thumb = thumb.width / float(W0)

        # Check if coordinates are in level-0 space (LowerMagSlideReport) or need transformation
        # LowerMagSlideReport stores coordinates directly from TFRecords which are level-0 pixels
        coords_are_level0 = isinstance(self, LowerMagSlideReport)

        if coords_are_level0:
            # Coordinates are already in level-0 WSI pixel space
            # Just scale directly to thumbnail
            scale_draw_to_thumb = scale_level0_to_thumb
            log.debug(f"[calc_thumb] Using level-0 coordinate scale: {scale_draw_to_thumb:.4f} (thumb.width={thumb.width}, W0={W0})")
        else:
            # Original logic for regular SlideReport where coordinates may be in DRAW space
            # base_mpp (µm/px) at level-0
            try:
                base_mpp = float(getattr(wsi, 'mpp', 0.0)) or float(getattr(wsi, 'level_mpp', [0.0])[0] or 0.0)
                if base_mpp <= 0:
                    raise ValueError
            except Exception:
                base_mpp = 0.25  # conservative fallback

            # Prefer a physical mpp for the DRAW level, else fall back to magnification strings
            mpp_draw = None
            try:
                if isinstance(self.tile_um, (int, float)) and isinstance(self.tile_px, (int, float)) and self.tile_px > 0:
                    mpp_draw = float(self.tile_um) / float(self.tile_px)  # µm per DRAW px
            except Exception:
                mpp_draw = None

            if mpp_draw and mpp_draw > 0:
                d_draw = mpp_draw / base_mpp
            else:
                # Fallback: try to parse a magnification string for the DRAW level
                def _to_mag(val):
                    try:
                        return float(sf.util.to_mag(val)) if isinstance(val, str) else float(val)
                    except Exception:
                        return None

                mag_draw = _to_mag(getattr(self, 'tile_um', None))
                if mag_draw and mag_draw > 0:
                    d_draw = 10.0 / (mag_draw * base_mpp)
                else:
                    # Last resort: infer draw mag from source mag and mag_ratio if available
                    src_mag = _to_mag(getattr(self, 'source_tile_um', None))
                    ratio  = float(getattr(self, 'mag_ratio', 0) or 0)
                    if src_mag and ratio and ratio > 0:
                        mag_draw = src_mag / ratio
                        d_draw = 10.0 / (mag_draw * base_mpp)
                    else:
                        # Absolute fallback: assume draw==level-0 (no extra scaling)
                        d_draw = 1.0
                        log.debug("[calc_thumb] Falling back to d_draw=1.0 (no draw-level metadata).")

            scale_draw_to_thumb = d_draw * (thumb.width / float(W0))

        # ----- 3) Resolve box sizes -----
        # For LowerMagSlideReport with level-0 coordinates, calculate actual extraction sizes
        # using the same method as wsi.thumb() (which uses wsi.full_extract_px)

        if coords_are_level0:
            # For source tiles: need to create a temporary WSI with source magnification
            # to get the correct full_extract_px value
            try:
                source_wsi = sf.WSI(
                    self.path,
                    tile_px=self.tile_px,  # This doesn't matter for full_extract_px calculation
                    tile_um=getattr(self, 'source_tile_um'),
                    verbose=False,
                )
                s_box_draw = float(source_wsi.full_extract_px)  # Actual extraction size in level-0 pixels
                log.debug(f"[calc_thumb] Calculated source box size from WSI: {s_box_draw} pixels")
            except Exception as e:
                log.error(f"Failed to calculate source box size from WSI: {e}")
                raise

            # Target box is mag_ratio times the source box
            t_box_draw = s_box_draw * float(getattr(self, 'mag_ratio'))
            log.debug(f"[calc_thumb] Box sizes in level-0 pixels: source={s_box_draw:.0f}, target={t_box_draw:.0f}")
        else:
            # Original logic for regular SlideReport
            try:
                s_box_draw = float(getattr(self, 'source_tile_px'))
            except Exception as e:
                s_box_draw = float(getattr(self, 'tile_px')) / max(1.0, float(getattr(self, 'mag_ratio', 1)))
            try:
                t_box_draw = float(getattr(self, 'target_tile_px'))
            except Exception as e:
                t_box_draw = float(getattr(self, 'tile_px'))

        # ----- 4) Draw rectangles (centers are already in DRAW coordinates) -----
        def _draw(centers_draw, box_draw, color, width, label):
            if centers_draw is None:
                log.debug(f"DEBUG _draw: {label} centers_draw is None")
                return
            if len(centers_draw) == 0:
                log.debug(f"DEBUG _draw: {label} centers_draw is empty")
                return
            half = box_draw / 2.0
            w_th = box_draw * scale_draw_to_thumb
            for i, coord_pair in enumerate(centers_draw):
                # Handle both (x, y) tuples and [x, y] arrays
                if len(coord_pair) == 2:
                    cx, cy = coord_pair[0], coord_pair[1]
                else:
                    log.error(f"DEBUG: Invalid coordinate format: {coord_pair}")
                    continue
                x_draw = float(cx) - half
                y_draw = float(cy) - half
                x_th = x_draw * scale_draw_to_thumb
                y_th = y_draw * scale_draw_to_thumb
                rect_coords = [x_th, y_th, x_th + w_th, y_th + w_th]
                draw.rectangle(rect_coords, outline=color, width=width)

        source_coords = getattr(self, 'source_thumb_coords', None)
        target_coords = getattr(self, 'target_thumb_coords', None)

        # Debug the actual coordinate values received in calc_thumb
        if source_coords is None:
            log.debug(f"DEBUG calc_thumb: source_coords is None for {self.path}")

        if target_coords is None:
            log.debug(f"DEBUG calc_thumb: target_coords is None for {self.path}")
        
        # Use black for source tiles, red for target tiles - with thin lines
        _draw(source_coords, s_box_draw, (0, 0, 0), 2, "SOURCE")
        _draw(target_coords, t_box_draw, (255, 0, 0), 3, "TARGET_RED") 
        
        # ----- 5) Save -----
        self._thumb = thumb

    def create_combination_visualization(self) -> Optional[bytes]:
        """Create a visualization showing source→target tile relationships."""
        target_locations = self._extract_target_locations()
        if not target_locations:
            return None
        try:
            import matplotlib.pyplot as plt
            with sf.util.matplotlib_backend('Agg'):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Estimate source locations by upsampling each target grid cell
                src = []
                R = self.mag_ratio
                for tx, ty in target_locations:
                    for i in range(R):
                        for j in range(R):
                            src.append((tx * R + i, ty * R + j))

                if src:
                    sx, sy = zip(*src)
                    ax1.scatter(sx, sy, s=2, alpha=0.6)
                    ax1.set_title(f'Source Tiles ({self.source_tile_um}) — est. {len(src)}')
                    ax1.set_xlabel('Grid X'); ax1.set_ylabel('Grid Y'); ax1.grid(True, alpha=0.3)

                tx, ty = zip(*target_locations)
                ax2.scatter(tx, ty, s=8, alpha=0.8)
                ax2.set_title(f'Combined Tiles ({self.tile_um}) — {len(target_locations)}')
                ax2.set_xlabel('Grid X'); ax2.set_ylabel('Grid Y'); ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                    plt.savefig(temp.name, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    with open(temp.name, 'rb') as f:
                        return f.read()
        except Exception as e:
            log.error(f"Error creating combination visualization for {self.path}: {e}")
            return None

    def create_before_after_comparison(self) -> Optional[bytes]:
        """Create before/after example grids by visualizing a split of combined tiles."""
        if not self.images:
            return None
        try:
            import matplotlib.pyplot as plt
            n_examples = min(3, len(self.images))
            with sf.util.matplotlib_backend('Agg'):
                fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, 8))
                # Normalize axes shape for n_examples==1
                if n_examples == 1:
                    axes = np.array([[axes[0]], [axes[1]]])

                for i in range(n_examples):
                    img = Image.open(io.BytesIO(self.images[i])).convert('RGB')
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'Combined Tile {i+1} ({self.tile_um})')
                    axes[0, i].axis('off')

                    img_array = np.array(img)
                    h, w = img_array.shape[:2]
                    R = max(self.mag_ratio, 1)
                    tile_h, tile_w = h // R, w // R
                    before_grid = np.zeros_like(img_array)

                    for r in range(R):
                        for c in range(R):
                            y0, y1 = r * tile_h, (r + 1) * tile_h
                            x0, x1 = c * tile_w, (c + 1) * tile_w
                            tile_region = img_array[y0:y1, x0:x1].copy()
                            # white borders
                            tile_region[:2, :] = 255
                            tile_region[-2:, :] = 255
                            tile_region[:, :2] = 255
                            tile_region[:, -2:] = 255
                            before_grid[y0:y1, x0:x1] = tile_region

                    axes[1, i].imshow(before_grid)
                    axes[1, i].set_title(f'Source Tiles ({self.source_tile_um}) — {R}×{R}')
                    axes[1, i].axis('off')

                plt.tight_layout()
                with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                    plt.savefig(temp.name, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    with open(temp.name, 'rb') as f:
                        return f.read()
        except Exception as e:
            log.error(f"Error creating before/after comparison for {self.path}: {e}")
            return None

# -----------------------------------------------------------------------------

class ExtractionPDF(FPDF):
    # Length is 220
    def __init__(
        self,
        *args,
        title: str = 'Tile extraction report',
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.title_msg = title

    def header(self) -> None:
        package_directory = os.path.dirname(os.path.abspath(__file__))
        logo = join(package_directory, 'slideflow-logo-name-small.jpg')

        self.set_font('Arial', size=9)
        self.cell(70)  # Moves right
        self.set_text_color(70, 70, 70)
        self.cell(50, 8, 'Intended for Research Use Only', align='C')
        self.ln(10)

        self.set_text_color(0, 0, 0)
        # Framed title
        self.set_font('Arial', 'B', 16)
        top = self.y  # type: ignore
        self.cell(40, 10, self.title_msg, 0, 1)
        self.y = top
        self.cell(150)
        self.image(logo, 160, 20, w=40)
        # Line break
        self.line(10, 30, 200, 30)
        self.ln(10)
        self.set_font('Arial', '', 10)
        top = self.y
        datestring = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        self.cell(20, 10, f'Generated: {datestring}', 0, 1)
        self.y = top
        self.cell(150)
        self.cell(40, 10, sf.__version__, align='R')
        self.ln(15)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + ' of {nb}', 0, 0, 'C')


class ExtractionReport:
    """Creates a PDF report summarizing extracted tiles, from a collection of
    tile extraction reports."""

    def __init__(
        self,
        reports: List[SlideReport],
        meta: SimpleNamespace = None,
        bb_threshold: float = 0.05,
        title: str = 'Tile extraction report',
        *,
        pool: Optional[Any] = None
    ) -> None:
        """Initializer.

        Args:
            reports (list(:class:`SlideReport`)): List of SlideReport objects.
        """
        import matplotlib.pyplot as plt

        self.bb_threshold = bb_threshold
        self.reports = reports
        self.meta = meta
        pdf = ExtractionPDF(title=title)
        pdf.alias_nb_pages()
        pdf.add_page()

        # Render thumbnails, if a multiprocesing pool is provided.
        if pool is not None:
            log.debug("Rendering thumbnails with pool.")
            thumbnails = pool.map(render_thumbnail, reports)
            log.debug("Rendering tile images with pool.")
            image_rows = pool.map(render_image_row, reports)
            log.debug("Report render complete.")
        else:
            thumbnails = [r.thumb for r in reports]
            image_rows = [r.image_row() for r in reports]

        if meta is not None and hasattr(meta, 'ws_frac'):
            n_tiles = np.array([r.num_tiles for r in reports if r is not None])
            bb = np.array([r.blur_burden for r in reports if r is not None])
            bb_names = [r.path for r in reports if r is not None]
            self.warn_txt = ''
            for slide, b in zip(bb_names, bb):
                if b is not None and b > self.bb_threshold:
                    self.warn_txt += f'{slide},{b}\n'

            with sf.util.matplotlib_backend('Agg'):
                if np.any(n_tiles) and self.num_tiles_chart(n_tiles):
                    with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                        plt.savefig(temp.name)
                        pdf.image(temp.name, 107, pdf.y, w=50)
                        plt.close()

                if np.any(bb) and self.blur_chart(bb):
                    with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                        plt.savefig(temp.name)
                        pdf.image(temp.name, 155, pdf.y, w=50)
                        plt.close()

            # Bounding box
            pdf.set_x(20)
            pdf.set_y(pdf.y+5)
            x = pdf.x
            y = pdf.y
            pdf.set_line_width(0.5)
            pdf.set_draw_color(120, 120, 120)
            pdf.cell(95, 30, '', 1, 0, 'L')
            pdf.set_x(x)

            # First column
            pdf.set_y(y+1)
            pdf.set_font('Arial', style='B')
            for m in ('Tile size (px)', 'Tile size (um)', 'QC', 'Total slides',
                      'ROI method', 'Slides skipped', 'Stride'):
                pdf.cell(20, 4, m, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_y(y+1)
            pdf.set_font('Arial')
            if isinstance(meta.qc, list):
                qc = f"{len(meta.qc)} total"
            elif isinstance(meta.qc, str):
                qc = meta.qc
            else:
                qc = f"1 total"
            for m in (meta.tile_px, meta.tile_um, qc, meta.total_slides,
                      meta.roi_method, meta.slides_skipped, meta.stride):
                pdf.cell(30)
                pdf.cell(20, 4, str(m), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Second column
            pdf.set_y(y+1)
            pdf.set_font('Arial', style='B', size=10)
            for m in ('G.S. fraction', 'G.S. threshold', 'W.S. fraction',
                      'W.S. threshold', 'Normalizer', 'Format', 'Backend'):
                pdf.cell(45)
                pdf.cell(20, 4, m, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_y(y+1)
            pdf.set_font('Arial')
            for m in (meta.gs_frac, meta.gs_thresh, meta.ws_frac,
                      meta.ws_thresh, meta.normalizer, meta.img_format,
                      sf.slide_backend()):
                pdf.cell(75)
                pdf.cell(20, 4, str(m), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(20)

            # Save thumbnail first
            pdf.set_font('Arial', 'B', 7)
            n_images = 0
            log.debug("Rendering PDF pages with thumbnails.")
            for i, report in enumerate(reports):
                if report is None:
                    continue
                thumb = thumbnails[i]
                if thumb:
                    # Create a new row every 2 slides
                    if n_images % 2 == 0:
                        pdf.cell(50, 90, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    # Slide thumbnail
                    with tempfile.NamedTemporaryFile() as temp:
                        thumb.save(temp, format="JPEG", quality=75)
                        thumb_w, thumb_h = thumb.size
                        x = pdf.get_x()+((n_images+1) % 2 * 100)
                        y = pdf.get_y()-85
                        if (thumb_w / thumb_h) * 80 > 90:
                            pdf.image(temp.name, x, y, w=90)
                        else:
                            calc_w = 80 * (thumb_w / thumb_h)
                            offset = (80 - calc_w) / 2
                            pdf.image(temp.name, x+offset+5, y, h=80)
                        n_images += 1

                    # Slide label
                    y = pdf.get_y()
                    pdf.set_y(y-92)
                    if n_images % 2 == 1:
                        x = pdf.get_x()
                        pdf.cell(100, 5)

                    name = path_to_name(report.path)
                    if isinstance(report.num_tiles, int):
                        num_tiles = report.num_tiles
                    else:
                        num_tiles = 0
                    pdf.multi_cell(90, 3, f'{name}\n{num_tiles} tiles', 0, 'C')
                    if n_images % 2 == 1:
                        pdf.set_x(x)
                    pdf.set_y(y)

                if n_images % 2 == 0:
                    pdf.ln(1)

        # Now save rows of sample tiles
        for image_row, report in zip(image_rows, reports):
            if report is None:
                continue
            if image_row:
                pdf.set_font('Arial', '', 7)
                pdf.cell(10, 10, report.path, 0, 1)
                with tempfile.NamedTemporaryFile() as temp:
                    temp.write(image_row)
                    x = pdf.get_x()
                    y = pdf.get_y()
                    try:
                        pdf.image(
                            temp.name,
                            x,
                            y,
                            w=19*len(report.images),
                            h=19,
                            type='jpg'
                        )
                    except (RuntimeError, UnidentifiedImageError) as e:
                        log.error(f"Error writing image to PDF: {e}")
            pdf.ln(20)
        self.pdf = pdf

    def num_tiles_chart(self, num_tiles: np.ndarray) -> bool:
        import matplotlib.pyplot as plt
        import seaborn as sns
        if np.any(num_tiles):
            with sf.util.matplotlib_backend('Agg'):
                plt.rc('font', size=14)
                sns.histplot(num_tiles, bins=20)
                plt.title('Number of tiles extracted')
                plt.ylabel('Number of slides', fontsize=16, fontname='Arial')
                plt.xlabel('Tiles extracted', fontsize=16, fontname='Arial')
            return True
        else:
            return False

    def blur_chart(self, blur_arr: np.ndarray) -> bool:
        import matplotlib.pyplot as plt
        import seaborn as sns
        if np.any(blur_arr):
            num_warn = np.count_nonzero(blur_arr > self.bb_threshold)
            if num_warn:
                warn_txt = f'\nwarn = {num_warn}'
            else:
                warn_txt = ''
            with np.errstate(divide='ignore'):
                log_b = np.log(blur_arr)
            log_b = log_b[np.isfinite(log_b)]

            with sf.util.matplotlib_backend('Agg'):
                plt.rc('font', size=14)
                sns.histplot(log_b, bins=20)
                plt.title('Quality Control: Blur Burden'+warn_txt)
                plt.ylabel('Count', fontsize=16, fontname='Arial')
                plt.xlabel('log(blur burden)', fontsize=16, fontname='Arial')
                plt.axvline(x=-3, color='r', linestyle='--')
            return True
        else:
            return False

    def save(self, filename: str) -> None:
        self.pdf.output(filename)

    def update_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Update and save tile extraction report as CSV."""

        if len(self.reports):
            print("Updating CSV for {} reports.".format(len(self.reports)))
        else:
            print("Skipping CSV update; no extraction reports found.")
            return None
        if exists(filename):
            ex_df = pd.read_csv(filename)
            ex_df.set_index('slide')
        else:
            ex_df = None
        assert self.meta is not None
        if not self.meta.qc:
            qc_str = 'None'
        elif isinstance(self.meta.qc, str):
            qc_str = self.meta.qc
        elif isinstance(self.meta.qc, list):
            qc_str = ', '.join([str(s) for s in self.meta.qc])
        else:
            qc_str = str(self.meta.qc)
        df = pd.DataFrame({
            'slide':        pd.Series([path_to_name(r.path) for r in self.reports]),
            'num_tiles':    pd.Series([r.data['num_tiles'] for r in self.reports]),
            'tile_px':      pd.Series([self.meta.tile_px for r in self.reports]),
            'tile_um':      pd.Series([self.meta.tile_um for r in self.reports]),
            'rois':         pd.Series([r.data['num_rois'] for r in self.reports]),
            'stride':       pd.Series([self.meta.stride for r in self.reports]),
            'qc':           pd.Series([qc_str for r in self.reports]),
            'gs_fraction':  pd.Series([self.meta.gs_frac for r in self.reports]),
            'gs_threshold': pd.Series([self.meta.gs_thresh for r in self.reports]),
            'ws_fraction':  pd.Series([self.meta.ws_frac for r in self.reports]),
            'ws_threshold': pd.Series([self.meta.ws_thresh for r in self.reports]),
            'normalizer':   pd.Series([self.meta.normalizer for r in self.reports]),
            'img_format':   pd.Series([self.meta.img_format for r in self.reports]),
            'date':         pd.Series([r.timestamp for r in self.reports]),
            'backend':      pd.Series([sf.slide_backend() for r in self.reports]),
            'slideflow_version': pd.Series([sf.__version__ for r in self.reports])
        })
        df.set_index('slide')
        if ex_df is not None:
            df = pd.concat([df, ex_df[~ex_df.slide.isin(df.slide.unique())]])
        df.to_csv(filename, index=False)
        return df

# -----------------------------------------------------------------------------

class LowerMagExtractionReport(ExtractionReport):
    """ExtractionReport variant for lower magnification tile extraction."""

    def __init__(
        self,
        reports: List[LowerMagSlideReport],
        meta: Optional[Any] = None,
        title: str = 'Lower Magnification Tile Extraction Report',
        *,
        pool: Optional[Any] = None
    ) -> None:
        # Initialize parent (bb_threshold not really used here but keep signature)
        super().__init__(
            reports=reports,
            meta=meta,
            bb_threshold=0.05,
            title=title,
            pool=pool
        )
        # Add lower mag specific content to PDF
        self._add_lower_mag_summary()
        self._add_combination_metrics()

    def _add_lower_mag_summary(self) -> None:
        pdf = self.pdf
        # Aggregate stats safely
        total_slides = len([r for r in self.reports if r is not None])
        total_source_tiles = int(sum((r.data or {}).get('total_source_tiles', 0) for r in self.reports if r is not None))
        total_combined_tiles = int(sum((r.data or {}).get('num_tiles', 0) for r in self.reports if r is not None))
        total_discarded = int(sum((r.data or {}).get('discarded_tiles', 0) for r in self.reports if r is not None))

        effs = [r.combination_efficiency for r in self.reports if r is not None and hasattr(r, 'combination_efficiency') and r.combination_efficiency is not None]
        avg_efficiency = float(np.mean(effs)) if len(effs) else 0.0

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Lower Magnification Extraction Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        pdf.set_font('Arial', '', 10)
        summary_lines = [
            f"Total slides processed: {total_slides}",
            f"Source tiles processed: {total_source_tiles:,}",
            f"Combined tiles created: {total_combined_tiles:,}",
            f"Tiles discarded: {total_discarded:,}",
            f"Average combination efficiency: {avg_efficiency:.1f}%"
        ]
        for line in summary_lines:
            pdf.cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)

    def _add_combination_metrics(self) -> None:
        pdf = self.pdf
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Tile Combination Metrics', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        # Table header
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(60, 6, 'Slide', 1, 0, 'C')
        pdf.cell(25, 6, 'Source Tiles', 1, 0, 'C')
        pdf.cell(25, 6, 'Combined', 1, 0, 'C')
        pdf.cell(25, 6, 'Efficiency', 1, 0, 'C')
        pdf.cell(25, 6, 'Coverage Red.', 1, 0, 'C')
        pdf.cell(25, 6, 'Incomplete', 1, 1, 'C')

        # Rows
        pdf.set_font('Arial', '', 7)
        for report in self.reports:
            if report is None:
                continue
            slide_name = path_to_name(report.path)[:40]
            data = report.data or {}
            source_tiles = int(data.get('total_source_tiles', 0))
            combined = int(data.get('num_tiles', 0))
            efficiency = float(getattr(report, 'combination_efficiency', 0.0) or 0.0)
            coverage_red = float(getattr(report, 'spatial_coverage_reduction', 0.0) or 0.0)
            incomplete = float(getattr(report, 'incomplete_groups_percent', 0.0) or 0.0)

            pdf.cell(60, 5, slide_name, 1, 0, 'L')
            pdf.cell(25, 5, f"{source_tiles}", 1, 0, 'C')
            pdf.cell(25, 5, f"{combined}", 1, 0, 'C')
            pdf.cell(25, 5, f"{efficiency:.1f}%", 1, 0, 'C')
            pdf.cell(25, 5, f"{coverage_red:.1f}%", 1, 0, 'C')
            pdf.cell(25, 5, f"{incomplete:.1f}%", 1, 1, 'C')
        pdf.ln(10)