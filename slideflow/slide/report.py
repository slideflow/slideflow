'''Functions for slide extraction reports (PDF).'''

from __future__ import absolute_import, division, print_function

import io
import os
import tempfile
import pandas as pd
import numpy as np
import cv2

from fpdf import FPDF, XPos, YPos
from PIL import Image, UnidentifiedImageError
from datetime import datetime
from os.path import join, exists
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

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
            pil_img = Image.open(io.BytesIO(img))
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

            if np.any(n_tiles) and self.num_tiles_chart(n_tiles):
                with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                    plt.savefig(temp.name)
                    pdf.image(temp.name, 107, pdf.y, w=50)
                    plt.clf()

            if np.any(bb) and self.blur_chart(bb):
                with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                    plt.savefig(temp.name)
                    pdf.image(temp.name, 155, pdf.y, w=50)

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