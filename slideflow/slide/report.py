'''Functions for slide extraction reports (PDF).'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import numpy as np
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from PIL import Image
from datetime import datetime
from fpdf import FPDF

import slideflow as sf
from slideflow.util import log, path_to_name  # noqa F401


class SlideReport:
    '''Report to summarize tile extraction from a slide, including
    example images of extracted tiles.'''

    def __init__(self, images, path, thumb=None, data=None, compress=True):
        """Initializer.

        Args:
            images (list(str)): List of JPEG image strings (example tiles).
            path (str): Path to slide.
            data (dict, optional): Dictionary of slide extraction report
                metadata. Defaults to None.
            compress (bool, optional): Compresses images to reduce image sizes.
                Defaults to True.
        """

        self.data = data
        self.path = path
        if thumb is None:
            self.thumb = None
        else:
            self.thumb = Image.fromarray(np.array(thumb)[:, :, 0:3])
        if not compress:
            self.images = images
        else:
            self.images = [self._compress(img) for img in images]

    @property
    def blur_burden(self):
        if 'blur_burden' in self.data:
            return self.data['blur_burden']
        else:
            return None

    @property
    def num_tiles(self):
        if 'num_tiles' in self.data:
            return self.data['num_tiles']
        else:
            return None

    def _compress(self, img):
        with io.BytesIO() as output:
            Image.open(io.BytesIO(img)).save(output, format="JPEG", quality=75)
            return output.getvalue()

    def image_row(self):
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
    def __init__(self, *args, title='Tile extraction report', **kwargs):
        super().__init__(*args, **kwargs)
        self.title_msg = title

    def header(self):
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
        top = self.y
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

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + ' of {nb}', 0, 0, 'C')


class ExtractionReport:
    """Creates a PDF report summarizing extracted tiles, from a collection of
    tile extraction reports."""

    def __init__(self, reports, meta=None, bb_threshold=0.05,
                 title='Tile extraction report'):
        """Initializer.

        Args:
            reports (list(:class:`SlideReport`)): List of SlideReport objects.
        """

        self.bb_threshold = bb_threshold
        pdf = ExtractionPDF(title=title)
        pdf.alias_nb_pages()
        pdf.add_page()

        if hasattr(meta, 'ws_frac'):
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
                pdf.cell(20, 4, m, ln=1)
            pdf.set_y(y+1)
            pdf.set_font('Arial')
            for m in (meta.tile_px, meta.tile_um, meta.qc, meta.total_slides,
                      meta.roi_method, meta.slides_skipped, meta.stride):
                pdf.cell(30)
                pdf.cell(20, 4, str(m), ln=1)

            # Second column
            pdf.set_y(y+1)
            pdf.set_font('Arial', style='B', size=10)
            for m in ('G.S. fraction', 'G.S. threshold', 'W.S. fraction',
                      'W.S. threshold', 'Normalizer', 'Format'):
                pdf.cell(45)
                pdf.cell(20, 4, m, ln=1)
            pdf.set_y(y+1)
            pdf.set_font('Arial')
            for m in (meta.gs_frac, meta.gs_thresh, meta.ws_frac,
                      meta.ws_thresh, meta.normalizer, meta.img_format):
                pdf.cell(75)
                pdf.cell(20, 4, str(m), ln=1)

            pdf.ln(20)

            # Save thumbnail first
            pdf.set_font('Arial', 'B', 7)
            n_images = 0
            for i, report in enumerate(reports):
                if report is None:
                    continue
                if report.thumb:

                    # Create a new row every 2 slides
                    if n_images % 2 == 0:
                        pdf.cell(50, 90, ln=1)

                    # Slide thumbnail
                    with tempfile.NamedTemporaryFile() as temp:
                        report.thumb.save(temp, format="JPEG", quality=75)
                        thumb_w, thumb_h = report.thumb.size
                        x = pdf.get_x()+((n_images+1) % 2 * 100)
                        y = pdf.get_y()-85
                        if (thumb_w / thumb_h) * 80 > 90:
                            pdf.image(temp.name, x, y, w=90, type='jpg')
                        else:
                            calc_w = 80 * (thumb_w / thumb_h)
                            offset = (80 - calc_w) / 2
                            pdf.image(temp.name, x+offset+5, y, h=80, type='jpg')
                        n_images += 1

                    # Slide label
                    y = pdf.get_y()
                    pdf.set_y(y-92)
                    if n_images % 2 == 1:
                        x = pdf.get_x()
                        pdf.cell(100, 5)

                    name = path_to_name(report.path)
                    if isinstance(report.num_tiles, int):
                        n_tiles = report.num_tiles
                    else:
                        n_tiles = 0
                    label = name + f'\n{n_tiles} tiles'
                    pdf.multi_cell(90, 3, label, 0, 'C')
                    if n_images % 2 == 1:
                        pdf.set_x(x)
                    pdf.set_y(y)

                if n_images % 2 == 0:
                    pdf.ln(1)

        # Now save rows of sample tiles
        for report in reports:
            if report is None:
                continue
            image_row = report.image_row()
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
                    except RuntimeError as e:
                        log.error(f"Error writing image to PDF: {e}")
            pdf.ln(20)

        self.pdf = pdf

    def num_tiles_chart(self, num_tiles):
        if np.any(num_tiles):
            plt.rc('font', size=14)
            sns.histplot(num_tiles, bins=20)
            plt.title('Number of tiles extracted')
            plt.ylabel('Number of slides', fontsize=16, fontname='Arial')
            plt.xlabel('Tiles extracted', fontsize=16, fontname='Arial')
            return True
        else:
            return False

    def blur_chart(self, blur_arr):
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

    def save(self, filename):
        self.pdf.output(filename)