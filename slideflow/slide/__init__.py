# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''This module includes tools to convolutionally section whole slide images into tiles.
These tessellated tiles can be exported as PNG or JPG as raw images or stored in the binary
format TFRecords, with or without data augmentation.

Requires: libvips (https://libvips.github.io/libvips/).'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import io
import types
import time
import numpy as np
import csv
import pyvips as vips
import shapely.geometry as sg
import shapely.affinity as sa
import cv2
import json
import random
import tempfile
import warnings
import seaborn as sns
import slideflow as sf
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import multiprocessing as mp
import skimage
import skimage.filters

from os.path import join, exists
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from PIL import Image, ImageDraw, UnidentifiedImageError
from slideflow.util import log, SUPPORTED_FORMATS, UserError
from slideflow.slide.normalizers import StainNormalizer
from datetime import datetime
from functools import partial
from tqdm import tqdm
from fpdf import FPDF

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

def OPS_LEVEL_HEIGHT(l):
    return f'openslide.level[{l}].height'
def OPS_LEVEL_WIDTH(l):
    return f'openslide.level[{l}].width'
def OPS_LEVEL_DOWNSAMPLE(l):
    return f'openslide.level[{l}].downsample'

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

def _polyArea(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def _convert_img_to_format(image, img_format):
    if img_format.lower() == 'png':
        return cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
    elif img_format.lower() in ('jpg', 'jpeg'):
        return cv2.imencode('.jpg',
                              cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                              [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
    else:
        raise ValueError(f"Unknown image format {img_format}")

def _draw_roi(img, coords):
    # Draw ROIs
    annPolys = [sg.Polygon(b) for b in coords]
    if isinstance(img, np.ndarray):
        annotated_img = Image.fromarray(img)
    elif isinstance(img, str):
        annotated_img = Image.open(io.BytesIO(img))
    draw = ImageDraw.Draw(annotated_img)
    for poly in annPolys:
        x,y = poly.exterior.coords.xy
        zipped = list(zip(x.tolist(),y.tolist()))
        draw.line(zipped, joint='curve', fill='red', width=5)
    return np.asarray(annotated_img)

def _roi_coords_from_image(c, args):
    # Scale ROI according to downsample level
    extract_scale = (args.extract_px / args.full_extract_px)

    # Scale ROI according to image resizing
    resize_scale = (args.tile_px / args.extract_px)

    def proc_ann(ann):
        coord = ann.coordinates                                    # Scale to full image size
        coord = np.add(coord, np.array([-1*c[0], -1*c[1]]))        # Offset coordinates to extraction window
        coord = np.multiply(coord, (extract_scale * resize_scale)) # Rescale according to downsampling and resizing
        return coord

    # Filter out ROIs not in this tile
    coords = []
    ll = np.array([0, 0])
    ur = np.array([args.tile_px, args.tile_px])
    for roi in args.rois:
        coord = proc_ann(roi)
        coords_in_tile = coord[np.all(np.logical_and(ll <= coord, coord <= ur), axis=1)]
        if len(coords_in_tile) > 3:
            coords += [coords_in_tile]


    # Convert ROI to bounding box that fits within tile
    boxes = []
    yolo_anns = []
    for coord in coords:
        max_vals = np.max(coord, axis=0)
        min_vals = np.min(coord, axis=0)
        max_x, max_y = min(max_vals[0], args.tile_px), min(max_vals[1], args.tile_px)
        min_x, min_y = max(min_vals[0], 0), max(0, min_vals[1])

        width = (max_x - min_x) / args.tile_px
        height = (max_y - min_y) / args.tile_px
        x_center = ((max_x + min_x) / 2) / args.tile_px
        y_center = ((max_y + min_y) / 2) / args.tile_px
        yolo_anns += [[x_center, y_center, width, height]]

        boxes += [np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])]

    return coords, boxes, yolo_anns

def _wsi_extraction_worker(c, args):
    '''Multiprocessing worker for WSI. Extracts a tile at the given coordinates.'''

    index = c[2]
    grid_xi = c[4]
    grid_yi = c[5]

    # Check if the center of the current window lies within any annotation; if not, skip
    x_coord = int((c[0]+args.full_extract_px/2)/args.roi_scale)
    y_coord = int((c[1]+args.full_extract_px/2)/args.roi_scale)

    if args.roi_method != 'ignore' and bool(args.annPolys):
        point_in_roi = any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in args.annPolys])
        # If the extraction method is 'inside', skip the tile if it's not in an ROI
        if (args.roi_method == 'inside') and not point_in_roi:
            return 'skip'
        # If the extraction method is 'outside', skip the tile if it's in an ROI
        elif (args.roi_method == 'outside') and point_in_roi:
            return 'skip'

    # If downsampling is enabled, read image from highest level to perform filtering;
    # Otherwise filter from our target level
    slide = _VIPSWrapper(args.path, silent=True)
    if args.whitespace_fraction < 1 or args.grayspace_fraction < 1:
        if args.filter_downsample_ratio > 1:
            filter_extract_px = args.extract_px // args.filter_downsample_ratio
            filter_region = slide.read_region((c[0], c[1]), args.filter_downsample_level, [filter_extract_px, filter_extract_px])
        else:
            # Read the region and resize to target size
            filter_region = slide.read_region((c[0], c[1]), args.downsample_level, [args.extract_px, args.extract_px])

        # Perform whitespace filtering [Libvips]
        if args.whitespace_fraction < 1:
            fraction = filter_region.bandmean().relational_const('more', args.whitespace_threshold).avg() / 255
            if fraction > args.whitespace_fraction: return

        # Perform grayspace filtering [Libvips]
        if args.grayspace_fraction < 1:
            hsv_region = filter_region.sRGB2HSV()
            fraction = hsv_region[1].relational_const('less', args.grayspace_threshold*255).avg() / 255
            if fraction > args.grayspace_fraction: return

    # If dry run, return the current coordinates only
    if args.dry_run:
        return {'loc': [x_coord, y_coord]}, index

    # Normalizer
    normalizer = None if not args.normalizer else StainNormalizer(method=args.normalizer, source=args.normalizer_source)

    # Read the target downsample region now, if we were filtering at a different level
    region = slide.read_region((c[0], c[1]), args.downsample_level, [args.extract_px, args.extract_px])
    if region.bands == 4: region = region.flatten() # removes alpha
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
            try:    image = normalizer.jpeg_to_jpeg(image)
            except: return # The image could not be normalized, which happens when a tile is primarily one solid color
    else:
        image = vips2numpy(region)  # Read regions into memory and convert to numpy arrays

        # Apply normalization
        if normalizer:
            try:    image = normalizer.rgb_to_rgb(image)
            except: return # The image could not be normalized, which happens when a tile is primarily one solid color

    # Include ROI / bounding box processing.
    # Used to visualize ROIs on extracted tiles, or to generate YoloV5 labels.
    if args.yolo or args.draw_roi:
        coords, boxes, yolo_anns = _roi_coords_from_image(c, args)
    if args.draw_roi:
        image = _draw_roi(image, coords)

    return_dict = {'image': image}
    if args.yolo:
        return_dict.update({'yolo': yolo_anns})
    if args.include_loc == 'grid':
        return_dict.update({'loc': [grid_xi, grid_yi]})
    elif args.include_loc:
        return_dict.update({'loc': [x_coord, y_coord]})
    return return_dict, index

def vips2numpy(vi):
    '''Converts a VIPS image into a numpy array'''
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def log_extraction_params(**kwargs):
    '''Logs tile extraction parameters.'''

    ws_f = DEFAULT_WHITESPACE_FRACTION if 'whitespace_fraction' not in kwargs else kwargs['whitespace_fraction']
    ws_t = DEFAULT_WHITESPACE_THRESHOLD if 'whitespace_threshold' not in kwargs else kwargs['whitespace_threshold']
    gs_f = DEFAULT_GRAYSPACE_FRACTION if 'grayspace_fraction' not in kwargs else kwargs['grayspace_fraction']
    gs_t = DEFAULT_GRAYSPACE_THRESHOLD if 'grayspace_threshold' not in kwargs else kwargs['grayspace_threshold']

    if 'normalizer' in kwargs:
        log.info(f'Extracting tiles using {sf.util.bold(kwargs["normalizer"])} normalization')
    if ws_f < 1:
        log.info('Filtering tiles by whitespace fraction')
        log.debug(f'Whitespace defined as RGB avg > {ws_t} (exclude if >={ws_f*100:.0f}% whitespace')
    if gs_f < 1:
        log.info('Filtering tiles by grayspace fraction')
        log.debug(f'Grayspace defined as HSV avg < {gs_t} (exclude if >={gs_f*100:.0f}% grayspace)')

class TileCorruptionError(Exception):
    '''Raised when image normalization fails due to tile corruption.'''
    pass

class SlideReport:
    '''Report to summarize tile extraction from a slide, including example images of extracted tiles.'''

    def __init__(self, images, path, thumb=None, data=None, compress=True):
        """Initializer.

        Args:
            images (list(str)): List of JPEG image strings (example tiles).
            path (str): Path to slide.
            data (dict, optional): Dictionary of slide extraction report metadata. Defaults to None.
            compress (bool, optional): Compresses images to reduce image sizes. Defaults to True.
        """

        self.data = data
        self.path = path
        self.thumb = None if thumb is None else Image.fromarray(np.array(thumb)[:,:,0:3])

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
    def header(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        logo = join(package_directory, 'slideflow-logo-name-small.jpg')

        #if self.page_no() == 1:
        self.set_font('Arial', size=9)
        self.cell(70) # Moves right
        self.set_text_color(70,70,70)
        self.cell(50, 8, 'Intended for Research Use Only', align='C')
        self.ln(10)

        self.set_text_color(0,0,0)
        # Framed title
        #self.cell(30, 10, 'Title', 1, 0, 'C')
        self.set_font('Arial', 'B', 16)
        top = self.y
        self.cell(40, 10, 'Tile extraction report', 0, 1)
        self.y = top
        self.cell(150)
        self.image(logo, 160, 20, w=40)
        # Line break
        self.line(10, 30, 200, 30)
        self.ln(10)
        self.set_font('Arial', '', 10)
        top = self.y
        self.cell(20, 10, f'Generated: {datetime.now().strftime("%m/%d/%Y %H:%M:%S")}', 0, 1)
        self.y = top
        self.cell(150)
        self.cell(40, 10, sf.__version__, align='R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + ' of {nb}', 0, 0, 'C')

class ExtractionReport:
    """Creates a PDF report summarizing extracted tiles, from a collection of tile extraction reports."""

    def __init__(self, reports, meta=None, bb_threshold=0.05):
        """Initializer.

        Args:
            reports (list(:class:`SlideReport`)): List of SlideReport objects.
            tile_px (int): Tile size in pixels.
            tile_um (int): Tile size in microns.
        """

        self.bb_threshold = bb_threshold
        pdf = ExtractionPDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Update meta values for grayspace/whitespace filtering
        if meta.ws_frac is None:     meta.ws_frac = DEFAULT_WHITESPACE_FRACTION
        if meta.ws_thresh is None:   meta.ws_thresh = DEFAULT_WHITESPACE_THRESHOLD
        if meta.gs_frac is None:     meta.gs_frac   = DEFAULT_GRAYSPACE_FRACTION
        if meta.gs_thresh is None:   meta.gs_thresh  = DEFAULT_GRAYSPACE_THRESHOLD
        if meta.img_format is None:  meta.img_format  = 'png'

        num_tiles = np.array([r.num_tiles for r in reports if r is not None])
        bb = np.array([r.blur_burden for r in reports if r is not None])
        bb_names = [r.path for r in reports if r is not None]
        self.warn_txt = ''
        for slide, b in zip(bb_names, bb):
            if b is not None and b > self.bb_threshold:
                self.warn_txt += f'{slide},{b}\n'

        if np.any(num_tiles) and self.num_tiles_chart(num_tiles):
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
        pdf.set_draw_color(120,120,120)
        pdf.cell(95, 30, '', 1, 0, 'L')
        pdf.set_x(x)

        # First column
        pdf.set_y(y+1)
        pdf.set_font('Arial', style='B')
        for m in ('Tile size (px)', 'Tile size (um)', 'QC', 'Total slides', 'ROI method', 'Slides skipped', 'Stride'):
            pdf.cell(20,4,m, ln=1)
        pdf.set_y(y+1)
        pdf.set_font('Arial')
        for m in (meta.tile_px, meta.tile_um, meta.qc, meta.total_slides, meta.roi_method, meta.slides_skipped, meta.stride):
            pdf.cell(30)
            pdf.cell(20,4,str(m), ln=1)

        # Second column
        pdf.set_y(y+1)
        pdf.set_font('Arial', style='B', size=10)
        for m in ('G.S. fraction', 'G.S. threshold', 'W.S. fraction', 'W.S. threshold', 'Normalizer', 'Format'):
            pdf.cell(45)
            pdf.cell(20,4,m, ln=1)
        pdf.set_y(y+1)
        pdf.set_font('Arial')
        for m in (meta.gs_frac, meta.gs_thresh, meta.ws_frac, meta.ws_thresh, meta.normalizer, meta.img_format):
            pdf.cell(75)
            pdf.cell(20,4,str(m), ln=1)

        pdf.ln(20)

        # Save thumbnail first
        pdf.set_font('Arial', '', 7)
        n_images = 0
        for i, report in enumerate(reports):
            if report is None: continue
            if report.thumb:

                # Create a new row every 2 slides
                if n_images % 2 == 0:
                    pdf.cell(50, 90, ln=1)

                # Slide thumbnail
                with tempfile.NamedTemporaryFile() as temp:
                    report.thumb.save(temp, format="JPEG", quality=75)
                    x = pdf.get_x()+(n_images%2 * 100)
                    y = pdf.get_y()-85
                    pdf.image(temp.name, x, y, w=90, type='jpg')
                    n_images += 1

                # Slide label
                y = pdf.get_y()
                pdf.set_y(y-88)
                if n_images % 2 == 1:
                    x = pdf.get_x()
                    pdf.cell(100, 5)
                    #pdf.set_x(x+100)

                pdf.cell(90, 0, sf.util.path_to_name(report.path), 0, 1, 'C')
                if n_images % 2 == 1:
                    pdf.set_x(x)
                pdf.set_y(y)

            if n_images % 2 == 0:
                pdf.ln(1)

        # Now save rows of sample tiles
        for i, report in enumerate(reports):
            if report is None: continue
            image_row = report.image_row()
            if image_row:
                pdf.set_font('Arial', '', 7)
                pdf.cell(10, 10, report.path, 0, 1)
                with tempfile.NamedTemporaryFile() as temp:
                    temp.write(image_row)
                    x = pdf.get_x()
                    y = pdf.get_y()
                    try:
                        pdf.image(temp.name, x, y, w=19*len(report.images), h=19, type='jpg')
                    except RuntimeError as e:
                        log.error(f"Error writing image to PDF: {e}")
            pdf.ln(20)

        self.pdf = pdf

    def num_tiles_chart(self, num_tiles):
        if np.any(num_tiles):
            plt.rc('font', size=14)
            h = sns.histplot(num_tiles, bins=20)
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
            plt.rc('font', size=14)
            h = sns.histplot(log_b, bins=20)
            plt.title('Quality Control: Blur Burden'+warn_txt)
            plt.ylabel('Count', fontsize=16, fontname='Arial')
            plt.xlabel('log(blur burden)', fontsize=16, fontname='Arial')
            plt.axvline(x=-3, color='r', linestyle='--')
            return True
        else:
            return False

    def save(self, filename):
        self.pdf.output(filename)

class _VIPSWrapper:
    '''Wrapper for VIPS to preserve openslide-like functions.'''

    def __init__(self, path, buffer=None, silent=False):
        self.path = path
        self.buffer = buffer

        if buffer == 'vmtouch':
            os.system(f'vmtouch -q -t "{self.path}"')
        self.full_image = vips.Image.new_from_file(path, fail=True, access=vips.enums.Access.RANDOM)
        loaded_image = self.full_image

        # Load image properties
        self.properties = {}
        for field in loaded_image.get_fields():
            self.properties.update({field: loaded_image.get(field)})
        self.dimensions = (int(self.properties[OPS_WIDTH]), int(self.properties[OPS_HEIGHT]))

        # If Openslide MPP is not available, try reading from metadata
        if OPS_MPP_X not in self.properties.keys():
            if not silent:
                log.debug(f"Unable to detect openslide Microns-Per-Pixel (MPP) property, will search EXIF data")
            try:
                with Image.open(path) as img:
                    if TIF_EXIF_KEY_MPP in img.tag.keys():
                        if not silent:
                            log.debug(f"Setting MPP to {img.tag[TIF_EXIF_KEY_MPP][0]} per EXIF field {TIF_EXIF_KEY_MPP}")
                        self.properties[OPS_MPP_X] = img.tag[TIF_EXIF_KEY_MPP][0]
                    elif sf.util.path_to_ext(path).lower() in ('tif', 'tiff') and 'xres' in loaded_image.get_fields():
                        xres = loaded_image.get('xres') # 4000.0
                        if xres == 4000.0 and loaded_image.get('resolution-unit') == 'cm':
                            # xres = xres # though resolution from tiffinfo says 40000 pixels/cm, for some reason the xres val is 4000.0, so multiple by 10
                            mpp_x = (1/xres)*1000 # convert from pixels/cm to cm/pixels then convert to microns by multiplying by 1000
                            self.properties[OPS_MPP_X] = str(mpp_x)
                            if not silent:
                                log.debug(f"Setting MPP to {mpp_x} using TIFF 'xres' field {loaded_image.get('xres')} and {loaded_image.get('resolution-unit')}")
                    else:
                        log.warning(f"Unable to detect openslide Microns-Per-Pixel (MPP) property from slide {sf.util.path_to_name(path)}")
            except UnidentifiedImageError:
                log.error(f"PIL image reading error; unable to read slide {sf.util.path_to_name(path)}.")

        # Prepare downsample levels
        self.loaded_downsample_levels = {
            0: self.full_image,
        }
        if OPS_LEVEL_COUNT in self.properties:
            self.level_count = int(self.properties[OPS_LEVEL_COUNT])
            # Calculate level metadata
            self.levels = []
            for l in range(self.level_count):
                width = int(loaded_image.get(OPS_LEVEL_WIDTH(l)))
                height = int(loaded_image.get(OPS_LEVEL_HEIGHT(l)))
                downsample = float(loaded_image.get(OPS_LEVEL_DOWNSAMPLE(l)))
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

        self.level_downsamples = [l['downsample'] for l in self.levels]
        self.level_dimensions = [l['dimensions'] for l in self.levels]

    def get_best_level_for_downsample(self, downsample):
        '''Return best level to match a given desired downsample.'''
        max_downsample = 0
        for d in self.level_downsamples:
            if d < downsample:
                max_downsample = d
        try:
            max_level = self.level_downsamples.index(max_downsample)
        except:
            return 0
        return max_level

    def get_downsampled_image(self, level):
        '''Returns a VIPS image of a given downsample.'''
        if level in range(len(self.levels)):
            if level in self.loaded_downsample_levels:
                return self.loaded_downsample_levels[level]
            else:
                downsampled_image = vips.Image.new_from_file(self.path,
                                                             level=level,
                                                             fail=True,
                                                             access=vips.enums.Access.RANDOM)
                self.loaded_downsample_levels.update({
                    level: downsampled_image
                })
                return downsampled_image
        else:
            return False

    def read_region(self, base_level_dim, downsample_level, extract_size):
        '''Extracts a region from the image at the given downsample level.'''
        base_level_x, base_level_y = base_level_dim
        extract_width, extract_height = extract_size
        downsample_factor = self.level_downsamples[downsample_level]
        downsample_x = int(base_level_x / downsample_factor)
        downsample_y = int(base_level_y / downsample_factor)

        image = self.get_downsampled_image(downsample_level)
        region = image.crop(downsample_x, downsample_y, extract_width, extract_height)
        return region

    def unbuffer(self):
        if self.buffer == 'vmtouch':
            os.system(f'vmtouch -e "{self.path}"')

class _JPGslideToVIPS(_VIPSWrapper):
    '''Wrapper for JPG files, which do not possess separate levels, to preserve openslide-like functions.'''

    def __init__(self, path, buffer=None):
        self.buffer = buffer
        self.path = path
        self.full_image = vips.Image.new_from_file(path)
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
        with Image.open(path) as img:
            if TIF_EXIF_KEY_MPP in img.tag.keys():
                log.info(f"Setting MPP to {img.tag[TIF_EXIF_KEY_MPP][0]} per EXIF field {TIF_EXIF_KEY_MPP}")
                self.properties[OPS_MPP_X] = img.tag[TIF_EXIF_KEY_MPP][0]
            else:
                log.info(f"Setting MPP to default {DEFAULT_JPG_MPP}")
                self.properties[OPS_MPP_X] = DEFAULT_JPG_MPP

class ROI:
    '''Object container for ROI annotations.'''

    def __init__(self, name):
        self.name = name
        self.coordinates = []
    def __repr__(self):
        return f"<ROI (coords={len(self.coordinates)})>"
    def add_coord(self, coord):
        self.coordinates.append(coord)
    def scaled_area(self, scale):
        return np.multiply(self.coordinates, 1/scale)
    def print_coord(self):
        for c in self.coordinates: print(c)
    def add_shape(self, shape):
        for point in shape:
            self.add_coord(point)

class _BaseLoader:
    '''Object that loads an SVS slide and makes preparations for tile extraction.
    Should not be used directly; this class must be inherited and extended by either WSI or TMA child classes.'''

    def __init__(self, path, tile_px, tile_um, stride_div, enable_downsample=True,
                    buffer=None, pb=None, pb_counter=None, counter_lock=None):
        self.load_error = False

        # if a progress bar is not directly provided, use the provided multiprocess-friendly progress bar counter and lock
        #     (for multiprocessing, as ProgressBar cannot be pickled)
        if not pb:
            self.pb_counter = pb_counter
            self.counter_lock = counter_lock
        # Otherwise, use the provided progress bar's counter and lock
        else:
            self.pb_counter = pb.get_counter()
            self.counter_lock = pb.get_lock()

        self.name = sf.util.path_to_name(path)
        self.shortname = sf.util._shortname(self.name)
        self.tile_px = tile_px
        self.tile_um = tile_um
        self.tile_mask = None
        self.enable_downsample = enable_downsample
        self.thumb_image = None
        self.stride_div = stride_div
        self.path = path
        self.qc_mask = None
        self.qc_mpp = None
        self.qc_method = None
        self.blur_burden = None
        filetype = sf.util.path_to_ext(path)

        # Initiate supported slide reader
        if not os.path.exists(path):
            raise OSError(f"Could not find slide {path}; file does not exist.")
        if filetype.lower() in sf.util.SUPPORTED_FORMATS:
            if filetype.lower() == 'jpg':
                self.slide = _JPGslideToVIPS(path)
            else:
                self.slide = _VIPSWrapper(path, buffer=buffer)
        else:
            log.error(f"Unsupported file type '{filetype}' for slide {self.name}.")
            self.load_error = True
            return

        # Collect basic slide information
        try:
            self.mpp = float(self.slide.properties[OPS_MPP_X])
        except KeyError:
            log.error(f"Slide {sf.util.green(self.name)} missing MPP property ({OPS_MPP_X})")
            self.load_error = True
            return
        self.full_shape = self.slide.dimensions
        self.full_extract_px = int(self.tile_um / self.mpp)

        # Load downsampled level based on desired extraction size
        downsample_desired = self.full_extract_px / tile_px
        if enable_downsample:
            self.downsample_level = self.slide.get_best_level_for_downsample(downsample_desired)
        else:
            self.downsample_level = 0
        self.downsample_factor = self.slide.level_downsamples[self.downsample_level]
        self.shape = self.slide.level_dimensions[self.downsample_level]

        # Calculate pixel size of extraction window using downsampling
        self.extract_px = self.full_extract_px // self.downsample_factor
        self.full_stride = self.full_extract_px // stride_div
        self.stride = self.extract_px // stride_div

        # Calculate filter dimensions (low magnification for filtering out white background and performing edge detection)
        self.filter_dimensions = self.slide.level_dimensions[-1]
        self.filter_magnification = self.filter_dimensions[0] / self.full_shape[0]
        self.filter_px = int(self.full_extract_px * self.filter_magnification)

    @property
    def dimensions(self):
        return self.slide.dimensions

    @property
    def properties(self):
        return self.slide.properties

    @property
    def vendor(self):
        if OPS_VENDOR in self.slide.properties:
            return self.slide.properties[OPS_VENDOR]
        else:
            return None

    def mpp_to_dim(self, mpp):
        width = int((self.mpp * self.full_shape[0]) / mpp)
        height = int((self.mpp * self.full_shape[1]) / mpp)
        return (width, height)

    def dim_to_mpp(self, dimensions):
        return (self.full_shape[0] * self.mpp) / dimensions[0]

    def remove_qc(self):
        self._build_coord()
        self.qc_method = None
        log.debug(f'QC removed from slide {self.shortname}')

    def qc(self, method, blur_radius=3, blur_threshold=0.02, filter_threshold=0.6, blur_mpp=4):
        """Applies quality control to a slide, performing filtering based on a whole-slide image thumbnail.

        'blur' method filters out blurry or out-of-focus slide sections.
        'otsu' method filters out background based on automatic saturation thresholding in the HSV colorspace.
        'both' applies both methods of filtering.

        Args:
            method (str): Quality control method, includes 'blur', 'otsu', or 'both'.
            blur_radius (int, optional): Blur radius.
            blur_threshold (float, optional): Blur threshold.
            filter_threshold (float, optional): Percent of a tile detected as background that will trigger a tile to be
                discarded. Defaults to 0.6.
            blur_mpp (float, optional): Size of WSI thumbnail on which to perform blur QC, in microns-per-pixel.
                Defaults to 4 (equivalent magnification = 2.5 X)
        """

        if method not in ('blur', 'otsu', 'both'):
            raise ValueError(f"Unknown method {method}, valid methods include 'blur', 'otsu', 'both'")
        starttime = time.time()

        self.qc_mpp = blur_mpp
        self.qc_method = method
        target_mask_mpp = 40

        # Blur QC must be performed at a set microns-per-pixel rather than downsample level, as blur detection is
        # much for sensitive to effective magnification than  Otsu's thresholding
        if method in ('blur', 'both'):
            thumb = np.array(self.thumb(mpp=blur_mpp))
            if thumb.shape[-1] == 4:
                thumb = thumb[:,:,:3]
            gray = rgb2gray(thumb)
            img_laplace = np.abs(skimage.filters.laplace(gray))
            blur_mask = skimage.filters.gaussian(img_laplace, sigma=blur_radius) <= blur_threshold
            qc_ratio = 1 / self.slide.level_downsamples[self.slide.level_count-1]
            qc_ratio = self.mpp / blur_mpp
            self.qc_mask = blur_mask

        # Otsu's thresholding can be done on the lowest downsample level
        if method in ('otsu', 'both'):
            otsu_thumb = vips.Image.new_from_file(self.path, fail=True, access=vips.enums.Access.RANDOM, level=self.slide.level_count-1)
            otsu_thumb = vips2numpy(otsu_thumb)
            if otsu_thumb.shape[-1] == 4:
                otsu_thumb = otsu_thumb[:,:,:3]
            hsv_img = cv2.cvtColor(otsu_thumb, cv2.COLOR_RGB2HSV)
            img_med = cv2.medianBlur(hsv_img[:,:,1], 7)
            _, otsu_mask = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
            otsu_mask = otsu_mask.astype(np.bool)
            qc_ratio = 1 / self.slide.level_downsamples[self.slide.level_count-1]
            self.qc_mask = otsu_mask

        # If performing both, ensure the mask sizes are equivalent (shrinks to the size of the smaller mask - Otsu)
        if method == 'both':
            blur_mask = skimage.transform.resize(blur_mask, otsu_mask.shape).astype(np.bool)
            self.qc_mask = np.logical_or(blur_mask, otsu_mask)
            self.blur_burden = np.count_nonzero(np.logical_and(blur_mask, np.logical_xor(blur_mask, otsu_mask))) / (blur_mask.shape[0] * blur_mask.shape[1])
            log.debug(f"Blur burden: {self.blur_burden}")

        # Filter coordinates
        qc_width = int(self.full_extract_px * qc_ratio)
        to_delete = []
        for i, c in enumerate(self.coord):
            qc_x = int(c[0] * qc_ratio)
            qc_y = int(c[1] * qc_ratio)
            submask = self.qc_mask[qc_y:(qc_y+qc_width), qc_x:(qc_x+qc_width)]
            if np.mean(submask) > filter_threshold:
                to_delete += [i]
        self.coord = np.delete(np.array(self.coord), to_delete, axis=0)
        self.tile_mask = np.delete(self.tile_mask, to_delete)
        self.estimated_num_tiles = self.coord.shape[0]
        img = Image.fromarray(img_as_ubyte(self.qc_mask))
        log.debug(f'QC complete for slide {self.shortname} (time: {time.time()-starttime:.2f}s)')
        return img

    def square_thumb(self, width=512):
        '''Returns a square thumbnail of the slide, with black bar borders.

        Args:
            width (int): Width/height of thumbnail in pixels.

        Returns:
            PIL image
        '''
        # Get thumbnail image and dimensions via fastest method available
        if 'slide-associated-images' in self.slide.properties and 'thumbnail' in self.slide.properties['slide-associated-images']:
            vips_thumb = vips.Image.openslideload(self.slide.path, associated='thumbnail')
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

    def thumb(self, mpp=None, width=None, coords=None):
        '''Returns PIL thumbnail of the slide.

        Args:
            mpp (float, optional): Microns-per-pixel, used to determine thumbnail size.
            width (int, optional): Alternatively, goal thumbnail width may be supplied.
            coords (list(int), optional): List of tile extraction coordinates to show as rectangles
                on the thumbnail, in [(x_center, y_center), ...] format. Defaults to None.

        Returns:
            PIL image
        '''

        assert (mpp is None or width is None), "Either mpp must be supplied or width, but not both"
        # If no values provided, create thumbnail of width 1024
        if mpp is None and width is None:
            width = 1024

        # Calculate goal width/height according to specified microns-per-pixel (MPP)
        if mpp:
            width = int((self.mpp * self.full_shape[0]) / mpp)
        # Otherwise, calculate approximate mpp based on provided width (to generate proportional height)
        else:
            mpp = (self.mpp * self.full_shape[0]) / width
        # Calculate appropriate height
        height = int((self.mpp * self.full_shape[1]) / mpp)

        # Get thumb via libvips & convert PIL Image
        if self.vendor and self.vendor == 'leica':
            # The libvips thumbnail function does not work appropriately with Leica SCN images,
            # so a downsample level must be manually specified.
            thumbnail = vips.Image.new_from_file(self.path, fail=True, access=vips.enums.Access.RANDOM, level=self.slide.level_count-1)
        else:
            thumbnail = vips.Image.thumbnail(self.path, width)
        np_thumb = vips2numpy(thumbnail)
        image = Image.fromarray(np_thumb).resize((width, height))

        if coords:
            draw = ImageDraw.Draw(image)
            ratio = width / self.dimensions[0]
            wh = (self.full_extract_px * ratio) / 2
            for (x, y) in coords:
                x, y = x * ratio * self.roi_scale, y * ratio * self.roi_scale
                coords = (x-wh, y-wh, x+wh, y+wh)
                draw.rectangle(coords, outline='black', width=2)
            return image
        else:
            return image

    def build_generator(self, **kwargs):
        lead_msg = f'Extracting {self.tile_um}um tiles'
        resize_msg = f'(resizing {self.extract_px}px -> {self.tile_px}px)'
        stride_msg = f'stride: {int(self.stride)}px'
        log.debug(f"{self.shortname}: {lead_msg} {resize_msg}; {stride_msg}")
        if self.tile_px > self.extract_px:
            upscale_msg = 'Tiles will be up-scaled with bilinear interpolation'
            upscale_amount = f'({self.extract_px}px -> {self.tile_px}px)'
            log.warn(f"{self.shortname}: [{sf.util.red('!WARN!')}] {upscale_msg} {upscale_amount}")

        def empty_generator():
            yield None

        return empty_generator

    def loaded_correctly(self):
        '''Returns True if slide loaded correctly without errors; False if otherwise.'''
        if self.load_error:
            return False
        try:
            loaded_correctly = bool(self.shape)
        except:
            return False
        return loaded_correctly

    def extract_tiles(self, tfrecord_dir=None, tiles_dir=None, img_format='jpg', report=True, **kwargs):
        """Extracts tiles from slide using the build_generator() method,
        saving tiles into a TFRecord file or as loose JPG tiles in a directory.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images into a subdirectory (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal storage in tfrecords.
                PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
                Defaults to 'jpg'.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            full_core (bool, optional): Extract an entire detected core, rather than subdividing into image tiles.
                Defaults to False.
            shuffle (bool): Shuffle images during extraction.
            num_threads (int): Number of threads to allocate to tile extraction workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI annotations (.txt) in the tile directory.
                Requires that tiles_dir is set. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles. Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted, but do not export any images.
                Defaults to None.
        """

        if img_format not in ('png', 'jpg', 'jpeg'):
            raise ValueError(f"Unknown image format {img_format}, must be either 'png' or 'jpg'")

        # Make base directories
        if tfrecord_dir:
            if not exists(tfrecord_dir): os.makedirs(tfrecord_dir)
        if tiles_dir:
            tiles_dir = os.path.join(tiles_dir, self.name)
            if not os.path.exists(tiles_dir): os.makedirs(tiles_dir)

        # Log to keep track of when tiles have finished extracting
        # To be used in case tile extraction is interrupted, so the slide can be flagged for re-extraction
        if tfrecord_dir or tiles_dir:
            unfinished_marker = join((tfrecord_dir if tfrecord_dir else tiles_dir), f'{self.name}.unfinished')
            with open(unfinished_marker, 'w') as marker_file:
                marker_file.write(' ')
        if tfrecord_dir:
            writer = sf.io.TFRecordWriter(join(tfrecord_dir, self.name+".tfrecords"))

        generator = self.build_generator(show_progress=(self.counter_lock is None), img_format=img_format, **kwargs)
        slidename_bytes = bytes(self.name, 'utf-8')

        if not generator:
            log.error(f"No tiles extracted from slide {sf.util.green(self.name)}")
            return

        sample_tiles = []
        generator_iterator = generator()
        locations = []
        num_wrote_to_tfr = 0
        dry_run = kwargs['dry_run'] if 'dry_run' in kwargs else False

        for index, tile_dict in enumerate(generator_iterator):
            location = tile_dict['loc']
            locations += [location]

            if dry_run: continue

            image_string = tile_dict['image']
            if len(sample_tiles) < 10:
                sample_tiles += [image_string]
            elif (not tiles_dir and not tfrecord_dir) and not dry_run:
                break
            if tiles_dir:
                with open(join(tiles_dir, f'{self.shortname}_{index}.{img_format}'), 'wb') as outfile:
                    outfile.write(image_string)
                if 'yolo' in tile_dict and len(tile_dict['yolo']):
                    with open(join(tiles_dir, f'{self.shortname}_{index}.txt'), 'w') as outfile:
                        for ann in tile_dict['yolo']:
                            outfile.write("0 {:.3f} {:.3f} {:.3f} {:.3f}\n".format(ann[0], ann[1], ann[2], ann[3]))
            if tfrecord_dir:
                record = sf.io.serialized_record(slidename_bytes, image_string, location[0], location[1])
                writer.write(record)
                num_wrote_to_tfr += 1
        if tfrecord_dir:
            writer.close()
            if not num_wrote_to_tfr:
                os.remove(join(tfrecord_dir, self.name+".tfrecords"))
                log.info(f'No tiles extracted for slide {sf.util.green(self.name)}')
        if self.counter_lock is None:
            generator_iterator.close()

        if tfrecord_dir or tiles_dir:
            try:
                os.remove(unfinished_marker)
            except:
                log.error(f"Unable to mark slide {self.name} as tile extraction complete")

        # Unbuffer slide
        self.slide.unbuffer()

        # Generate extraction report
        if report:
            report_data = {
                'blur_burden': self.blur_burden,
                'num_tiles': len(locations),
            }
            slide_report = SlideReport(sample_tiles,
                                       self.slide.path,
                                       data=report_data,
                                       thumb=self.thumb(coords=locations, rois=(self.roi_method != 'ignore')))
            return slide_report

    def preview(self, rois=True, **kwargs):
        """Performs a dry run of tile extraction without saving any images, returning a PIL image of the slide
        thumbnail annotated with a grid of tiles that were marked for extraction.

        Args:
            rois (bool, optional): Draw ROI annotation(s) onto the image. Defaults to True.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            full_core (bool, optional): Extract an entire detected core, rather than subdividing into image tiles.
                Defaults to False.
            num_threads (int): Number of threads to allocate to tile extraction workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI annotations (.txt) in the tile directory.
                Requires that tiles_dir is set. Defaults to False.
        """

        generator = self.build_generator(show_progress=(self.counter_lock is None), dry_run=True, **kwargs)
        locations = []

        for index, tile_dict in enumerate(generator()):
            locations += [tile_dict['loc']]

        return self.thumb(coords=locations, rois=rois)

class WSI(_BaseLoader):
    '''Loads a slide and its annotated region of interest (ROI).'''

    def __init__(self, path, tile_px, tile_um, stride_div=1, enable_downsample=True, roi_dir=None, rois=None,
                 roi_method='inside', skip_missing_roi=False, randomize_origin=False, buffer=None, pb=None,
                 pb_counter=None, counter_lock=None, silent=False):

        """Loads slide and ROI(s).

        Args:
            path (str): Path to slide.
            tile_px (int): Size of tiles to extract, in pixels.
            tile_um (int): Size of tiles to extract, in microns.
            stride_div (int, optional): Stride divisor for tile extraction (1 = no tile overlap; 2 = 50% overlap, etc).
                Defaults to 1.
            enable_downsample (bool, optional): Allow use of downsampled intermediate layers in the slide image pyramid,
                which greatly improves tile extraction speed. May result in artifacts for slides with incompletely
                generated intermediates pyramid layers. Defaults to True.
            roi_dir (str, optional): Directory in which to search for ROI CSV files. Defaults to None.
            rois (list(str)): Alternatively, a list of ROI paths can be explicitly provided. Defaults to None.
            roi_method (str): Either 'inside', 'outside', or 'ignore'. Determines how ROIs are used to extract tiles.
                Defaults to 'inside'.
            skip_missing_roi (bool, optional): Skip tiles that are missing a ROI file. Defaults to False.
            randomize_origin (bool, optional): Offset the starting grid by a random amount. Defaults to False.
            buffer (str): Path to directory. Slides will be copied to the directory as a buffer before extraction.
                Vastly improves extraction speed if an SSD or ramdisk buffer is used. Defaults to None
            pb (:class:`slideflow.util.ProgressBar`, optional): Multiprocessing-capable ProgressBar instance; will
                update progress bar during tile extraction if provided. Used for multiprocessing tile extraction.
            pb_counter (obj): Multiprocessing counter (a multiprocessing Value, from Progress Bar) used to follow
                tile extraction progress. Defaults to None.
            counter_lock (obj): Lock object for updating pb_counter, if provided. Defaults to None.
            silent (bool, optional): Suppresses warnings about slide skipping if ROIs are missing. Defaults to False.
        """

        super().__init__(path, tile_px, tile_um, stride_div, enable_downsample, buffer, pb, pb_counter, counter_lock)

        # Initialize calculated variables
        self.extracted_x_size = 0
        self.extracted_y_size = 0
        self.estimated_num_tiles = 0
        self.annPolys = []
        self.roi_scale = 10
        self.rois = []
        self.roi_method = roi_method
        self.randomize_origin = randomize_origin

        if not self.loaded_correctly():
            return

        # Build coordinate grid
        self._build_coord()

        # Look in ROI directory if available
        if roi_dir and exists(join(roi_dir, self.name + ".csv")):
            self.load_csv_roi(join(roi_dir, self.name + ".csv"))

        # Else try loading ROI from same folder as slide
        elif exists(self.name + ".csv"):
            self.load_csv_roi(sf.util.path_to_name(path) + ".csv")
        elif rois and self.name in [sf.util.path_to_name(r) for r in rois]:
            matching_rois = []
            for rp in rois:
                rn = sf.util.path_to_name(rp)
                if rn == self.name:
                    matching_rois += [rp]
            if len(matching_rois) > 1:
                log.warning(f" Multiple matching ROIs found for {self.name}; using {matching_rois[0]}")
            self.load_csv_roi(matching_rois[0])

        # Handle missing ROIs
        if not len(self.rois) and roi_method != 'ignore' and not (rois or roi_dir):
            # No ROIs found because the user did not provide rois or roi_dir, but the roi_method is not set to 'ignore',
            # indicating that this may be user error.
            warn_msg = f"No ROIs provided for {self.name} (suppress this warning with roi_method='ignore')"
            if not silent and not (rois is None and roi_dir is None):
                log.warning(warn_msg)
            else:
                log.debug(warn_msg)
        if not len(self.rois) and skip_missing_roi and roi_method != 'ignore':
            warn_msg = f"No ROI found for {sf.util.green(self.name)}, skipping slide"
            if not silent:  log.warning(warn_msg)
            else:           log.debug(warn_msg)
            self.shape = None
            self.load_error = True
            return None
        elif not len(self.rois):
            info_msg = f"No ROI found for {sf.util.green(self.name)}, using whole slide."
            if not silent and roi_method != 'ignore':
                log.info(info_msg)
            else:
                log.debug(info_msg)
            self.roi_method = 'ignore'

        mpp_roi_msg = f'{self.mpp} um/px | {len(self.rois)} ROI(s)'
        size_msg = f'Size: {self.full_shape[0]} x {self.full_shape[1]}'
        log.debug(f"{self.shortname}: Slide info: {mpp_roi_msg} | {size_msg}")
        log.debug(f"{self.shortname}: Grid shape: {self.grid.shape} | Tiles to extract: {self.estimated_num_tiles}")

        # Abort if errors were raised during ROI loading
        if self.load_error:
            log.error(f'Skipping slide {sf.util.green(self.name)} due to loading error')
            return None

    def __repr__(self):
        base = "WSI(\n"
        base += "  path = {!r},\n".format(self.path)
        base += "  tile_px = {!r},\n".format(self.tile_px)
        base += "  tile_um = {!r},\n".format(self.tile_um)
        base += "  stride_div = {!r},\n".format(self.stride_div)
        base += "  enable_downsample = {!r},\n".format(self.enable_downsample)
        base += "  roi_method = {!r},\n".format(self.roi_method)
        base += ")"
        return base

    def _build_coord(self):
        '''Set up coordinate grid.'''

        # Calculate window sizes, strides, and coordinates for windows
        self.extracted_x_size = self.full_shape[0] - self.full_extract_px
        self.extracted_y_size = self.full_shape[1] - self.full_extract_px

        # Randomize origin, if desired
        if self.randomize_origin:
            start_x = random.randint(0, self.full_stride-1)
            start_y = random.randint(0, self.full_stride-1)
            log.info(f"Random origin: X: {start_x}, Y: {start_y}")
        else:
            start_x = start_y = 0

        # Coordinates must be in level 0 (full) format for the read_region function
        index = 0
        self.coord = []
        y_range = np.arange(start_y, (self.full_shape[1]+1) - self.full_extract_px, self.full_stride)
        x_range = np.arange(start_x, (self.full_shape[0]+1) - self.full_extract_px, self.full_stride)
        for yi, y in enumerate(y_range):
            for xi, x in enumerate(x_range):
                y = int(y)
                x = int(x)
                is_unique = ((y % self.full_extract_px == 0) and (x % self.full_extract_px == 0))
                self.coord.append([x, y, index, is_unique, xi, yi])
                index += 1
        self.coord = np.array(self.coord)
        self.estimated_num_tiles = self.coord.shape[0]
        self.tile_mask = np.asarray([False for i in range(len(self.coord))], dtype=np.bool)
        self.grid = np.zeros((len(x_range), len(y_range)))

    def extract_tiles(self, tfrecord_dir=None, tiles_dir=None, img_format='jpg', report=True, **kwargs):
        """Extracts tiles from slide using the build_generator() method,
        saving tiles into a TFRecord file or as loose JPG tiles in a directory.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images into a subdirectory (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal storage in tfrecords.
                PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
                Defaults to 'jpg'.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            full_core (bool, optional): Extract an entire detected core, rather than subdividing into image tiles.
                Defaults to False.
            shuffle (bool): Shuffle images during extraction.
            num_threads (int): Number of threads to allocate to tile extraction workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI annotations (.txt) in the tile directory.
                Requires that tiles_dir is set. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles. Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted, but do not export any images.
                Defaults to None.
        """
        return super().extract_tiles(tfrecord_dir, tiles_dir, img_format, report, **kwargs)

    def build_generator(self, shuffle=True, whitespace_fraction=None, whitespace_threshold=None,
                        grayspace_fraction=None, grayspace_threshold=None, normalizer=None, normalizer_source=None,
                        include_loc=True, num_threads=8, show_progress=False, img_format='numpy', full_core=None,
                        yolo=False, draw_roi=False, pool=None, dry_run=False):

        """Builds tile generator to extract tiles from this slide.

        Args:
            shuffle (bool): Shuffle images during extraction.
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            include_loc (bool, optional): Return (x,y) origin coordinates for each tile along with tile images.
            show_progress (bool, optional): Show a progress bar for tile extraction.
            img_format (str, optional): Image format. Either 'numpy', 'jpg', or 'png'. Defaults to 'numpy'.
            yolo (bool, optional): Include yolo-formatted tile-level ROI annotations in the return dictionary,
                under the key 'yolo'. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles. Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted, but do not export any images.
                Defaults to None.

        Returns:
            dict, with keys 'image' (image data), 'yolo' (optional yolo-formatted annotations, (x_center, y_center,
            width, height)) and 'grid' ((x, y) slide or grid coordinates)

        """

        super().build_generator()

        if self.estimated_num_tiles == 0:
            log.warning(f"No tiles extracted at the given micron size for slide {sf.util.green(self.name)}")
            return None

        # Shuffle coordinates to randomize extraction order
        if shuffle:
            np.random.shuffle(self.coord)

        # Set whitespace / grayspace fraction to global defaults if not provided
        if whitespace_fraction is None:     whitespace_fraction  = DEFAULT_WHITESPACE_FRACTION
        if whitespace_threshold is None:    whitespace_threshold = DEFAULT_WHITESPACE_THRESHOLD
        if grayspace_fraction is None:      grayspace_fraction   = DEFAULT_GRAYSPACE_FRACTION
        if grayspace_threshold is None:     grayspace_threshold  = DEFAULT_GRAYSPACE_THRESHOLD

        # Get information about highest level downsample, as we will filter on that layer if downsampling is enabled
        if self.enable_downsample:
            filter_downsample_level = len(self.slide.level_downsamples) - 1
            filter_downsample_factor = self.slide.level_downsamples[filter_downsample_level]
            filter_downsample_ratio = filter_downsample_factor // self.slide.level_downsamples[self.downsample_level]
        else:
            filter_downsample_level = self.downsample_level
            filter_downsample_ratio = 1

        worker_args = {
            'full_extract_px': self.full_extract_px,
            'roi_scale': self.roi_scale,
            'roi_method': self.roi_method,
            'rois': self.rois,
            'annPolys': self.annPolys,
            'estimated_num_tiles': self.estimated_num_tiles,
            'downsample_level': self.downsample_level,
            'filter_downsample_level': filter_downsample_level,
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
            'include_loc': include_loc,
            'img_format': img_format,
            'yolo': yolo,
            'draw_roi': draw_roi,
            'dry_run': dry_run
        }
        worker_args = types.SimpleNamespace(**worker_args)

        def generator():
            nonlocal pool
            if pool is None:
                log.debug("Building tile extraction generator with a shared pool")
                pool = mp.Pool(processes=num_threads)
                should_close = True
            else:
                log.debug(f"Building tile extraction generator with {num_threads} thread workers")
                should_close = False
            if show_progress:
                pbar = tqdm(total=self.estimated_num_tiles, ncols=80)
            for idx, res in enumerate(pool.imap(partial(_wsi_extraction_worker, args=worker_args), self.coord)):
                if res == 'skip':
                    continue

                # Increase progress bars, if provided
                if show_progress:
                    pbar.update(1)
                elif self.counter_lock is not None:
                    with self.counter_lock:
                        self.pb_counter.value += 1

                if res is None:
                    continue
                else:
                    tile, _ = res
                    self.tile_mask[idx] = True
                    yield tile
            if show_progress:
                pbar.close()
            if should_close:
                pool.close()

            name_msg = sf.util.green(self.shortname)
            num_msg = f'({np.sum(self.tile_mask)} tiles of {len(self.coord)} possible)'
            log.debug(f"Finished tile extraction for {name_msg} {num_msg}")
            if not dry_run:
                print(f"\r\033[KFinished tile extraction for {name_msg} {num_msg}")

        return generator

    def thumb(self, mpp=None, width=None, coords=None, rois=False, linewidth=2, color='black'):
        """Returns PIL Image of thumbnail with ROI overlay.

        Args:
            mpp (float, optional): Microns-per-pixel, used to determine thumbnail size.
            width (int, optional): Alternatively, goal thumbnail width may be supplied.
            coords (list(int), optional): List of tile extraction coordinates to show as rectangles
                on the thumbnail, in [(x_center, y_center), ...] format. Defaults to None.
            rois (bool, optional): Draw ROIs onto thumbnail. Defaults to False.
            linewidth (int, optional): Width of ROI overlay line. Defaults to 2.
            color (str, optional): Color of ROI overlay. Defaults to black.

        Returns:
            PIL image
        """

        if rois:
            assert (mpp is None or width is None), "Either mpp must be supplied or width, but not both"
            # If no values provided, create thumbnail of width 1024
            if mpp is None and width is None:
                width = 1024
            if mpp is not None:
                roi_scale = self.full_shape[0] / (int((self.mpp * self.full_shape[0]) / mpp))
            else:
                roi_scale = self.full_shape[0] / width

        thumb = super().thumb(mpp=mpp, width=width, coords=coords)

        if rois:
            annPolys = [sg.Polygon(annotation.scaled_area(roi_scale)) for annotation in self.rois]
            draw = ImageDraw.Draw(thumb)
            for poly in annPolys:
                x,y = poly.exterior.coords.xy
                zipped = list(zip(x.tolist(),y.tolist()))
                draw.line(zipped, joint='curve', fill=color, width=linewidth)
            return thumb
        else:
            return thumb

    def load_csv_roi(self, path):
        '''Loads CSV ROI from a given path.'''

        roi_dict = {}
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            try:
                headers = next(reader, None)
                headers = [h.lower() for h in headers]
                index_name = headers.index("roi_name")
                index_x = headers.index("x_base")
                index_y = headers.index("y_base")
            except:
                log.error(f'Unable to read CSV ROI file {sf.util.green(path)}, please check file integrity and ' + \
                                'ensure headers contain "ROI_name", "X_base", and "Y_base".')
                self.load_error = True
                return
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
                    self.annPolys += [sg.Polygon(annotation.scaled_area(self.roi_scale))]
                except ValueError:
                    log.warning(f"Unable to use ROI {i} in slide {sf.util.green(self.name)}, at least 3 points required " + \
                                "to create a geometric shape.")
            roi_area = sum([poly.area for poly in self.annPolys])
        else:
            roi_area = 1
        total_area = (self.full_shape[0]/self.roi_scale) * (self.full_shape[1]/self.roi_scale)
        self.roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

        if self.roi_method == 'inside':
            self.estimated_num_tiles = int(self.coord.shape[0] * self.roi_area_fraction)
        else:
            self.estimated_num_tiles = int(self.coord.shape[0] * (1-self.roi_area_fraction))

        return len(self.rois)

    def load_json_roi(self, path, scale=10):
        '''Loads ROI from a JSON file.'''

        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], scale)
            self.rois.append(ROI(f"Object{len(self.rois)}"))
            self.rois[-1].add_shape(area_reduced)
        return len(self.rois)

class TMA(_BaseLoader):
    '''Loads a TMA-formatted slide and detects tissue cores.'''

    QUEUE_SIZE = 8
    HEIGHT_MIN = 20
    WIDTH_MIN = 20
    BLACK = (0,0,0)
    BLUE = (255, 100, 100)
    GREEN = (75,220,75)
    LIGHTBLUE = (255, 180, 180)
    RED = (100, 100, 200)
    WHITE = (255,255,255)

    def __init__(self, path, tile_px, tile_um, stride_div=1, annotations_dir=None,
                    enable_downsample=True, report_dir=None, buffer=None, pb=None, pb_id=0):
        '''Initializer.

        Args:
            path:               Path to slide
            tile_px:            Size of tiles to extract, in pixels
            tile_um:            Size of tiles to extract, in microns
            stride_div:         Stride divisor for tile extraction (1 = no tile overlap; 2 = 50% overlap, etc)
            enable_downsample:  Bool, if True, allows use of downsampled intermediate layers in the slide image pyramid,
                                    which greatly improves tile extraction speed.
            buffer:             Path to directory. Slides will be copied here prior to extraction.
            pb:                 ProgressBar instance; will update progress bar during tile extraction if provided
            pb_id:              ID of bar in ProgressBar, defaults to 0
        '''
        super().__init__(path, tile_px, tile_um, stride_div, enable_downsample, buffer, pb)

        if not self.loaded_correctly():
            return

        self.object_rects = []
        self.box_areas = []
        self.DIM = self.slide.dimensions
        target_thumb_width = self.DIM[0] / 100
        target_thumb_mpp = self.dim_to_mpp((target_thumb_width, -1))
        self.thumb_image = np.array(self.thumb(mpp=target_thumb_mpp))[:,:,:-1]
        self.THUMB_DOWNSCALE = self.DIM[0] / self.mpp_to_dim(target_thumb_mpp)[0]
        self.pb = pb
        self.pb_id = pb_id
        num_cores, self.estimated_num_tiles = self._detect_cores(report_dir=report_dir)
        size_msg = f'Size: {self.full_shape[0]} x {self.full_shape[1]}'
        log.info(f"{self.shortname}: Slide info: {self.mpp} um/px | {size_msg}")

    def _get_sub_image(self, rect):
        '''Gets a sub-image from the slide using the specified rectangle as a guide.'''
        box = cv2.boxPoints(rect) * self.THUMB_DOWNSCALE
        box = np.int0(box)

        rect_width = int(rect[1][0] * self.THUMB_DOWNSCALE / self.downsample_factor)
        rect_height = int(rect[1][1] * self.THUMB_DOWNSCALE / self.downsample_factor)

        region_x_min = int(min([b[0] for b in box]))
        region_x_max = max([b[0] for b in box])
        region_y_min = int(min([b[1] for b in box]))
        region_y_max = max([b[1] for b in box])

        region_width  = int((region_x_max - region_x_min) / self.downsample_factor)
        region_height = int((region_y_max - region_y_min) / self.downsample_factor)

        extracted = vips2numpy(self.slide.read_region((region_x_min, region_y_min),
                                                      self.downsample_level,
                                                      (region_width, region_height)))[:,:,:-1]
        relative_box = (box - [region_x_min, region_y_min]) / self.downsample_factor

        src_pts = relative_box.astype("float32")
        dst_pts = np.array([[0, (rect_height)-1],
                            [0, 0],
                            [(rect_width)-1, 0],
                            [(rect_width)-1, (rect_height)-1]], dtype="float32")

        P = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped=cv2.warpPerspective(extracted, P, (rect_width, rect_height))
        return warped

    def _resize_to_target(self, image_tile):
        '''Resizes image tile to the desired target output size.'''
        target_MPP = self.tile_um / self.tile_px
        current_MPP = self.mpp * self.downsample_factor
        resize_factor = current_MPP / target_MPP
        return cv2.resize(image_tile, (0, 0), fx=resize_factor, fy=resize_factor)

    def _split_core(self, image):
        '''Splits core into desired sub-images.'''
        height, width, channels = image.shape
        num_y = int(height / self.tile_px)
        num_x = int(width  / self.tile_px)

        # If the desired micron tile size is too large, expand and center the source image
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
            num_x = int(width  / self.tile_px)

        y_start = int((height - (num_y * self.tile_px))/2)
        x_start = int((width  - (num_x * self.tile_px))/2)

        subtiles = []

        for y in range(num_y):
            for x in range(num_x):
                sub_x_start = x_start + (x * self.tile_px)
                sub_y_start = y_start + (y * self.tile_px)
                subtiles += [image[sub_y_start:sub_y_start+self.tile_px, sub_x_start:sub_x_start+self.tile_px]]

        return subtiles

    def _detect_cores(self, report_dir=None):
        # Prepare annotated image
        img_annotated = self.thumb_image.copy()

        # Create background mask for edge detection
        white = np.array([255,255,255])
        buffer = 28
        mask = cv2.inRange(self.thumb_image, np.array([0,0,0]), white-buffer)

        # Fill holes and dilate mask
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        dilating_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        closing = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, closing_kernel)
        dilated = cv2.dilate(closing, dilating_kernel)

        # Use edge detection to find individual cores
        contours, heirarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        # Filter out small regions that likely represent background noise
        # Also generate image showing identified cores
        num_filtered = 0
        for i, component in enumerate(zip(contours, heirarchy[0])):
            cnt = component[0]
            heir = component[1]
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]
            if width > self.WIDTH_MIN and height > self.HEIGHT_MIN and heir[3] < 0:
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
                #cv2.putText(img_annotated, f'{num_filtered}', (cX+10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.BLACK, 2)
            elif heir[3] < 0:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_annotated, [box], 0, self.RED, 2)

        log.info(f"Number of detected cores: {num_filtered}")

        # Write annotated image to ExtractionReport
        if report_dir:
            cv2.imwrite(join(report_dir, "tma_extraction_report.jpg"), cv2.resize(img_annotated, (1400, 1000)))

        return num_filtered, num_filtered

    def extract_tiles(self, tfrecord_dir=None, tiles_dir=None, img_format='jpg', report=True, **kwargs):
        """Extracts tiles from slide using the build_generator() method,
        saving tiles into a TFRecord file or as loose JPG tiles in a directory.

        Args:
            tfrecord_dir (str): If provided, saves tiles into a TFRecord file (named according to slide name) here.
            tiles_dir (str): If provided, saves loose images into a subdirectory (per slide name) here.
            img_format (str): 'png' or 'jpg'. Format of images for internal storage in tfrecords.
                PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
                Defaults to 'jpg'.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            full_core (bool, optional): Extract an entire detected core, rather than subdividing into image tiles.
                Defaults to False.
            shuffle (bool): Shuffle images during extraction.
            num_threads (int): Number of threads to allocate to tile extraction workers.
            yolo (bool, optional): Export yolo-formatted tile-level ROI annotations (.txt) in the tile directory.
                Requires that tiles_dir is set. Defaults to False.
            draw_roi (bool, optional): Draws ROIs onto extracted tiles. Defaults to False.
            dry_run (bool, optional): Determine tiles that would be extracted, but do not export any images.
                Defaults to None.
        """
        return super().extract_tiles(tfrecord_dir, tiles_dir, img_format, report, **kwargs)

    def build_generator(self, shuffle=True, whitespace_fraction=None, whitespace_threshold=None, grayspace_fraction=None,
                        grayspace_threshold=None, normalizer=None, normalizer_source=None, include_loc=True,
                        num_threads=8, pool=None, img_format='numpy', full_core=False):

        """Builds tile generator to extract of tiles across the slide.

        Args:
            shuffle (bool): Shuffle images during extraction.
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            include_loc (bool, optional): Include location information in returned dictionary. Defaults to True.
            num_threads (int, optional): Number of threads for pool. Unused if `pool` is specified.
            pool (:obj:`multiprocessing.Pool`, optional): Multiprocessing pool to use. By using a shared pool, a slide
                no longer needs to spin up its own new pool for tile extraction, decreasing tile extraction time for
                large datasets. Defaults to None (create a new pool, using `num_threads`).
            img_format (str, optional): 'png', 'jpg', or 'numpy'. Format images should be returned in.
            full_core (bool, optional): Extract an entire detected core, rather than subdividing into image tiles.
                Defaults to False.
        """

        super().build_generator()

        if include_loc:
            log.warning("Tile location logging for TMA slides is not yet complete; recording all locations as (0, 0).")

        # Setup normalization
        normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        # Shuffle TMAs
        if shuffle:
            random.shuffle(self.object_rects)

        # Set whitespace / grayspace fraction to global defaults if not provided
        if whitespace_fraction is None:     whitespace_fraction  = DEFAULT_WHITESPACE_FRACTION
        if whitespace_threshold is None:    whitespace_threshold = DEFAULT_WHITESPACE_THRESHOLD
        if grayspace_fraction is None:      grayspace_fraction   = DEFAULT_GRAYSPACE_FRACTION
        if grayspace_threshold is None:     grayspace_threshold  = DEFAULT_GRAYSPACE_THRESHOLD

        # Establish extraction queues
        rectangle_queue = mp.Queue()
        extraction_queue = mp.Queue(self.QUEUE_SIZE)

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
            unique_tile = True
            extraction_pool = mp.Pool(num_threads, section_extraction_worker,(rectangle_queue, extraction_queue,))

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
                        self.pb.increase_bar_value(id=self.pb_id)

                    resized_core = self._resize_to_target(image_core)

                    if full_core:
                        resized = cv2.resize(image_core, (self.tile_px, self.tile_px))

                        # Convert to final image format
                        if img_format != 'numpy':
                            resized = _convert_img_to_format(resized, img_format)

                        if include_loc:
                            yield {'image': resized, 'loc': [0, 0]}
                        else:
                            yield {'image': resized}
                    else:
                        subtiles = self._split_core(resized_core)
                        for subtile in subtiles:
                            # Perform whitespace filtering
                            if whitespace_fraction < 1:
                                fraction = (np.mean(subtile, axis=2) > whitespace_threshold).sum() / (self.tile_px**2)
                                if fraction > whitespace_fraction: continue

                            # Perform grayspace filtering
                            if grayspace_fraction < 1:
                                hsv_image = mcol.rgb_to_hsv(subtile)
                                fraction = (hsv_image[:,:,1] < grayspace_threshold).sum() / (self.tile_px**2)
                                if fraction > grayspace_fraction: continue

                            # Apply normalization
                            if normalizer:
                                try:
                                    subtile = normalizer.rgb_to_rgb(subtile)
                                except:
                                    # The image could not be normalized, which happens when
                                    # a tile is primarily one solid color (background)
                                    continue

                            # Convert to final image format
                            if img_format != 'numpy':
                                subtile = _convert_img_to_format(subtile, img_format)

                            if include_loc:
                                yield {'image': subtile, 'loc': [0, 0]}
                            else:
                                yield {'image': subtile}

            extraction_pool.close()

        return generator
