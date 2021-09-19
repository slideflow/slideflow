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
import io
import types
import tensorflow as tf
import numpy as np
import csv
import pyvips as vips
import shapely.geometry as sg
import cv2
import json
import random
import tempfile
import warnings

import matplotlib.colors as mcol
import slideflow.util as sfutil
import multiprocessing as mp

from os.path import join, exists
from PIL import Image, ImageDraw, UnidentifiedImageError
from slideflow.util import log, StainNormalizer, SUPPORTED_FORMATS
from slideflow.io.tfrecords import tfrecord_example
from datetime import datetime
from functools import partial
from tqdm import tqdm
from fpdf import FPDF

#TODO: optionally randomize center of individual tile extraction

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 100000000000
DEFAULT_JPG_MPP = 1
OPS_LEVEL_COUNT = 'openslide.level-count'
OPS_MPP_X = 'openslide.mpp-x'
TIF_EXIF_KEY_MPP = 65326
OPS_WIDTH = 'width'
OPS_HEIGHT = 'height'
EXTRACT_INSIDE = 'inside'
EXTRACT_OUTSIDE = 'outside'
IGNORE_ROI = 'ignore'

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

def polyArea(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def vips2numpy(vi):
    '''Converts a VIPS image into a numpy array'''
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def slide_extraction_worker(c, args):
    '''Multiprocessing working for WSI. Extracts a tile at the given coordinates'''
    slide = VIPSWrapper(args.path)
    normalizer = None if not args.normalizer else StainNormalizer(method=args.normalizer, source=args.normalizer_source)

    index = c[2]
    grid_xi = c[4]
    grid_yi = c[5]

    # Check if the center of the current window lies within any annotation; if not, skip
    x_coord = int((c[0]+args.full_extract_px/2)/args.ROI_SCALE)
    y_coord = int((c[1]+args.full_extract_px/2)/args.ROI_SCALE)

    if args.roi_method != IGNORE_ROI and bool(args.annPolys):
        point_in_roi = any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in args.annPolys])
        # If the extraction method is EXTRACT_INSIDE, skip the tile if it's not in an ROI
        if (args.roi_method == EXTRACT_INSIDE) and not point_in_roi:
            return 'skip'
        # If the extraction method is EXTRACT_OUTSIDE, skip the tile if it's in an ROI
        elif (args.roi_method == EXTRACT_OUTSIDE) and point_in_roi:
            return 'skip'

    # Read the region and resize to target size
    region = slide.read_region((c[0], c[1]), args.downsample_level, [args.extract_px, args.extract_px])
    region = region.thumbnail_image(args.tile_px)

    # Read regions into memory and convert to numpy arrays
    np_image = vips2numpy(region)[:,:,:-1]

    if args.dual_extract:
        try:
            surrounding_region = slide.read_region((c[0]-args.full_stride,
                                                            c[1]-args.full_stride),
                                                            args.downsample_level,
                                                            [args.extract_px*3, args.extract_px*3])
            surrounding_region = surrounding_region.thumbnail_image(args.tile_px)
            outer_region = vips2numpy(surrounding_region)[:,:,:-1]
        except:
            return

        # Apply normalization
        if normalizer:
            np_image = normalizer.rgb_to_rgb(np_image)
            outer_region = normalizer.rgb_to_rgb(outer_region)

        return {"input_1": np_image, "input_2": outer_region}, index
    else:
        # Perform whitespace filtering
        if args.whitespace_fraction < 1:
            fraction = (np.mean(np_image, axis=2) > args.whitespace_threshold).sum() / (args.tile_px**2)
            if fraction > args.whitespace_fraction: return

        # Perform grayspace filtering
        if args.grayspace_fraction < 1:
            hsv_image = mcol.rgb_to_hsv(np_image)
            fraction = (hsv_image[:,:,1] < args.grayspace_threshold).sum() / (args.tile_px**2)
            if fraction > args.grayspace_fraction: return

        # Apply normalization
        if normalizer:
            try:
                np_image = normalizer.rgb_to_rgb(np_image)
            except:
                # The image could not be normalized, which happens when a tile is primarily one solid color (background)
                return

        if args.include_loc == 'grid':
            return {'image': np_image, 'loc': [grid_xi, grid_yi]}, index
        elif args.include_loc:
            return {'image': np_image, 'loc': [x_coord, y_coord]}, index
        else:
            return {'image': np_image}, index

class InvalidTileSplitException(Exception):
    '''Raised when invalid tile splitting parameters are given to WSI.'''
    pass

class TileCorruptionError(Exception):
    '''Raised when image normalization fails due to tile corruption.'''
    pass

class UserError(Exception):
    pass

class SlideReport:
    '''Report to summarize tile extraction from a slide,
    including example images of extracted tiles.'''

    def __init__(self, images, path, data=None, compress=True):
        '''Initializer.

        Args:
            images:		List of JPEG image strings (example tiles)
            path:		Path to slide
            data:		Dictionary of slide extraction report metadata
            compress:	Bool, if True, compresses images to reduce image sizes
        '''
        self.data = data
        self.path = path
        if not compress:
            self.images = images
        else:
            self.images = []
            for image in images:
                with io.BytesIO() as output:
                    Image.open(io.BytesIO(image)).save(output, format="JPEG", quality=75)
                    self.images += [output.getvalue()]

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
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

class ExtractionReport:
    '''Creates a PDF report summarizing extracted tiles,
        from a collection of tile extraction reports.'''
    def __init__(self, reports, tile_px=None, tile_um=None):
        '''Initializer.

        Args:
            reports:	List of SlideReport objects
            tile_px:	Tile size in pixels.
            tile_um:	Tile size in microns.
        '''
        pdf = ExtractionPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(40, 10, 'Tile extraction report', 0, 1)
        pdf.set_font('Arial', '', 12)
        if tile_px and tile_um:
            pdf.cell(20, 10, f'Tile size: {tile_px}px, {tile_um}um', 0, 1)
        pdf.cell(20, 10, f'Generated: {datetime.now()}', 0, 1)

        for i, report in enumerate(reports):
            pdf.set_font('Arial', '', 7)
            pdf.cell(10, 10, report.path, 0, 1)
            image_row = report.image_row()
            if image_row:
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

    def save(self, filename):
        self.pdf.output(filename)

class VIPSWrapper:
    '''Wrapper for VIPS to preserve openslide-like functions.'''
    def __init__(self, path, buffer=None):
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
            log.warning(f"Unable to detect openslide Microns-Per-Pixel (MPP) property, will search EXIF data")
            try:
                with Image.open(path) as img:
                    if TIF_EXIF_KEY_MPP in img.tag.keys():
                        log.info(f"Setting MPP to {img.tag[TIF_EXIF_KEY_MPP][0]} per EXIF field {TIF_EXIF_KEY_MPP}")
                        self.properties[OPS_MPP_X] = img.tag[TIF_EXIF_KEY_MPP][0]
            except UnidentifiedImageError:
                log.error(f"PIL image reading error; slide {sfutil.path_to_name(path)} is corrupt.")

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

class JPGslideToVIPS(VIPSWrapper):
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

class ROIObject:
    '''Object container for ROI annotations.'''
    def __init__(self, name):
        self.name = name
        self.coordinates = []
    def add_coord(self, coord):
        self.coordinates.append(coord)
    def scaled_area(self, scale):
        return np.multiply(self.coordinates, 1/scale)
    def print_coord(self):
        for c in self.coordinates: print(c)
    def add_shape(self, shape):
        for point in shape:
            self.add_coord(point)

class BaseLoader:
    '''Object that loads an SVS slide and makes preparations for tile extraction.
    Should not be used directly; this class must be inherited and extended by a child class'''
    def __init__(self, path, tile_px, tile_um, stride_div, enable_downsample=False,
                    silent=False, buffer=None, pb=None, pb_counter=None, counter_lock=None, print_fn=None):
        self.load_error = False
        self.silent = silent

        # if a progress bar is not directly provided, use the provided multiprocess-friendly progress bar counter and lock
        # 	(for multiprocessing, as ProgressBar cannot be pickled)
        if not pb:
            self.pb_counter = pb_counter
            self.counter_lock = counter_lock
        # Otherwise, use the provided progress bar's counter and lock
        else:
            self.pb_counter = pb.get_counter()
            self.counter_lock = pb.get_lock()
            print_fn = pb.print if not print_fn else print_fn

        self.print = None if silent else (print if not print_fn else print_fn)
        self.error_print = print if not print_fn else print_fn

        self.name = sfutil.path_to_name(path)
        self.shortname = sfutil._shortname(self.name)
        self.tile_px = tile_px
        self.tile_um = tile_um
        self.tile_mask = None
        self.enable_downsample = enable_downsample
        self.thumb_image = None
        self.stride_div = stride_div
        self.path = path
        filetype = sfutil.path_to_ext(path)

        # Initiate supported slide reader
        if filetype.lower() in sfutil.SUPPORTED_FORMATS:
            if filetype.lower() == 'jpg':
                self.slide = JPGslideToVIPS(path)
            else:
                self.slide = VIPSWrapper(path, buffer=buffer)
        else:
            log.error(f"Unsupported file type '{filetype}' for slide {self.name}.")
            self.load_error = True
            return

        # Collect basic slide information
        try:
            self.MPP = float(self.slide.properties[OPS_MPP_X])
        except KeyError:
            log.error(f"Slide {sfutil.green(self.name)} missing MPP property ({OPS_MPP_X})")
            self.load_error = True
            return
        self.full_shape = self.slide.dimensions
        self.full_extract_px = int(self.tile_um / self.MPP)

        # Load downsampled level based on desired extraction size
        downsample_desired = self.full_extract_px / tile_px
        if enable_downsample:
            self.downsample_level = self.slide.get_best_level_for_downsample(downsample_desired)
        else:
            self.downsample_level = 0
        self.downsample_factor = self.slide.level_downsamples[self.downsample_level]
        self.shape = self.slide.level_dimensions[self.downsample_level]

        # Calculate pixel size of extraction window using downsampling
        self.extract_px = int(self.full_extract_px / self.downsample_factor)
        self.full_stride = int(self.full_extract_px / stride_div)
        self.stride = int(self.extract_px / stride_div)

        # Calculate filter dimensions (low magnification for filtering out white background and performing edge detection)
        self.filter_dimensions = self.slide.level_dimensions[-1]
        self.filter_magnification = self.filter_dimensions[0] / self.full_shape[0]
        self.filter_px = int(self.full_extract_px * self.filter_magnification)

    def mpp_to_dim(self, mpp):
        width = int((self.MPP * self.full_shape[0]) / mpp)
        height = int((self.MPP * self.full_shape[1]) / mpp)
        return (width, height)

    def dim_to_mpp(self, dimensions):
        return (self.full_shape[0] * self.MPP) / dimensions[0]

    def square_thumb(self, width=512):
        '''Returns a square thumbnail of the slide, with black bar borders.

        Args:
            width:		Width/height of thumbnail in pixels.

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

    def thumb(self, mpp=None, width=None):
        '''Returns PIL thumbnail of the slide.

        Args:
            mpp:	Microns-per-pixel of thumbnail (determines size of thumbnail to return)

        Returns:
            PIL image
        '''

        assert (mpp is None or width is None), "Either mpp must be supplied or width, but not both"
        # If no values provided, create thumbnail of width 1024
        if mpp is None and width is None:
            width = 1024

        # Calculate goal width/height according to specified microns-per-pixel (MPP)
        if mpp:
            width = int((self.MPP * self.full_shape[0]) / mpp)
        # Otherwise, calculate approximate mpp based on provided width (to generate proportional height)
        else:
            mpp = (self.MPP * self.full_shape[0]) / width
        # Calculate appropriate height
        height = int((self.MPP * self.full_shape[1]) / mpp)
        # Only regenerate thumbnail if not regenerated previously
        if not self.thumb_image:
            # Get thumbnail image and dimensions via fastest method available
            thumbnail = vips.Image.thumbnail(self.path, width)

            # Convert thumb image to PIL
            np_thumb = vips2numpy(thumbnail)
            self.thumb_image = Image.fromarray(np_thumb).resize((width, height))
        return self.thumb_image

    def build_generator(self, **kwargs):
        lead_msg = f'Extracting {sfutil.bold(self.tile_um)}um tiles'
        resize_msg = f'(resizing {sfutil.bold(self.extract_px)}px -> {sfutil.bold(self.tile_px)}px)'
        stride_msg = f'stride: {sfutil.bold(int(self.stride))}px'
        log.info(f"{self.shortname}: {lead_msg} {resize_msg}; {stride_msg}")
        if self.tile_px > self.extract_px:
            upscale_msg = 'Tiles will be up-scaled with bilinear interpolation'
            upscale_amount = f'({self.extract_px}px -> {self.tile_px}px)'
            log.info(f"{self.shortname}: [{sfutil.red('!WARN!')}] {upscale_msg} {upscale_amount}")

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

    def extract_tiles(self, tfrecord_dir=None, tiles_dir=None, split_fraction=None, split_names=None,
                      img_format='png', shuffle=True, num_threads=4, **kwargs):
        '''Extractes tiles from slide and saves into a TFRecord file or as loose JPG tiles in a directory.
        Args:
            tfrecord_dir:			If provided, saves tiles into a TFRecord file (named according to slide name) here.
            tiles_dir:				If provided, saves loose images into a subdirectory (per slide name) here.
            split_fraction:			List of float. If provided, splits the extracted tiles into subsets
                                        (e.g. for validation set) using these fractions.
                                        Should add up to 1 (except for fractions of -1).
                                        Remaining tiles are split between fractions of "-1".
            split_names:			List of names to label the split fractions
            img_format:				'png' or 'jpg'. Format of images for internal storage in tfrecords.
                                        PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
            shuffle:                Bool. If true, will shuffle tiles prior to storage. (default = True)
            num_threads:            Number of threads to allocate to tile extraction workers.

        Kwargs:
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
            whitespace_fraction:	Float 0-1. Fraction of whitespace which causes a tile to be discarded.
                                        If 1, will not perform whitespace filtering.
            whitespace_threshold:	Int 0-255. Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction:		Float 0-1. Fraction of grayspace which causes a tile to be discarded.
                                        If 1, will not perform grayspace filtering.
            grayspace_threshold:	Int 0-1. HSV (hue, saturation, value) is calculated for each pixel.
                                        If a pixel's saturation is below this threshold, it is considered grayspace.
            full_core:				Bool. Only used for TMA. If true, will extract full image cores
                                        regardless of supplied tile micron size.
        '''
        assert img_format in ('png', 'jpg', 'jpeg')
        if tfrecord_dir is None and tiles_dir is None:
            raise UserError("Must supply either tfrecord_dir or tiles_dir as destination for tile extraction.")

        # Make base directories
        if tfrecord_dir:
            if not exists(tfrecord_dir): os.makedirs(tfrecord_dir)
        if tiles_dir:
            if not os.path.exists(tiles_dir): os.makedirs(tiles_dir)

        if tfrecord_dir or tiles_dir:
            # Log to keep track of when tiles have finished extracting
            # To be used in case tile extraction is interrupted, so the slide can be flagged for re-extraction
            unfinished_marker = join((tfrecord_dir if tfrecord_dir else tiles_dir), f'{self.name}.unfinished')
            with open(unfinished_marker, 'w') as marker_file:
                marker_file.write(' ')

            if split_fraction and split_names:
                # Tile splitting error checking
                if len(split_fraction) != len(split_names):
                    raise InvalidTileSplitException(f'When splitting tiles, "fraction" length ({len(split_fraction)})' + \
                                                        f' should equal length of "names" ({len(split_names)})')
                if sum([i for i in split_fraction if i != -1]) > 1:
                    raise InvalidTileSplitException("Unable to split tiles; sum of split_fraction is greater than 1")
                # Calculate dynamic splitting
                if -1 in split_fraction:
                    num_to_dynamic_split = sum([i for i in split_fraction if i == -1])
                    dynamic_fraction = (1 - sum([i for i in split_fraction if i != -1])) / num_to_dynamic_split
                    split_fraction = [s if s != -1 else dynamic_fraction for s in split_fraction]
                # Prepare subfolders for splitting
                if tfrecord_dir:
                    tfrecord_writers = []
                    for name in split_names:
                        if not exists(join(tfrecord_dir, name)):
                            os.makedirs(join(tfrecord_dir, name))
                            tfrecord_writers += [tf.io.TFRecordWriter(join(tfrecord_dir, name, self.name+".tfrecords"))]
                if tiles_dir:
                    for name in split_names:
                        if not exists(join(tiles_dir, name)):
                            os.makedirs(join(tiles_dir, name))
            elif tfrecord_dir:
                tfrecord_writer = tf.io.TFRecordWriter(join(tfrecord_dir, self.name+".tfrecords"))

        generator = self.build_generator(shuffle=shuffle, num_threads=num_threads, **kwargs)
        slidename_bytes = bytes(self.name, 'utf-8')

        if not generator:
            log.error(f"No tiles extracted from slide {sfutil.green(self.name)}")
            return

        sample_tiles = []
        generator_iterator = generator()

        # If not using a multiprocessing progress bar, use tqdm
        if self.counter_lock is None:
            generator_iterator = tqdm(generator_iterator, total=self.estimated_num_tiles, ncols=80)

        for index, tile_dict in enumerate(generator_iterator):

            # Increase multiprocessing progress bar, if provided
            if self.counter_lock is not None:
                with self.counter_lock:
                    self.pb_counter.value += 1

            tile = tile_dict['image']
            location = tile_dict['loc']
            # Convert numpy array (in RGB) to jpeg string using CV2 (which first requires BGR format)
            if img_format.lower() == 'png':
                image_string = cv2.imencode('.png', cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))[1].tobytes()
            elif img_format.lower() in ('jpg', 'jpeg'):
                image_string = cv2.imencode('.jpg',
                                            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
            else:
                raise ValueError(f"Unknown image format {img_format}, must be either 'png' or 'jpg'")
            if len(sample_tiles) < 10:
                sample_tiles += [image_string]
            elif not tiles_dir and not tfrecord_dir:
                break
            if tiles_dir:
                if split_fraction and split_names:
                    save_dir = join(tiles_dir, random.choices(split_names, weights=split_fraction))
                else:
                    save_dir = tiles_dir
                with open(join(save_dir, f'{self.shortname}_{index}.{img_format}'), 'wb') as outfile:
                    outfile.write(image_string)
            if tfrecord_dir:
                if split_fraction and split_names:
                    writer = random.choices(tfrecord_writers, weights=split_fraction)
                else:
                    writer = tfrecord_writer
                tf_example = tfrecord_example(slidename_bytes, image_string, location[0], location[1])
                writer.write(tf_example.SerializeToString())

        if tfrecord_dir or tiles_dir:
            # Mark extraction of current slide as finished
            try:
                os.remove(unfinished_marker)
            except:
                log.error(f"Unable to mark slide {self.name} as tile extraction complete")

        # Unbuffer slide
        self.slide.unbuffer()

        # Generate extraction report
        report = SlideReport(sample_tiles, self.slide.path)

        log.info(f"Finished tile extraction for slide {sfutil.green(self.shortname)}")

        return report

class WSI(BaseLoader):
    '''Extension of slideflow.slide.BaseLoader. Loads a slide and its annotated region of interest (ROI).'''

    def __init__(self,
                 path,
                 tile_px,
                 tile_um,
                 stride_div=1,
                 enable_downsample=False,
                 roi_dir=None,
                 roi_list=None,
                 roi_method=EXTRACT_INSIDE,
                 skip_missing_roi=False,
                 randomize_origin=False,
                 silent=False,
                 buffer=None,
                 pb=None,
                 pb_counter=None,
                 counter_lock=None,
                 print_fn=None):

        '''Initializer.

        Args:
            path:				Path to slide
            tile_px:			Size of tiles to extract, in pixels
            tile_um:			Size of tiles to extract, in microns
            stride_div:			Stride divisor for tile extraction (1 = no tile overlap; 2 = 50% overlap, etc)
            enable_downsample:	Bool, if True, allows use of downsampled intermediate layers in the slide image pyramid,
                                    which greatly improves tile extraction speed.
            roi_dir:			Directory in which to search for ROI CSV files
            roi_list:			Alternatively, a list of ROI paths can be explicitly provided
            roi_method:			Either inside, outside, or ignore. Determines how ROIs are used to extract tiles
            skip_missing_roi:	Bool, if True, will skip tiles that are missing a ROI file
            silent:				Bool, if True, will hide logging output
            buffer:				Path to directory. Slides will be copied to the directory as a buffer before extraction.
                                    Vastly improves extraction speed if an SSD or ramdisk buffer is used.
            pb:					ProgressBar instance; will update progress bar during tile extraction if provided
            pb_id:				ID of bar in ProgressBar, defaults to 0
        '''
        super().__init__(path,
                         tile_px,
                         tile_um,
                         stride_div,
                         enable_downsample,
                         silent,
                         buffer,
                         pb,
                         pb_counter,
                         counter_lock,
                         print_fn)

        # Initialize calculated variables
        self.extracted_x_size = 0
        self.extracted_y_size = 0
        self.estimated_num_tiles = 0
        self.coord = []
        self.annPolys = []
        self.ROI_SCALE = 10
        self.rois = []
        self.roi_method = roi_method

        if not self.loaded_correctly():
            return

        # Build coordinate grid
        self._build_coord(randomize_origin=randomize_origin)

        # Look in ROI directory if available
        if roi_dir and exists(join(roi_dir, self.name + ".csv")):
            self.load_csv_roi(join(roi_dir, self.name + ".csv"))

        # Else try loading ROI from same folder as slide
        elif exists(self.name + ".csv"):
            self.load_csv_roi(sfutil.path_to_name(path) + ".csv")
        elif roi_list and self.name in [sfutil.path_to_name(r) for r in roi_list]:
            matching_rois = []
            for rp in roi_list:
                rn = sfutil.path_to_name(rp)
                if rn == self.name:
                    matching_rois += [rp]
            if len(matching_rois) > 1:
                log.warning(f" Multiple matching ROIs found for {self.name}; using {matching_rois[0]}")
            self.load_csv_roi(matching_rois[0])

        # Handle missing ROIs
        if not len(self.rois) and skip_missing_roi and roi_method != 'ignore':
            log.error(f"No ROI found for {sfutil.green(self.name)}, skipping slide")
            self.shape = None
            self.load_error = True
            return None
        elif not len(self.rois):
            self.estimated_num_tiles = int(len(self.coord))
            log.warning(f"[{sfutil.green(self.shortname)}]  No ROI found in {roi_dir}, using whole slide.")
            self.roi_method = 'ignore'

        mpp_roi_msg = f'{self.MPP} um/px | {len(self.rois)} ROI(s)'
        size_msg = f'Size: {self.full_shape[0]} x {self.full_shape[1]}'
        log.info(f"{self.shortname}: Slide info: {mpp_roi_msg} | {size_msg}")

        # Abort if errors were raised during ROI loading
        if self.load_error:
            log.error(f'Skipping slide {sfutil.green(self.name)} due to loading error')
            return None

    def _build_coord(self, randomize_origin):
        # Calculate window sizes, strides, and coordinates for windows
        self.extracted_x_size = self.full_shape[0] - self.full_extract_px
        self.extracted_y_size = self.full_shape[1] - self.full_extract_px

        # Randomize origin, if desired
        if randomize_origin:
            start_x = random.randint(0, self.full_stride-1)
            start_y = random.randint(0, self.full_stride-1)
            log.info(f"Random origin: X: {start_x}, Y: {start_y}")
        else:
            start_x = start_y = 0

        # Coordinates must be in level 0 (full) format for the read_region function
        index = 0
        y_range = np.arange(start_y, (self.full_shape[1]+1) - self.full_extract_px, self.full_stride)
        x_range = np.arange(start_x, (self.full_shape[0]+1) - self.full_extract_px, self.full_stride)
        for yi, y in enumerate(y_range):
            for xi, x in enumerate(x_range):
                y = int(y)
                x = int(x)
                is_unique = ((y % self.full_extract_px == 0) and (x % self.full_extract_px == 0))
                self.coord.append([x, y, index, is_unique, xi, yi])
                index += 1

        self.grid = np.zeros((len(x_range), len(y_range)))
        log.info(f"Grid shape: {self.grid.shape}")

    def build_generator(self, dual_extract=False, shuffle=True, whitespace_fraction=1.0,
                            whitespace_threshold=230, grayspace_fraction=0.6, grayspace_threshold=0.05,
                            normalizer=None, normalizer_source=None, include_loc=True, num_threads=4, 
                            show_progress=False, **kwargs):

        '''Builds generator to supervise extraction of tiles across the slide.

        Args:
            dual_extract:			If true, will extract base image and the surrounding region.
            shuffle:				If true, will shuffle images during extraction
            whitespace_fraction:	Float from 0-1, representing a percent. Tiles with this percent of pixels
                                        (or more) classified as "whitespace" and will be skipped during extraction.
            whitespace_threshold:	Int from 0-255, pixel brightness above which a pixel is considered whitespace
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
        '''
        super().build_generator()

        if self.estimated_num_tiles == 0:
            log.warning(f"No tiles extracted at the given micron size for slide {sfutil.green(self.name)}")
            return None

        # Shuffle coordinates to randomize extraction order
        if shuffle:
            random.shuffle(self.coord)

        worker_args = {
            'full_extract_px': self.full_extract_px,
            'ROI_SCALE': self.ROI_SCALE,
            'roi_method': self.roi_method,
            'annPolys': self.annPolys,
            'estimated_num_tiles': self.estimated_num_tiles,
            'downsample_level': self.downsample_level,
            'path': self.path,
            'extract_px': self.extract_px,
            'tile_px': self.tile_px,
            'dual_extract': dual_extract,
            'full_stride': self.full_stride,
            'normalizer': normalizer,
            'normalizer_source': normalizer_source,
            'whitespace_fraction': whitespace_fraction,
            'whitespace_threshold': whitespace_threshold,
            'grayspace_fraction': grayspace_fraction,
            'grayspace_threshold': grayspace_threshold,
            'include_loc': include_loc
        }
        worker_args = types.SimpleNamespace(**worker_args)

        def generator():
            log.debug(f"Building tile extraction generator with {num_threads} thread workers")
            self.tile_mask = np.asarray([False for i in range(len(self.coord))], dtype=np.bool)

            with mp.Pool(processes=num_threads) as p:
                iterator = p.imap(partial(slide_extraction_worker, args=worker_args), self.coord)
                if show_progress:
                    iterator = tqdm(iterator, total=self.estimated_num_tiles, ncols=80)
                for res in iterator:
                    if res == 'skip':
                        continue
                    if show_progress:
                        iterator.update(1)
                    if res is None:
                        continue
                    else:
                        tile, idx = res
                        self.tile_mask[idx] = True
                        yield tile
                if show_progress:
                    iterator.close()

            name_msg = sfutil.green(self.shortname)
            num_msg = f'({np.sum(self.tile_mask)} tiles of {len(self.coord)} possible)'
            log.info(f"{self.shortname}: Finished tile extraction for {name_msg} {num_msg}")

        return generator

    def annotated_thumb(self, mpp=None, width=None):
        '''Returns PIL Image of thumbnail with ROI overlay.

        Args:
            mpp:	Microns-per-pixel, used to determine thumbnail size

        Returns:
            PIL image
        '''
        if mpp is not None:
            ROI_SCALE = self.full_shape[0] / (int((self.MPP * self.full_shape[0]) / mpp))
        else:
            ROI_SCALE = self.full_shape[0] / width
        annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.rois]
        annotated_thumb = self.thumb(mpp=mpp, width=width).copy()
        draw = ImageDraw.Draw(annotated_thumb)
        for poly in annPolys:
            x,y = poly.exterior.coords.xy
            zipped = list(zip(x.tolist(),y.tolist()))
            draw.polygon(zipped, outline="#000000")
        return annotated_thumb

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
                log.error(f'Unable to read CSV ROI file {sfutil.green(path)}, please check file integrity and ' + \
                                'ensure headers contain "ROI_name", "X_base", and "Y_base".')
                self.load_error = True
                return
            for row in reader:
                roi_name = row[index_name]
                x_coord = int(float(row[index_x]))
                y_coord = int(float(row[index_y]))

                if roi_name not in roi_dict:
                    roi_dict.update({roi_name: ROIObject(roi_name)})
                roi_dict[roi_name].add_coord((x_coord, y_coord))

            for roi_object in roi_dict.values():
                self.rois.append(roi_object)

        # Load annotations as shapely.geometry objects
        if self.roi_method != IGNORE_ROI:
            self.annPolys = []
            for i, annotation in enumerate(self.rois):
                try:
                    self.annPolys += [sg.Polygon(annotation.scaled_area(self.ROI_SCALE))]
                except ValueError:
                    log.warning(f"Unable to use ROI {i} in slide {sfutil.green(self.name)}, at least 3 points required " + \
                                "to create a geometric shape.")
            roi_area = sum([poly.area for poly in self.annPolys])
        else:
            roi_area = 1
        total_area = (self.full_shape[0]/self.ROI_SCALE) * (self.full_shape[1]/self.ROI_SCALE)
        roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

        if self.roi_method == EXTRACT_INSIDE:
            self.estimated_num_tiles = int(len(self.coord) * roi_area_fraction)
        else:
            self.estimated_num_tiles = int(len(self.coord) * (1-roi_area_fraction))

        return len(self.rois)

    def load_json_roi(self, path, scale=10):
        '''Loads ROI from a JSON file.'''
        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], scale)
            self.rois.append(ROIObject(f"Object{len(self.rois)}"))
            self.rois[-1].add_shape(area_reduced)
        return len(self.rois)

    def predict(self, model, layers=['postconv'], normalizer=None, normalizer_source=None, whitespace_fraction=1.0,
                    whitespace_threshold=230, grayspace_fraction=0.6, grayspace_threshold=0.05,
                    batch_size=128, dtype=np.float16, **kwargs):

        from slideflow.model import ModelActivationsInterface
        model_interface = ModelActivationsInterface(model, layers=layers)
        prediction_grid = np.zeros((self.grid.shape[0], self.grid.shape[1], model_interface.num_features), dtype=dtype)

        generator = self.build_generator(shuffle=False,
                                        normalizer=normalizer,
                                        normalizer_source=normalizer_source,
                                        whitespace_fraction=whitespace_fraction,
                                        whitespace_threshold=whitespace_threshold,
                                        grayspace_fraction=grayspace_fraction,
                                        grayspace_threshold=grayspace_threshold,
                                        include_loc='grid',
                                        **kwargs)

        if not generator:
            log.error(f"No tiles extracted from slide {sfutil.green(self.name)}")
            return

        def _parse_function(record):
            image = record['image']
            loc = record['loc']
            parsed_image = tf.image.per_image_standardization(image)
            parsed_image = tf.image.convert_image_dtype(parsed_image, tf.float32)
            parsed_image.set_shape([self.tile_px, self.tile_px, 3])
            return parsed_image, loc

        # Generate dataset from the generator
        log.info("Setting up tile generator")
        with tf.name_scope('dataset_input'):
            output_signature={'image':tf.TensorSpec(shape=(self.tile_px,self.tile_px,3), dtype=tf.uint8),
                              'loc':tf.TensorSpec(shape=(2), dtype=tf.uint32)}
            tile_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            tile_dataset = tile_dataset.map(_parse_function, num_parallel_calls=8)
            tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
            tile_dataset = tile_dataset.prefetch(8)

        act_arr = []
        loc_arr = []
        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            act, logits = model_interface.predict(batch_images)
            act_arr += [act]
            loc_arr += [batch_loc.numpy()]
        act_arr = np.concatenate(act_arr)
        loc_arr = np.concatenate(loc_arr)

        for i, act in enumerate(act_arr):
            xi = loc_arr[i][0]
            yi = loc_arr[i][1]
            prediction_grid[xi][yi] = act

        return prediction_grid

class TMA(BaseLoader):
    '''Extension of slideflow.slide.BaseLoader. Loads a TMA-formatted slide and detects tissue cores.'''

    QUEUE_SIZE = 8
    NUM_EXTRACTION_WORKERS = 8
    HEIGHT_MIN = 20
    WIDTH_MIN = 20
    BLACK = (0,0,0)
    BLUE = (255, 100, 100)
    GREEN = (75,220,75)
    LIGHTBLUE = (255, 180, 180)
    RED = (100, 100, 200)
    WHITE = (255,255,255)

    def __init__(self, path, tile_px, tile_um, stride_div, annotations_dir=None,
                    enable_downsample=False, silent=False, report_dir=None, buffer=None, pb=None, pb_id=0):
        '''Initializer.

        Args:
            path:				Path to slide
            tile_px:			Size of tiles to extract, in pixels
            tile_um:			Size of tiles to extract, in microns
            stride_div:			Stride divisor for tile extraction (1 = no tile overlap; 2 = 50% overlap, etc)
            enable_downsample:	Bool, if True, allows use of downsampled intermediate layers in the slide image pyramid,
                                    which greatly improves tile extraction speed.
            silent:				Bool, if True, will hide logging output
            buffer:				Path to directory. Slides will be copied here prior to extraction.
            pb:					ProgressBar instance; will update progress bar during tile extraction if provided
            pb_id:				ID of bar in ProgressBar, defaults to 0
        '''
        super().__init__(path, tile_px, tile_um, stride_div, enable_downsample, silent, buffer, pb)

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
        log.info(f"{self.shortname}: Slide info: {self.MPP} um/px | {size_msg}")

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
        current_MPP = self.MPP * self.downsample_factor
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
                area = polyArea([b[0] for b in box], [b[1] for b in box])
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

    def build_generator(self, shuffle=True, whitespace_fraction=1.0, whitespace_threshold=230,
                            grayspace_fraction=0.6, grayspace_threshold=0.05,
                            normalizer=None, normalizer_source=None, **kwargs):

        '''Builds generator to supervise extraction of tiles across the slide.

        Args:
            shuffle:				If true, will shuffle images during extraction
            whitespace_fraction:	Float from 0-1, representing a percent. Tiles with this percent of pixels
                                        (or more) classified as "whitespace" and will be skipped during extraction.
            whitespace_threshold:	Int from 0-255, pixel brightness above which a pixel is considered whitespace
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
            export_full_core:	If true, will also save a thumbnail of each fully extracted core.'''

        super().build_generator()

        # Process kwargs
        full_core = None if 'full_core' not in kwargs else kwargs['full_core']

        # Setup normalization
        normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        # Shuffle TMAs
        if shuffle:
            random.shuffle(self.object_rects)

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
            extraction_pool = mp.Pool(self.NUM_EXTRACTION_WORKERS, section_extraction_worker,(rectangle_queue, extraction_queue,))

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
                        yield cv2.resize(image_core, (self.tile_px, self.tile_px))
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

                            yield subtile

            extraction_pool.close()

        return generator
