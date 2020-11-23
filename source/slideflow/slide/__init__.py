# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''This module includes tools to convolutionally section whole slide images into tiles
using python Generators. These tessellated tiles can be exported as JPGs, with or without
data augmentation, or used as input for a trained Tensorflow model. Model predictions 
can then be visualized as a heatmap overlay.

Requires: libvips (https://libvips.github.io/libvips/).'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import io
import tensorflow as tf
import numpy as np
import imageio
import argparse
import pickle
import csv
import pyvips as vips
import shapely.geometry as sg
import cv2
import json
import time
import multiprocessing
import random
import tempfile
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
import slideflow.util as sfutil

from os.path import join, isfile, exists
from math import sqrt
from PIL import Image, ImageDraw
from multiprocessing.dummy import Pool as DPool
from multiprocessing import Process, Pool, Queue
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
from slideflow.util import log, StainNormalizer
from slideflow.util.fastim import FastImshow
from slideflow.io.tfrecords import image_example
from statistics import mean, median
from pathlib import Path
from fpdf import FPDF
from datetime import datetime

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 100000000000
DEFAULT_JPG_MPP = 1
OPS_LEVEL_COUNT = 'openslide.level-count'
OPS_MPP_X = 'openslide.mpp-x'
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

class InvalidTileSplitException(Exception):
	'''Raised when invalid tile splitting parameters are given to SlideReader.'''
	pass

class TileCorruptionError(Exception):
	'''Raised when image normalization fails due to tile corruption.'''
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
					pdf.image(temp.name, x, y, w=19*len(report.images), h=19, type='jpg')
			pdf.ln(20)
			
		self.pdf = pdf

	def save(self, filename):
		self.pdf.output(filename)

class OpenslideToVIPS:
	'''Wrapper for VIPS to preserve openslide-like functions.'''
	def __init__(self, path, buffer=None):
		self.path = path
		self.buffer = buffer
		
		if buffer == 'vmtouch':
			os.system(f'vmtouch -q -t "{self.path}"')
		self.full_image = vips.Image.new_from_file(path, fail=True, access=vips.enums.Access.RANDOM)
		loaded_image = self.full_image
		
		self.properties = {}
		for field in loaded_image.get_fields():
			self.properties.update({field: loaded_image.get(field)})
		self.dimensions = (int(self.properties[OPS_WIDTH]), int(self.properties[OPS_HEIGHT]))
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
				downsampled_image = vips.Image.new_from_file(self.path, level=level, fail=True, access=vips.enums.Access.RANDOM)
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

class JPGslideToVIPS(OpenslideToVIPS):
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

class SlideLoader:
	'''Object that loads an SVS slide and makes preparations for tile extraction.
	Should not be used directly; this class must be inherited and extended by a child class!'''
	def __init__(self, path, size_px, size_um, stride_div, enable_downsample=False,
					silent=False, buffer=None, pb=None):
		self.load_error = False
		self.silent = silent

		self.print = None if silent else (print if not pb else pb.print)
		self.error_print = print if not pb else pb.print

		self.pb = pb
		self.name = sfutil.path_to_name(path)
		self.shortname = sfutil._shortname(self.name)
		self.size_px = size_px
		self.size_um = size_um
		self.tile_mask = None
		self.enable_downsample = enable_downsample
		self.thumb_image = None
		filetype = sfutil.path_to_ext(path)

		# Initiate supported slide reader
		if filetype.lower() in sfutil.SUPPORTED_FORMATS:
			if filetype.lower() == 'jpg':
				self.slide = JPGslideToVIPS(path)
			else:
				self.slide = OpenslideToVIPS(path, buffer=buffer)
		else:
			log.error(f"Unsupported file type '{filetype}' for slide {self.name}.", 1, self.error_print)
			self.load_error = True
			return

		# Collect basic slide information
		try:
			self.MPP = float(self.slide.properties[OPS_MPP_X])
		except KeyError:
			log.error(f"Slide {sfutil.green(self.name)} missing microns-per-pixel property ({OPS_MPP_X}), unable to process", 1, self.error_print)
			self.load_error = True
			return
		self.full_shape = self.slide.dimensions
		self.full_extract_px = int(self.size_um / self.MPP)

		# Load downsampled level based on desired extraction size
		downsample_desired = self.full_extract_px / size_px
		if enable_downsample:
			self.downsample_level = self.slide.get_best_level_for_downsample(downsample_desired)
		else:
			self.downsample_level = 0
		self.downsample_factor = self.slide.level_downsamples[self.downsample_level]
		self.shape = self.slide.level_dimensions[self.downsample_level]

		# Calculate pixel size of extraction window using downsampling
		self.extract_px = int(self.full_extract_px / self.downsample_factor)
		self.full_stride = self.full_extract_px / stride_div
		self.stride = self.extract_px / stride_div

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

	def thumb(self, mpp=55):
		'''Returns PIL thumbnail of the slide.
		
		Args:
			mpp:	Microns-per-pixel of thumbnail (determines size of thumbnail to return)
			
		Returns:
			PIL image
		'''
		if not self.thumb_image:
			# Get thumbnail image and dimensions via fastest method available
			if 'slide-associated-images' in self.slide.properties and 'thumbnail' in self.slide.properties['slide-associated-images']:
				thumbnail = vips.Image.openslideload(self.slide.path, associated='thumbnail')
			elif self.enable_downsample:
				level = max(0, self.slide.level_count-2)
				thumbnail = self.slide.get_downsampled_image(level)
			else:
				thumbnail = self.slide.full_image.thumbnail_image(width)

			# Calculate goal width/height according to specified microns-per-pixel (MPP)
			width = int((self.MPP * self.full_shape[0]) / mpp)
			height = int((self.MPP * self.full_shape[1]) / mpp)

			# Convert thumb image to PIL
			np_thumb = vips2numpy(thumbnail)
			self.thumb_image = Image.fromarray(np_thumb).resize((width, height))
		return self.thumb_image

	def build_generator(self):
		log.label(self.shortname, f"Extracting {sfutil.bold(self.size_um)}um tiles (resizing {sfutil.bold(self.extract_px)}px -> {sfutil.bold(self.size_px)}px); stride: {sfutil.bold(int(self.stride))}px", 2, self.print)
		if self.size_px > self.extract_px:
			log.label(self.shortname, f"[{sfutil.fail('!WARN!')}] Tiles will be up-scaled with bilinear interpolation, ({self.extract_px}px -> {self.size_px}px)", 2, self.print)
	
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
						whitespace_fraction=1.0, whitespace_threshold=230, grayspace_fraction=0.6, grayspace_threshold=0.05,
						normalizer=None, normalizer_source=None, shuffle=True, **kwargs):
		'''Extractes tiles from slide and saves into a TFRecord file or as loose JPG tiles in a directory.
		Args:
			tfrecord_dir:			If provided, saves tiles into a TFRecord file (named according to slide name) in this directory.
			tiles_dir:				If provided, saves loose JPG tiles into a subdirectory (named according to slide name) in this directory.
			split_fraction:			List of float. If provided, splits the extracted tiles into subsets (e.g. for validation set) using these fractions.
										Should add up to 1 (except for fractions of -1). Remaining tiles are split between fractions of "-1".
			split_names:			List of names to label the split fractions
			whitespace_fraction:	Int from 0-100, representing a percent. Tiles with this percent of pixels classified as "whitespace" 
										will be skipped during extraction.
			whitespace_threshold:	Int from 0-255, pixel brightness above which a pixel is considered whitespace
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
			full_core:				Bool. Only used for TMAReader. If true, will extract full image cores regardless of supplied tile micron size.
		'''
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
					raise InvalidTileSplitException(f'When splitting tiles, length of "fraction" ({len(split_fraction)}) should equal length of "names" ({len(split_names)})')
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

		generator = self.build_generator(shuffle=shuffle,
										 normalizer=normalizer,
										 normalizer_source=normalizer_source,
										 whitespace_fraction=whitespace_fraction,
										 whitespace_threshold=whitespace_threshold,
										 grayspace_fraction=grayspace_fraction,
										 grayspace_threshold=grayspace_threshold,
										 **kwargs)
		slidename_bytes = bytes(self.name, 'utf-8')

		if not generator:
			log.error(f"No tiles extracted from slide {sfutil.green(self.name)}", 1, self.print)
			return

		sample_tiles = []
		for index, tile in enumerate(generator()):
			# Convert numpy array (in RGB) to jpeg string using CV2 (which first requires BGR format)
			image_string = cv2.imencode('.jpg', cv2.cvtColor(tile, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
			if len(sample_tiles) < 10:
				sample_tiles += [image_string]
			elif not tiles_dir and not tfrecord_dir:
				break
			if tiles_dir:
				if split_fraction and split_names:
					save_dir = join(tiles_dir, random.choices(split_names, weights=split_fraction))
				else:
					save_dir = tiles_dir
				with open(join(tiles_dir, f'{self.shortname}_{index}.jpg'), 'wb') as outfile:
					outfile.write(image_string)
			if tfrecord_dir:
				if split_fraction and split_names:
					writer = random.choices(tfrecord_writers, weights=split_fraction)
				else:
					writer = tfrecord_writer
				tf_example = image_example(slidename_bytes, image_string)
				writer.write(tf_example.SerializeToString())
		
		if tfrecord_dir or tiles_dir:
			# Mark extraction of current slide as finished
			try:
				os.remove(unfinished_marker)
			except:
				log.error(f"Unable to mark slide {self.name} as tile extraction complete", 1)

		# Unbuffer slide
		self.slide.unbuffer()

		# Generate extraction report
		report = SlideReport(sample_tiles, self.slide.path)

		log.complete(f"Finished tile extraction for slide {sfutil.green(self.shortname)}", 1, self.print)

		return report

class SlideReader(SlideLoader):
	'''Extension of slideflow.slide.SlideLoader. Loads a slide and its ROI annotations and sets up a tile generator.'''

	ROI_SCALE = 10

	def __init__(self, path, size_px, size_um, stride_div=1, enable_downsample=False, roi_dir=None, roi_list=None,
					roi_method=EXTRACT_INSIDE, skip_missing_roi=True, silent=False, buffer=None, pb=None, pb_id=0):
		'''Initializer.

		Args:
			path:				Path to slide
			size_px:			Size of tiles to extract, in pixels
			size_um:			Size of tiles to extract, in microns
			stride_div:			Stride divisor for tile extraction (1 = no tile overlap; 2 = 50% overlap, etc)
			enable_downsample:	Bool, if True, allows use of downsampled intermediate layers in the slide image pyramid,
									which greatly improves tile extraction speed.
			roi_dir:			Directory in which to search for ROI CSV files
			roi_list:			Alternatively, a list of ROI paths can be explicitly provided
			roi_method:			Either inside, outside, or ignore. Determines how ROIs are used to extract tiles
			skip_missing_roi:	Bool, if True, will skip tiles that are missing a ROI file
			silent:				Bool, if True, will hide logging output
			buffer:				Either 'vmtouch' or path to directory. If vmtouch, will use vmtouch to preload slide into memory before extraction.
									If a directory, slides will be copied to the directory as a buffer before extraction.
									Either method vastly improves tile extraction for slides on HDDs by maximizing sequential read speed
			pb:					ProgressBar instance; will update progress bar during tile extraction if provided
			pb_id:				ID of bar in ProgressBar, defaults to 0
		'''
		super().__init__(path, size_px, size_um, stride_div, enable_downsample, silent, buffer, pb)

		# Initialize calculated variables
		self.extracted_x_size = 0
		self.extracted_y_size = 0
		self.estimated_num_tiles = 0
		self.coord = []
		self.annPolys = []
		self.pb_id = pb_id

		if not self.loaded_correctly():
			return

		# Establish ROIs
		self.rois = []
		self.roi_method = roi_method

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
				log.warn(f" Multiple matching ROIs found for {self.name}; using {matching_rois[0]}", 1, self.print)
			self.load_csv_roi(matching_rois[0])	
		
		# Handle missing ROIs
		if not len(self.rois) and skip_missing_roi:
			log.error(f"No ROI found for {sfutil.green(self.name)}, skipping slide", 1, self.error_print)
			self.shape = None
			self.load_error = True
			return None
		elif not len(self.rois):
			# Calculate window sizes, strides, and coordinates for windows
			self.extracted_x_size = self.full_shape[0] - self.full_extract_px
			self.extracted_y_size = self.full_shape[1] - self.full_extract_px

			# Coordinates must be in level 0 (full) format for the read_region function
			index = 0
			for y in np.arange(0, (self.full_shape[1]+1) - self.full_extract_px, self.full_stride):
				for x in np.arange(0, (self.full_shape[0]+1) - self.full_extract_px, self.full_stride):
					y = int(y)
					x = int(x)
					is_unique = ((y % self.full_extract_px == 0) and (x % self.full_extract_px == 0))
					self.coord.append([x, y, index, is_unique])
					index += 1

			self.estimated_num_tiles = int(len(self.coord)) 
			log.warn(f"[{sfutil.green(self.shortname)}]  No ROI found in {roi_dir}, using whole slide.", 2, self.print)
				
				
		log.label(self.shortname, f"Slide info: {self.MPP} um/px | {len(self.rois)} ROI(s) | Size: {self.full_shape[0]} x {self.full_shape[1]}", 2, self.print)

		# Abort if errors were raised during ROI loading
		if self.load_error:
			log.error(f'Skipping slide {sfutil.green(self.name)} due to slide image or ROI loading error', 1, self.error_print)
			return None

	def build_generator(self, dual_extract=False, shuffle=True, whitespace_fraction=1.0, whitespace_threshold=230,
							grayspace_fraction=0.6, grayspace_threshold=0.05, normalizer=None, normalizer_source=None, **kwargs):
		'''Builds generator to supervise extraction of tiles across the slide.
		
		Args:
			dual_extract:			If true, will extract base image and the surrounding region.
			shuffle:				If true, will shuffle images during extraction
			whitespace_fraction:	Float from 0-1, representing a percent. Tiles with this percent of pixels (or more) classified as "whitespace" 
										will be skipped during extraction.
			whitespace_threshold:	Int from 0-255, pixel brightness above which a pixel is considered whitespace
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
		'''
		super().build_generator()

		if self.estimated_num_tiles == 0:
			log.warn(f"No tiles were able to be extracted at the given micron size for slide {sfutil.green(self.name)}", 1, self.print)
			return None

		# Create mask for indicating whether tile was extracted
		tile_mask = np.asarray([0 for i in range(len(self.coord))])
		
		# Shuffle coordinates to randomize extraction order
		if shuffle:
			random.shuffle(self.coord)

		# Setup normalization
		normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

		def generator():
			for c in self.coord:
				index = c[2]

				# Check if the center of the current window lies within any annotation; if not, skip
				x_coord = int((c[0]+self.full_extract_px/2)/self.ROI_SCALE)
				y_coord = int((c[1]+self.full_extract_px/2)/self.ROI_SCALE)

				if self.roi_method != IGNORE_ROI and bool(self.annPolys):
					# If the extraction method is EXTRACT_INSIDE, skip the tile if it's not in an ROI
					if (self.roi_method == EXTRACT_INSIDE) and not any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in self.annPolys]):
						continue
					# If the extraction method is EXTRACT_OUTSIDE, skip the tile if it's in an ROI
					elif (self.roi_method == EXTRACT_OUTSIDE) and any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in self.annPolys]):
						continue
				if self.pb:
					self.pb.increase_bar_value(id=self.pb_id)

				# Read the region and resize to target size
				region = self.slide.read_region((c[0], c[1]), self.downsample_level, [self.extract_px, self.extract_px])
				region = region.resize(float(self.size_px) / self.extract_px)

				# Read regions into memory and convert to numpy arrays
				np_image = vips2numpy(region)[:,:,:-1]

				if dual_extract:
					try:
						surrounding_region = self.slide.read_region((c[0]-self.full_stride, c[1]-self.full_stride), self.downsample_level, [self.extract_px*3, self.extract_px*3])
						surrounding_region = surrounding_region.resize(float(self.size_px) / (self.extract_px*3))
						outer_region = vips2numpy(surrounding_region)[:,:,:-1]
					except:
						continue
					
					# Apply normalization
					if normalizer:
						np_image = normalizer.rgb_to_rgb(np_image)
						outer_region = normalizer.rgb_to_rgb(outer_region)
					
					# Mark as extracted
					tile_mask[index] = 1

					yield {"input_1": np_image, "input_2": outer_region}, index
				else:
					# Perform whitespace filtering
					if whitespace_fraction < 1:
						fraction = (np.mean(np_image, axis=2) > whitespace_threshold).sum() / (self.size_px**2)
						if fraction > whitespace_fraction: continue

					# Perform grayspace filtering
					if grayspace_fraction < 1:
						hsv_image = mcol.rgb_to_hsv(np_image)
						fraction = (hsv_image[:,:,1] < grayspace_threshold).sum() / (self.size_px**2)
						if fraction > grayspace_fraction: continue

					# Apply normalization
					if normalizer:
						try:
							np_image = normalizer.rgb_to_rgb(np_image)
						except:
							# The image could not be normalized, which happens when a tile is primarily one solid color (background)
							continue

					# Mark as extracted
					tile_mask[index] = 1

					yield np_image

			log.label(self.shortname, f"Finished tile extraction for {sfutil.green(self.shortname)} ({sum(tile_mask)} tiles of {len(self.coord)} possible)", 2, self.print)
			self.tile_mask = tile_mask

		return generator

	def annotated_thumb(self, mpp=55):
		'''Returns PIL Image of thumbnail with ROI overlay.
		
		Args:
			mpp:	Microns-per-pixel, used to determine thumbnail size
			
		Returns:
			PIL image
		'''
		ROI_SCALE = self.full_shape[0]/(int((self.MPP * self.full_shape[0]) / mpp))
		annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.rois]
		annotated_thumb = self.thumb(mpp=mpp).copy()
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
				log.error(f'Unable to read CSV ROI file {sfutil.green(path)}, please check file integrity and ensure headers contain "ROI_name", "X_base", and "Y_base".', 1, self.error_print)
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

		# Calculate window sizes, strides, and coordinates for windows
		self.extracted_x_size = self.full_shape[0] - self.full_extract_px
		self.extracted_y_size = self.full_shape[1] - self.full_extract_px

		# Coordinates must be in level 0 (full) format for the read_region function
		index = 0
		for y in np.arange(0, (self.full_shape[1]+1) - self.full_extract_px, self.full_stride):
			for x in np.arange(0, (self.full_shape[0]+1) - self.full_extract_px, self.full_stride):
				y = int(y)
				x = int(x)
				is_unique = ((y % self.full_extract_px == 0) and (x % self.full_extract_px == 0))
				self.coord.append([x, y, index, is_unique])
				index += 1

		# Load annotations as shapely.geometry objects
		if self.roi_method != IGNORE_ROI:
			self.annPolys = []
			for i, annotation in enumerate(self.rois):
				try:
					self.annPolys += [sg.Polygon(annotation.scaled_area(self.ROI_SCALE))]
				except ValueError:
					log.warn(f"Unable to use ROI {i} in slide {sfutil.green(self.name)}, at least 3 points required to create a geometric shape.", 1, self.print)
			roi_area = sum([poly.area for poly in self.annPolys])
		else:
			roi_area = 1
		total_area = (self.full_shape[0]/self.ROI_SCALE) * (self.full_shape[1]/self.ROI_SCALE)
		roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

		self.estimated_num_tiles = int(len(self.coord) * roi_area_fraction) if self.roi_method == EXTRACT_INSIDE else int(len(self.coord) * (1-roi_area_fraction))

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
		
class TMAReader(SlideLoader):
	'''Extension of slideflow.slide.SlideLoader. Loads a TMA-formatted slide, detects tissue cores, and sets up a tile generator.'''
	
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

	def __init__(self, path, size_px, size_um, stride_div, annotations_dir=None, enable_downsample=False, silent=False, report_dir=None, buffer=None, pb=None, pb_id=0):
		'''Initializer.

		Args:
			path:				Path to slide
			size_px:			Size of tiles to extract, in pixels
			size_um:			Size of tiles to extract, in microns
			stride_div:			Stride divisor for tile extraction (1 = no tile overlap; 2 = 50% overlap, etc)
			enable_downsample:	Bool, if True, allows use of downsampled intermediate layers in the slide image pyramid,
									which greatly improves tile extraction speed.
			silent:				Bool, if True, will hide logging output
			buffer:				Either 'vmtouch' or path to directory. If vmtouch, will use vmtouch to preload slide into memory before extraction.
									If a directory, slides will be copied to the directory as a buffer before extraction.
									Either method vastly improves tile extraction for slides on HDDs by maximizing sequential read speed
			pb:					ProgressBar instance; will update progress bar during tile extraction if provided
			pb_id:				ID of bar in ProgressBar, defaults to 0
		'''
		super().__init__(path, size_px, size_um, stride_div, enable_downsample, silent, buffer, pb)

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
		log.label(self.shortname, f"Slide info: {self.MPP} um/px | Size: {self.full_shape[0]} x {self.full_shape[1]}", 2, self.print)

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

		extracted = vips2numpy(self.slide.read_region((region_x_min, region_y_min), self.downsample_level, (region_width, region_height)))[:,:,:-1]
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
		target_MPP = self.size_um / self.size_px
		current_MPP = self.MPP * self.downsample_factor
		resize_factor = current_MPP / target_MPP
		return cv2.resize(image_tile, (0, 0), fx=resize_factor, fy=resize_factor)

	def _split_core(self, image):
		'''Splits core into desired sub-images.'''
		height, width, channels = image.shape
		num_y = int(height / self.size_px)
		num_x = int(width  / self.size_px)

		# If the desired micron tile size is too large, expand and center the source image
		if not num_y or not num_x:
			expand_y = 0 if num_y else int((self.size_px-height)/2)+1
			expand_x = 0 if num_x else int((self.size_px-width)/2)+1
			image = cv2.copyMakeBorder(image, expand_y, expand_y, expand_x, expand_x, cv2.BORDER_CONSTANT, value=self.WHITE)
			height, width, _ = image.shape
			num_y = int(height / self.size_px)
			num_x = int(width  / self.size_px)

		y_start = int((height - (num_y * self.size_px))/2)
		x_start = int((width  - (num_x * self.size_px))/2)

		subtiles = []

		for y in range(num_y):
			for x in range(num_x):
				sub_x_start = x_start + (x * self.size_px)
				sub_y_start = y_start + (y * self.size_px)
				subtiles += [image[sub_y_start:sub_y_start+self.size_px, sub_x_start:sub_x_start+self.size_px]]

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

		log.info(f"Number of detected cores: {num_filtered}", 2, self.print)

		# Write annotated image to ExtractionReport
		if report_dir:
			cv2.imwrite(join(report_dir, "tma_extraction_report.jpg"), cv2.resize(img_annotated, (1400, 1000)))

		return num_filtered, num_filtered


	def build_generator(self, shuffle=True, whitespace_fraction=1.0, whitespace_threshold=230, grayspace_fraction=0.6, grayspace_threshold=0.05,
							normalizer=None, normalizer_source=None, **kwargs):
		'''Builds generator to supervise extraction of tiles across the slide.
		
		Args:
			shuffle:				If true, will shuffle images during extraction
			whitespace_fraction:	Float from 0-1, representing a percent. Tiles with this percent of pixels (or more) classified as "whitespace" 
										will be skipped during extraction.
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
		rectangle_queue = Queue()
		extraction_queue = Queue(self.QUEUE_SIZE)

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
			extraction_pool = Pool(self.NUM_EXTRACTION_WORKERS, section_extraction_worker,(rectangle_queue, extraction_queue,))

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
						yield cv2.resize(image_core, (self.size_px, self.size_px))
					else:
						subtiles = self._split_core(resized_core)
						for subtile in subtiles:
							# Perform whitespace filtering
							if whitespace_fraction < 1:
								fraction = (np.mean(subtile, axis=2) > whitespace_threshold).sum() / (self.size_px**2)
								if fraction > whitespace_fraction: continue

							# Perform grayspace filtering
							if grayspace_fraction < 1:
								hsv_image = mcol.rgb_to_hsv(subtile)
								fraction = (hsv_image[:,:,1] < grayspace_threshold).sum() / (self.size_px**2)
								if fraction > grayspace_fraction: continue

							# Apply normalization
							if normalizer:
								try:
									subtile = normalizer.rgb_to_rgb(subtile)
								except:
									# The image could not be normalized, which happens when a tile is primarily one solid color (background)
									continue
						
							yield subtile
					
			extraction_pool.close()
					
			#log.empty("Summary of extracted core areas (microns):", 1, self.print)
			#log.info(f"Min: {min(self.box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2, self.print)
			#log.info(f"Max: {max(self.box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2, self.print)
			#log.info(f"Mean: {mean(self.box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2, self.print)
			#log.info(f"Median: {median(self.box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2, self.print)

		return generator
