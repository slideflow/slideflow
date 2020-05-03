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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
import slideflow.util as sfutil

from os.path import join, isfile, exists
from math import sqrt
from PIL import Image
from multiprocessing.dummy import Pool as DPool
from multiprocessing import Process, Pool, Queue
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
from slideflow.util import log, ProgressBar
from slideflow.util.fastim import FastImshow
from statistics import mean, median
from pathlib import Path

# TODO: test JPG compatibility

Image.MAX_IMAGE_PIXELS = 100000000000
DEFAULT_JPG_MPP = 0.5
OPS_LEVEL_COUNT = 'openslide.level-count'
OPS_MPP_X = 'openslide.mpp-x'
OPS_WIDTH = 'width'
OPS_HEIGHT = 'height'
EXTRACT_INSIDE = 'inside'
EXTRACT_OUTSIDE = 'outside'

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
	return np.ndarray(buffer=vi.write_to_memory(),
					  dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
					  shape=[vi.height, vi.width, vi.bands])

class OpenslideToVIPS:
	'''Wrapper for VIPS to preserve openslide-like functions.'''
	def __init__(self, path):
		self.path = path
		self.full_image = vips.Image.new_from_file(path)
		self.properties = {}
		for field in self.full_image.get_fields():
			self.properties.update({field: self.full_image.get(field)})
		self.dimensions = (int(self.properties[OPS_WIDTH]), int(self.properties[OPS_HEIGHT]))
		self.level_count = int(self.properties[OPS_LEVEL_COUNT])
		self.loaded_downsample_levels = {
			0: self.full_image
		}

		# Calculate level metadata
		num_levels = int(self.full_image.get(OPS_LEVEL_COUNT))
		self.levels = []
		for l in range(num_levels):
			width = int(self.full_image.get(OPS_LEVEL_WIDTH(l)))
			height = int(self.full_image.get(OPS_LEVEL_HEIGHT(l)))
			downsample = float(self.full_image.get(OPS_LEVEL_DOWNSAMPLE(l)))
			self.levels += [{
				'dimensions': (width, height),
				'width': width,
				'height': height,
				'downsample': downsample
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
				downsampled_image = vips.Image.new_from_file(self.path, level=level)
				self.loaded_downsample_levels.update({
					level: downsampled_image
				})
				return downsampled_image
		else:
			return False

	def get_thumbnail(self, width, enable_downsample=False):
		'''Returns a PIL thumbnail Image of the whole slide of the given dimensions.'''
		thumbnail = self.full_image.thumbnail_image(width) 
		np_thumb = vips2numpy(thumbnail)
		pil_thumb = Image.fromarray(np_thumb)
		return pil_thumb

	def read_region(self, base_level_dim, downsample_level, extract_size):
		'''Extracts a region from the image at the given downsample level.'''
		base_level_x, base_level_y = base_level_dim
		extract_width, extract_height = extract_size
		downsample_factor = self.level_downsamples[downsample_level]
		downsample_x = int(base_level_x / downsample_factor)
		downsample_y = int(base_level_y / downsample_factor)

		image = self.get_downsampled_image(downsample_level)
		region = image.extract_area(downsample_x, downsample_y, extract_width, extract_height)
		return region

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
					export_folder=None, thumb_folder=None, silent=False, pb=None):
		self.load_error = False
		self.silent = silent
		if pb and not silent:
			self.print = pb.print
		elif silent:
			self.print = None
		else:
			self.print = print
		self.pb = pb
		self.p_id = None
		self.name = sfutil.path_to_name(path)
		self.shortname = sfutil._shortname(self.name)
		self.export_folder = export_folder
		self.size_px = size_px
		self.size_um = size_um
		self.tiles_path = None if not export_folder else join(self.export_folder, self.name)
		self.tile_mask = None
		self.enable_downsample = enable_downsample
		self.thumb_image = None
		self.thumb_folder = thumb_folder
		if not exists(thumb_folder): os.makedirs(thumb_folder)
		filetype = sfutil.path_to_ext(path)

		# Initiate supported slide (SVS, TIF) or JPG slide reader
		if filetype.lower() in sfutil.SUPPORTED_FORMATS:
			try:
				self.slide = OpenslideToVIPS(path) #ops.OpenSlide(path)
			except: #ops.lowlevel.OpenSlideUnsupportedFormatError:
				log.warn(f" Unable to read slide from {path} , skipping", 1, self.print)
				self.shape = None
				self.load_error = True
				return
		#elif filetype == "jpg":
		#	self.slide = JPGSlide(path, mpp=DEFAULT_JPG_MPP)
		else:
			log.error(f"Unsupported file type '{filetype}' for slide {self.name}.", 1, self.print)
			self.load_error = True
			return

		# Collect basic slide information
		try:
			self.MPP = float(self.slide.properties[OPS_MPP_X])#ops.PROPERTY_NAME_MPP_X])
		except KeyError:
			log.error(f"Corrupted SVS ({sfutil.green(self.name)}), skipping slide", 1, self.print)
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
	
	def thumb(self):
		'''Returns thumbnail of the slide.'''
		if self.thumb_folder:
			thumbs = os.listdir(self.thumb_folder)
			matching_thumbs = [tb for tb in thumbs if sfutil.path_to_name(tb) == self.name]
			if matching_thumbs:
				self.thumb_image = Image.open(join(self.thumb_folder, matching_thumbs[0]))
				return self.thumb_image
		if not self.thumb_image:
			print("Calculating thumbnail from slide...", end="")
			self.thumb_image = self.slide.get_thumbnail(2048, enable_downsample=self.enable_downsample)
			print("\r\033[K", end="")
			self.thumb_image.convert('RGB').save(join(self.thumb_folder, self.name+'.jpg'))
		return self.thumb_image

	def build_generator(self):
		log.label(self.shortname, f"Extracting {sfutil.bold(self.size_um)}um tiles (resizing {sfutil.bold(self.extract_px)}px -> {sfutil.bold(self.size_px)}px); stride: {sfutil.bold(int(self.stride))}px", 2, self.print)
		if self.size_px > self.extract_px:
			log.label(self.shortname, f"[{sfutil.fail('!WARN!')}] Tiles will be up-scaled with cubic interpolation, ({self.extract_px}px -> {self.size_px}px)", 2, self.print)
	
	def loaded_correctly(self):
		'''Returns True if slide loaded correctly without errors; False if otherwise.'''
		if self.load_error:
			return False
		try:
			loaded_correctly = bool(self.shape) 
		except:
			log.error(f"Slide failed to load properly for slide {sfutil.green(self.name)}", 1, self.print)
			sys.exit()
		return loaded_correctly
		
class TMAReader(SlideLoader):
	'''Helper object that loads a TMA-formatted slide, detects tissue cores, and sets up a tile generator.'''
	THUMB_DOWNSCALE = 100
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

	def __init__(self, path, size_px, size_um, stride_div, enable_downsample=False, export_folder=None, roi_dir=None, roi_list=None, pb=None):
		super().__init__(path, size_px, size_um, stride_div, enable_downsample, export_folder, pb)

		if not self.loaded_correctly():
			return

		self.annotations_dir = self.export_folder
		self.tiles_dir = self.export_folder
		self.DIM = self.slide.dimensions
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

		extracted = vips2numpy(self.slide.read_region((region_x_min, region_y_min), self.downsample_level, (region_width, region_height)))
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

	def build_generator(self, export=False, augment=False, export_full_core=False):
		'''Builds generator to supervise extraction of tiles across the slide.
		
		Args:
			export:				If true, will save tiles to the export_folder while extracting tiles.
			augment:			If true, will save flipped/rotated tiles while extracting.
			export_full_core:	If true, will also save a thumbnail of each fully extracted core.'''

		super().build_generator()

		log.empty(f"Extracting tiles from {sfutil.green(self.name)}, saving to {sfutil.green(self.tiles_dir)}", 1, self.print)
		img_orig = np.array(self.slide.get_thumbnail(self.DIM[0]/self.THUMB_DOWNSCALE, enable_downsample=self.enable_downsample))
		img_annotated = img_orig.copy()

		# Create background mask for edge detection
		white = np.array([255,255,255])
		buffer = 28
		mask = cv2.inRange(img_orig, np.array([0,0,0]), white-buffer)

		# Fill holes and dilate mask
		closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		dilating_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
		closing = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, closing_kernel)
		dilated = cv2.dilate(closing, dilating_kernel)

		# Use edge detection to find individual cores
		contours, heirarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

		# Filter out small regions that likely represent background noise
		# Also generate image showing identified cores
		box_areas = []
		object_rects = []
		num_filtered = 0
		for i, component in enumerate(zip(contours, heirarchy[0])):
			cnt = component[0]
			heir = component[1]
			rect = cv2.minAreaRect(cnt)
			width = rect[1][0]
			height = rect[1][1]
			if width > self.WIDTH_MIN and height > self.HEIGHT_MIN and heir[3] < 0:
				moment = cv2.moments(cnt)
				object_rects += [(len(object_rects), rect)]
				cX = int(moment["m10"] / moment["m00"])
				cY = int(moment["m01"] / moment["m00"])
				cv2.drawContours(img_annotated, contours, i, self.LIGHTBLUE)
				cv2.circle(img_annotated, (cX, cY), 4, self.GREEN, -1)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				area = polyArea([b[0] for b in box], [b[1] for b in box])
				box_areas += [area]
				cv2.drawContours(img_annotated, [box], 0, self.BLUE, 2)
				num_filtered += 1   
				#cv2.putText(img_annotated, f'{num_filtered}', (cX+10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.BLACK, 2)
			elif heir[3] < 0:
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(img_annotated, [box], 0, self.RED, 2)

		log.info(f"Number of detected cores: {num_filtered}", 2)

		# Write annotated image to file
		cv2.imwrite(join(self.annotations_dir, "annotated.jpg"), cv2.resize(img_annotated, (1400, 1000)))

		self.p_id = None if not self.pb else self.pb.add_bar(0, num_filtered, endtext=sfutil.green(self.shortname))
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

			for rect in object_rects:
				rectangle_queue.put(rect)
			rectangle_queue.put((-1, "DONE"))
			
			queue_progress = 0
			while True:
				queue_progress += 1
				tile_id, image_core = extraction_queue.get()
				if image_core == "DONE":
					break
				else:
					if self.pb:
						self.pb.update(self.p_id, queue_progress)
						self.pb.update_counter(1)

					sub_id = 0
					resized_core = self._resize_to_target(image_core)
					subtiles = self._split_core(resized_core)
					if export_full_core:
						cv2.imwrite(join(self.tiles_dir, f"tile{tile_id}.jpg"), image_core)
					for subtile in subtiles:
						sub_id += 1
						if export:
							cv2.imwrite(join(self.tiles_dir, f"tile{tile_id}_{sub_id}.jpg"), subtile)
						yield subtile#, tile_id, unique_tile
					
			extraction_pool.close()
					
			if self.pb: 
				self.pb.end(self.p_id)
			log.empty("Summary of extracted core areas (microns):", 1)
			log.info(f"Min: {min(box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2)
			log.info(f"Max: {max(box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2)
			log.info(f"Mean: {mean(box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2)
			log.info(f"Median: {median(box_areas) * self.THUMB_DOWNSCALE * self.MPP:.1f}", 2)

		return generator, None, None, None

	def export_tiles(self, augment=False, export_full_core=False):
		'''Exports all tiles.'''
		if not self.loaded_correctly():
			log.error(f"Unable to extract tiles; unable to load slide {sfutil.green(self.name)}", 1)
			return

		generator, _, _, _ = self.build_generator(export=True, augment=augment, export_full_core=export_full_core)

		if not generator:
			log.error(f"No tiles extracted from slide {sfutil.green(self.name)}", 1, self.print)
			return

		for tile in generator():
			pass

class SlideReader(SlideLoader):
	'''Helper object that loads a slide and its ROI annotations and sets up a tile generator.'''
	SKIP_MISSING_ROI = True
	def __init__(self, path, size_px, size_um, stride_div, enable_downsample=False, export_folder=None,
					roi_dir=None, roi_list=None, roi_method=EXTRACT_INSIDE, thumb_folder=None, silent=False, pb=None):

		super().__init__(path, size_px, size_um, stride_div, enable_downsample, export_folder, thumb_folder, silent, pb)

		if not self.loaded_correctly():
			return

		self.rois = []
		self.roi_method = roi_method
		# Look in roi_dir if available
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
				log.warn(f" Multiple matching ROIs found for {self.name}; using {matching_rois[0]}", 1)
			self.load_csv_roi(matching_rois[0])
		else:
			if self.SKIP_MISSING_ROI:
				log.error(f"No ROI found for {sfutil.green(self.name)}, skipping slide", 1, self.print)
				self.shape = None
				self.load_error = True
				return
			else:
				log.warn(f"[{sfutil.green(self.shortname)}]  No ROI found in {roi_dir}, using whole slide.", 2, self.print)

		log.label(self.shortname, f"Slide info: {self.MPP} um/px | {len(self.rois)} ROI(s) | Size: {self.full_shape[0]} x {self.full_shape[1]}", 2, self.print)

		# Abort if errors were raised during ROI loading
		if self.load_error:
			log.error(f'Skipping slide {sfutil.green(self.name)} due to slide image or ROI loading error', 1, self.print)
			return

	def build_generator(self, export=False, augment=False, dual_extract=False, whole_slide=False):
		'''Builds generator to supervise extraction of tiles across the slide.
		
		Args:
			export:				If true, will save tiles to the export_folder while extracting tiles.
			augment:			If true, will save flipped/rotated tiles while extracting.
			dual_extract:		If true, will extract base image and the surrounding region.'''
		super().build_generator()
		# Calculate window sizes, strides, and coordinates for windows
		coord = []
		slide_x_size = self.full_shape[0] - self.full_extract_px
		slide_y_size = self.full_shape[1] - self.full_extract_px

		# Coordinates must be in level 0 (full) format for the read_region function
		for y in np.arange(0, (self.full_shape[1]+1) - self.full_extract_px, self.full_stride):
			for x in np.arange(0, (self.full_shape[0]+1) - self.full_extract_px, self.full_stride):
				y = int(y)
				x = int(x)
				is_unique = ((y % self.full_extract_px == 0) and (x % self.full_extract_px == 0))
				coord.append([x, y, is_unique])

		# Load annotations as shapely.geometry objects
		ROI_SCALE = 10
		if whole_slide:
			annPolys = []
		else:
			annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.rois]
		roi_area = sum([poly.area for poly in annPolys])
		total_area = (self.full_shape[0]/ROI_SCALE) * (self.full_shape[1]/ROI_SCALE)
		roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

		total_logits_count = int(len(coord) * roi_area_fraction) if self.roi_method == EXTRACT_INSIDE else int(len(coord) * (1-roi_area_fraction))
		if total_logits_count == 0:
			log.warn(f"No tiles were able to be extracted at the given micron size for slide {sfutil.green(self.name)}", 1, self.print)
			return None, None, None, None
		# Create mask for indicating whether tile was extracted
		tile_mask = np.asarray([0 for i in range(len(coord))])
		self.p_id = None if not self.pb else self.pb.add_bar(0, total_logits_count, endtext='' if self.silent else sfutil.green(self.shortname))

		def generator():
			tile_counter=0
			if export and not os.path.exists(self.tiles_path): os.makedirs(self.tiles_path)
			for ci in range(len(coord)):
				c = coord[ci]

				# Check if the center of the current window lies within any annotation; if not, skip
				x_coord = int((c[0]+self.full_extract_px/2)/ROI_SCALE)
				y_coord = int((c[1]+self.full_extract_px/2)/ROI_SCALE)
				# If the extraction method is EXTRACT_INSIDE, skip the tile if it's not in an ROI
				if bool(annPolys) and (self.roi_method == EXTRACT_INSIDE) and not any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in annPolys]):
					continue
				# If the extraction method is EXTRACT_OUTSIDE, skip the tile if it's in an ROI
				elif bool(annPolys) and (self.roi_method == EXTRACT_OUTSIDE) and any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in annPolys]):
					continue
				tile_counter += 1
				if self.pb:
					self.pb.update(self.p_id, tile_counter)

				# Read the low-magnification level for filtering out background
				if self.enable_downsample:
					filter_region = vips2numpy(self.slide.read_region((c[0], c[1]), self.slide.level_count-1, [self.filter_px, self.filter_px]))[:,:,:-1]
				else:
					filter_region = vips2numpy(self.slide.read_region((c[0], c[1]), 0, [self.extract_px, self.extract_px]))[:,:,:-1]
				median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
				if median_brightness > 660:
					# Discard tile; median brightness (average RGB pixel) > 220
					continue

				if self.pb:
					self.pb.update_counter(1)

				# Read the region and resize to target size
				region = self.slide.read_region((c[0], c[1]), self.downsample_level, [self.extract_px, self.extract_px])
				region = region.resize(float(self.size_px) / self.extract_px)

				if dual_extract:
					try:
						surrounding_region = self.slide.read_region((c[0]-self.full_stride, c[1]-self.full_stride), self.downsample_level, [self.extract_px*3, self.extract_px*3])
						surrounding_region = surrounding_region.resize(float(self.size_px) / (self.extract_px*3))
					except:
						continue

				tile_mask[ci] = 1
				coord_label = ci
				unique_tile = c[2]
				if export and unique_tile:
					region.jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}.jpg'), Q=100)
					if augment:
						region.rot90().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug1.jpg'))
						region.flipver().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug2.jpg'))
						region.rot90().flipver().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug3.jpg'))
						region.fliphor().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug4.jpg'))
						region.rot90().fliphor().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug5.jpg'))
						region.fliphor().flipver().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug6.jpg'))
						region.rot90().fliphor().flipver().jpegsave(join(self.tiles_path, f'{self.shortname}_{ci}_aug7.jpg'))
				if dual_extract:
					yield {"input_1": vips2numpy(region)[:,:,:-1], "input_2": vips2numpy(surrounding_region)[:,:,:-1]}
				else:
					pil_region = Image.fromarray(vips2numpy(region)).convert('RGB')
					yield pil_region

			if self.pb: 
				self.pb.end(self.p_id)
				log.complete(f"Finished tile extraction for {sfutil.green(self.shortname)} ({sum(tile_mask)} tiles of {len(coord)} possible)", 1, self.print)
			self.tile_mask = tile_mask

		return generator, slide_x_size, slide_y_size, self.full_stride

	def export_tiles(self, augment=False):
		'''Exports tiles.'''
		if not self.loaded_correctly():
			log.error(f"Unable to extract tiles; unable to load slide {sfutil.green(self.name)}", 1)
			return

		generator, _, _, _ = self.build_generator(export=True, augment=augment)

		if not generator:
			log.error(f"No tiles extracted from slide {sfutil.green(self.name)}", 1, self.print)
			return

		for tile in generator():
			pass

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
				log.error(f'Unable to read CSV ROI file {sfutil.green(path)}, please check file integrity and ensure headers contain "ROI_name", "X_base", and "Y_base".', 1, self.print)
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

		return len(self.rois)

	def load_json_roi(self, path):
		'''Loads ROI from a JSON file.'''
		JSON_ANNOTATION_SCALE = 10
		with open(path, "r") as json_file:
			json_data = json.load(json_file)['shapes']
		for shape in json_data:
			area_reduced = np.multiply(shape['points'], JSON_ANNOTATION_SCALE)
			self.rois.append(ROIObject(f"Object{len(self.rois)}"))
			self.rois[-1].add_shape(area_reduced)
		return len(self.rois)