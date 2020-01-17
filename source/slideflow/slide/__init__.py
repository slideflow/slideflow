# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''This module includes tools to convolutionally section whole slide images into tiles
using python Generators. These tessellated tiles can be exported as JPGs, with or without
data augmentation, or used as input for a trained Tensorflow model. Model predictions 
can then be visualized as a heatmap overlay.

This module is compatible with SVS and JPG images.

Requires: Openslide (https://openslide.org/download/).'''

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
import openslide as ops
import shapely.geometry as sg
import cv2
import json
import time
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
import slideflow.util as sfutil
import slideflow.util.datasets as sfdatasets

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
# TODO: test removing BatchNorm fix
# TODO: remove final layer activations functions (duplicated in a separate module)

# For TMA reader:
# TODO: consolidate slide "thumbs" and the TMA "get_thumbnail"

Image.MAX_IMAGE_PIXELS = 100000000000
NUM_THREADS = 4
DEFAULT_JPG_MPP = 0.5
SKIP_MISSING_ROI = True

# BatchNormFix
tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization

def polyArea(x, y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

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

class JPGSlide:
	'''Object that provides cross-compatibility with certain OpenSlide methods when using JPG slides.'''
	def __init__(self, path, mpp):
		self.loaded_image = Image.open(path)
		self.dimensions = (self.loaded_image.size[1], self.loaded_image.size[0])
		# ImageIO version
		#self.loaded_image = imageio.imread(path)
		#self.dimensions = (self.loaded_image.shape[1], self.loaded_image.shape[0])
		self.properties = {ops.PROPERTY_NAME_MPP_X: mpp}
		self.level_dimensions = [self.dimensions]
		self.level_count = 1
		self.level_downsamples = [1.0]

	def get_thumbnail(self, dimensions):
		width = self.dimensions[1]
		height = self.dimensions[0]
		return self.loaded_image.resize([width, height], resample=Image.BICUBIC)
		# ImageIO version
		#return cv2.resize(self.loaded_image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

	def read_region(self, topleft, level, window):
		# Arg "level" required for code compatibility with slide reader but is not used
		# Window = [y, x] pixels (note: this is reverse compared to slide/SVS files in [x,y] format)

		return self.loaded_image.crop([topleft[0], topleft[1], topleft[0]+window[0], topleft[1]+window[1]])

		# ImageIO version
		#return self.loaded_image[topleft[1]:topleft[1] + window[1], 
		#						 topleft[0]:topleft[0] + window[0],]

	def get_best_level_for_downsample(self, downsample_desired):
		return 0

class SlideLoader:
	'''Object that loads an SVS slide and makes preparations for tile extraction.
	Should not be used directly; this class must be inherited and extended by a child class!'''
	def __init__(self, path, size_px, size_um, stride_div, export_folder=None, pb=None):
		self.load_error = False
		self.print = print if not pb else pb.print
		self.pb = pb
		self.p_id = None
		self.name = sfutil.path_to_name(path)
		self.shortname = sfutil._shortname(self.name)
		self.export_folder = export_folder
		self.size_px = size_px
		self.size_um = size_um
		self.tiles_path = None if not export_folder else join(self.export_folder, self.name)
		self.tile_mask = None
		filetype = sfutil.path_to_ext(path)

		# Initiate supported slide (SVS, TIF) or JPG slide reader
		if filetype.lower() in sfutil.SUPPORTED_FORMATS:
			try:
				self.slide = ops.OpenSlide(path)
			except ops.lowlevel.OpenSlideUnsupportedFormatError:
				log.warn(f" Unable to read slide from {path} , skipping", 1, self.print)
				self.shape = None
				self.load_error = True
				return
		elif filetype == "jpg":
			self.slide = JPGSlide(path, mpp=DEFAULT_JPG_MPP)
		else:
			log.error(f"Unsupported file type '{filetype}' for slide {self.name}.", 1, self.print)
			self.load_error = True
			return

		# Collect basic slide information
		try:
			self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
		except KeyError:
			log.error(f"Corrupted SVS ({sfutil.green(self.name)}), skipping slide", 1, self.print)
			self.load_error = True
			return
		self.full_shape = self.slide.dimensions
		self.full_extract_px = int(self.size_um / self.MPP)

		# Load downsampled level based on desired extraction size
		downsample_desired = self.full_extract_px / size_px
		self.downsample_level = self.slide.get_best_level_for_downsample(downsample_desired)
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

		# Generating thumbnail for heatmap
		self.thumbs_path = join(Path(path).parent, "thumbs")
		sfdatasets.make_dir(self.thumbs_path)
		goal_thumb_area = 4096*4096
		y_x_ratio = self.shape[1] / self.shape[0]
		thumb_x = sqrt(goal_thumb_area / y_x_ratio)
		thumb_y = thumb_x * y_x_ratio
		self.thumb = self.slide.get_thumbnail((int(thumb_x), int(thumb_y)))

	def build_generator(self):
		log.label(self.shortname, f"Extracting {sfutil.bold(self.size_um)}um tiles (resizing {sfutil.bold(self.extract_px)}px -> {sfutil.bold(self.size_px)}px); stride: {sfutil.bold(int(self.stride))}px", 2, self.print)
		if self.size_px > self.extract_px:
			log.label(self.shortname, f"[{sfutil.fail('!WARN!')}] Tiles will be up-scaled with cubic interpolation, ({self.extract_px}px -> {self.size_px}px)", 2, self.print)
	
	def loaded_correctly(self):
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

	def __init__(self, path, size_px, size_um, stride_div, export_folder=None, roi_dir=None, roi_list=None, pb=None):
		super().__init__(path, size_px, size_um, stride_div, export_folder, pb)

		if not self.loaded_correctly():
			return

		self.annotations_dir = self.export_folder
		self.tiles_dir = self.thumbs_path
		self.DIM = self.slide.dimensions
		log.label(self.shortname, f"Slide info: {self.MPP} um/px | Size: {self.full_shape[0]} x {self.full_shape[1]}", 2, self.print)

	def build_generator(self, export=False, augment=False, export_full_core=False):
		super().build_generator()

		log.empty(f"Extracting tiles from {sfutil.green(self.name)}, saving to {sfutil.green(self.tiles_dir)}", 1, self.print)
		img_orig = np.array(self.slide.get_thumbnail((self.DIM[0]/self.THUMB_DOWNSCALE, self.DIM[1]/self.THUMB_DOWNSCALE)))
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
					image_tile = self.get_sub_image(rect)
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
					resized_core = self.resize_to_target(image_core)
					subtiles = self.split_core(resized_core)
					if export_full_core:
						cv2.imwrite(join(self.tiles_dir, f"tile{tile_id}.jpg"), image_core)
					for subtile in subtiles:
						sub_id += 1
						if export:
							cv2.imwrite(join(self.tiles_dir, f"tile{tile_id}_{sub_id}.jpg"), subtile)
						yield subtile, tile_id, unique_tile
					
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
		if not self.loaded_correctly():
			log.error(f"Unable to extract tiles; unable to load slide {sfutil.green(self.name)}", 1)
			return

		generator, _, _, _ = self.build_generator(export=True, augment=augment, export_full_core=export_full_core)

		if not generator:
			log.error(f"No tiles extracted from slide {sfutil.green(self.name)}", 1, self.print)
			return

		for tile, tile_id, _ in generator():
			pass

	def resize_to_target(self, image_tile):
		target_MPP = self.size_um / self.size_px
		current_MPP = self.MPP * self.downsample_factor
		resize_factor = current_MPP / target_MPP
		return cv2.resize(image_tile, (0, 0), fx=resize_factor, fy=resize_factor)

	def split_core(self, image):
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

	def get_sub_image(self, rect):
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

		extracted = self.slide.read_region((region_x_min, region_y_min), self.downsample_level, (region_width, region_height))
		relative_box = (box - [region_x_min, region_y_min]) / self.downsample_factor

		src_pts = relative_box.astype("float32")
		dst_pts = np.array([[0, (rect_height)-1],
							[0, 0],
							[(rect_width)-1, 0],
							[(rect_width)-1, (rect_height)-1]], dtype="float32")

		P = cv2.getPerspectiveTransform(src_pts, dst_pts)
		warped=cv2.warpPerspective(np.array(extracted), P, (rect_width, rect_height))
		return warped

class SlideReader(SlideLoader):
	'''Helper object that loads a slide and its ROI annotations and sets up a tile generator.'''
	def __init__(self, path, size_px, size_um, stride_div, export_folder=None, roi_dir=None, roi_list=None, pb=None):
		super().__init__(path, size_px, size_um, stride_div, export_folder, pb)

		if not self.loaded_correctly():
			return

		self.rois = []
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
				log.warn(f" Multiple matching ROIs found for {self.name}; using {matching_rois[0]}")
			self.load_csv_roi(matching_rois[0])
		else:
			if SKIP_MISSING_ROI:
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

	def build_generator(self, export=False, augment=False):
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
		annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.rois]
		roi_area = sum([poly.area for poly in annPolys])
		total_area = (self.full_shape[0]/ROI_SCALE) * (self.full_shape[1]/ROI_SCALE)
		roi_area_fraction = 1 if not roi_area else (roi_area / total_area)

		total_logits_count = int(len(coord) * roi_area_fraction)
		if total_logits_count == 0:
			log.warn(f"No tiles were able to be extracted at the given micron size for slide {sfutil.green(self.name)}", 1, self.print)
			return None, None, None, None
		# Create mask for indicating whether tile was extracted
		tile_mask = np.asarray([0 for i in range(len(coord))])
		self.p_id = None if not self.pb else self.pb.add_bar(0, total_logits_count, endtext=sfutil.green(self.shortname))

		def generator():
			tile_counter=0
			if export and not os.path.exists(self.tiles_path): os.makedirs(self.tiles_path)
			for ci in range(len(coord)):
				c = coord[ci]
				# Check if the center of the current window lies within any annotation; if not, skip
				x_coord = int((c[0]+self.full_extract_px/2)/ROI_SCALE)
				y_coord = int((c[1]+self.full_extract_px/2)/ROI_SCALE)
				if bool(annPolys) and not any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in annPolys]):
					continue
				tile_counter += 1
				if self.pb:
					self.pb.update(self.p_id, tile_counter)
				# Read the low-mag level for filter checking
				filter_region = np.asarray(self.slide.read_region(c, self.slide.level_count-1, [self.filter_px, self.filter_px]))[:,:,:-1]
				median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
				if median_brightness > 660:
					# Discard tile; median brightness (average RGB pixel) > 220
					continue
				if self.pb:
					self.pb.update_counter(1)
				# Read the region and discard the alpha pixels
				region = self.slide.read_region(c, self.downsample_level, [self.extract_px, self.extract_px])
				region = region.resize((self.size_px, self.size_px))
				region = region.convert('RGB')
				tile_mask[ci] = 1
				coord_label = ci
				unique_tile = c[2]
				if export and unique_tile:
					region.save(join(self.tiles_path, f'{self.shortname}_{ci}.jpg'), "JPEG")
					if augment:
						region.transpose(Image.ROTATE_90).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug1.jpg'))
						region.transpose(Image.FLIP_TOP_BOTTOM).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug2.jpg'))
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug3.jpg'))
						region.transpose(Image.FLIP_LEFT_RIGHT).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug4.jpg'))
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug5.jpg'))
						region.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug6.jpg'))
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(join(self.tiles_path, f'{self.shortname}_{ci}_aug7.jpg'))
				yield region, coord_label, unique_tile

			if self.pb: 
				self.pb.end(self.p_id)
				log.complete(f"Finished tile extraction for {sfutil.green(self.shortname)} ({sum(tile_mask)} tiles of {len(coord)} possible)", 1, self.print)
			self.tile_mask = tile_mask

		return generator, slide_x_size, slide_y_size, self.full_stride

	def export_tiles(self, augment=False):
		if not self.loaded_correctly():
			log.error(f"Unable to extract tiles; unable to load slide {sfutil.green(self.name)}", 1)
			return

		generator, _, _, _ = self.build_generator(export=True, augment=augment)

		if not generator:
			log.error(f"No tiles extracted from slide {sfutil.green(self.name)}", 1, self.print)
			return

		for tile, _, _ in generator():
			pass

	def load_csv_roi(self, path):
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
		JSON_ANNOTATION_SCALE = 10
		with open(path, "r") as json_file:
			json_data = json.load(json_file)['shapes']
		for shape in json_data:
			area_reduced = np.multiply(shape['points'], JSON_ANNOTATION_SCALE)
			self.rois.append(ROIObject(f"Object{len(self.rois)}"))
			self.rois[-1].add_shape(area_reduced)
		return len(self.rois)

class Heatmap:
	'''Generates heatmap by calculating predictions from a sliding scale window across a slide. May also export final layer
	activations as model predictions are generated.'''

	def __init__(self, slide_path, model_path, size_px, size_um, use_fp16, stride_div=2, save_folder='', roi_dir=None, roi_list=None):
		self.save_folder = save_folder
		self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.DTYPE_INT = tf.int16 if use_fp16 else tf.int32
		self.MODEL_DIR = None
		self.logits = None

		# Load the slide
		self.slide = SlideReader(slide_path, size_px, size_um, stride_div, save_folder, roi_dir, roi_list)

		# Build the model
		self.MODEL_DIR = model_path

		# First, load the designated model
		_model = tf.keras.models.load_model(self.MODEL_DIR)

		# Now, construct a new model that outputs both predictions and final layer activations
		self.model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
										   outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])

		# Record the number of classes in the model
		self.NUM_CLASSES = _model.layers[-1].output_shape[-1]

		if not self.slide.loaded_correctly():
			log.error(f"Unable to load slide {self.slide.name} for heatmap generation", 1)
			return

	def _parse_function(self, image, label, mask):
		parsed_image = tf.image.per_image_standardization(image)
		parsed_image = tf.image.convert_image_dtype(parsed_image, self.DTYPE)
		return parsed_image, label, mask

	def calculate_logits(self, batch_size, activations=False):
		'''Convolutes across a whole slide, returning logits and final layer activations for tessellated image tiles'''

		# Create tile coordinate generator
		gen_slice, x_size, y_size, stride_px = self.slide.build_generator(export=False)

		if not gen_slice:
			log.error(f"No tiles extracted from slide {sfutil.green(self.slide.name)}", 1)
			return False, False, False, False

		# Generate dataset from the generator
		with tf.name_scope('dataset_input'):
			tile_dataset = tf.data.Dataset.from_generator(gen_slice, (tf.uint8, tf.int64, tf.bool))
			tile_dataset = tile_dataset.map(self._parse_function, num_parallel_calls=8)
			tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)

		logits_arr = []
		labels_arr = []

		prelogits_arr = [] # Final layer activations 
		logits_arr = []	# Logits (predictions) 
		unique_arr = []	# Boolean array indicating whether tile is unique (non-overlapping) 

		# Iterate through generator to calculate logits +/- final layer activations for all tiles
		for batch_images, batch_labels, batch_unique in tile_dataset:
			prelogits, logits = self.model.predict([batch_images, batch_images])
			batch_labels = batch_labels.numpy()
			batch_unique = batch_unique.numpy()
			prelogits_arr = prelogits if prelogits_arr == [] else np.concatenate([prelogits_arr, prelogits])
			logits_arr = logits if logits_arr == [] else np.concatenate([logits_arr, logits])
			labels_arr = batch_labels if labels_arr == [] else np.concatenate([labels_arr, batch_labels])
			unique_arr = batch_unique if unique_arr == [] else np.concatenate([unique_arr, batch_unique])

		# Sort the output (may be shuffled due to multithreading)
		try:
			sorted_indices = labels_arr.argsort()
		except AttributeError:
			# This occurs when the list is empty, likely due to an empty annotation area
			raise AttributeError("No tile calculations performed for this image, are you sure the annotation area isn't empty?")
		logits_arr = logits_arr[sorted_indices]
		labels_arr = labels_arr[sorted_indices]
		
		# Perform same functions on final layer activations
		flat_unique_logits = None
		if activations:
			prelogits_arr = prelogits_arr[sorted_indices]
			unique_arr = unique_arr[sorted_indices]
			# Find logits from non-overlapping tiles (will be used for metadata for saved final layer activations CSV)
			flat_unique_logits = [logits_arr[l] for l in range(len(logits_arr)) if unique_arr[l]]
			prelogits_out = [prelogits_arr[p] for p in range(len(prelogits_arr)) if unique_arr[p]]
			prelogits_labels = [labels_arr[l] for l in range(len(labels_arr)) if unique_arr[l]]
		else:
			prelogits_out = None
			prelogits_labels = None

		if self.slide.tile_mask is not None and x_size and y_size and stride_px:
			# Expand logits back to a full 2D map spanning the whole slide,
			#  supplying values of "0" where tiles were skipped by the tile generator
			x_logits_len = int(x_size / stride_px) + 1
			y_logits_len = int(y_size / stride_px) + 1
			expanded_logits = [[0] * self.NUM_CLASSES] * len(self.slide.tile_mask)
			li = 0
			for i in range(len(expanded_logits)):
				if self.slide.tile_mask[i] == 1:
					expanded_logits[i] = logits_arr[li]
					li += 1
			try:
				expanded_logits = np.asarray(expanded_logits, dtype=float)
			except ValueError:
				log.error("Mismatch with number of categories in model output and expected number of categories", 1)

			# Resize logits array into a two-dimensional array for heatmap display
			logits_out = np.resize(expanded_logits, [y_logits_len, x_logits_len, self.NUM_CLASSES])
		else:
			logits_out = logits_arr

		return logits_out, prelogits_out, prelogits_labels, flat_unique_logits

	def generate(self, batch_size=16, export_activations=False):
		# Calculate the final layer activations and logits/predictions
		self.logits, activations, activations_labels, logits_flat = self.calculate_logits(batch_size=batch_size, activations=export_activations)
		if (type(self.logits) == bool) and (not self.logits):
			log.error(f"Unable to create heatmap for slide {sfutil.green(self.slide.name)}", 1)
			return

		# Export final layer activations if requested
		if export_activations:
			log.empty("Writing csv...", 1)
			csv_started = os.path.exists(join(self.save_folder, 'heatmap_layer_activations.csv'))
			write_mode = 'a' if csv_started else 'w'
			with open(join(self.save_folder, 'heatmap_layer_activations.csv'), write_mode) as csv_file:
				csv_writer = csv.writer(csv_file, delimiter = ',')
				if not csv_started:
					csv_writer.writerow(["Tile_num", "Slide", "Category"] + [f"Logits{l}" for l in range(len(logits_flat[0]))] + [f"Node{n}" for n in range(len(activations[0]))])
				for l in range(len(activations)):
					logit = logits_flat[l].tolist()
					out = activations[l].tolist()
					csv_writer.writerow([activations_labels[l], self.slide.name] + logit + out)

	def prepare_figure(self):
		self.fig = plt.figure(figsize=(18, 16))
		self.ax = self.fig.add_subplot(111)
		self.fig.subplots_adjust(bottom = 0.25, top=0.95)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
		jetMap = np.linspace(0.45, 0.95, 255)
		cmMap = cm.nipy_spectral(jetMap)
		self.newMap = mcol.ListedColormap(cmMap)		

	def display(self):
		self.prepare_figure()
		heatmap_dict = {}
		implot = FastImshow(self.slide.thumb, self.ax, extent=None, tgt_res=1024)

		def slider_func(val):
			for h, s in heatmap_dict.values():
				h.set_alpha(s.val)

		for i in range(self.NUM_CLASSES):
			heatmap = self.ax.imshow(self.logits[:, :, i], extent=implot.extent, cmap=self.newMap, alpha = 0.0, interpolation='none', zorder=10) #bicubic
			ax_slider = self.fig.add_axes([0.25, 0.2-(0.2/self.NUM_CLASSES)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
			slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
			heatmap_dict.update({f"Class{i}": [heatmap, slider]})
			slider.on_changed(slider_func)

		self.fig.canvas.set_window_title(self.slide.name)
		implot.show()
		plt.show()

	def save(self):
		'''Displays and/or saves logits as a heatmap overlay.'''
		self.prepare_figure()
		heatmap_dict = {}
		implot = self.ax.imshow(self.slide.thumb, zorder=0)

		# Make heatmaps and sliders
		for i in range(self.NUM_CLASSES):
			heatmap = self.ax.imshow(self.logits[:, :, i], extent=implot.get_extent(), cmap=self.newMap, alpha = 0.0, interpolation='none', zorder=10) #bicubic
			heatmap_dict.update({i: heatmap})

		plt.savefig(os.path.join(self.save_folder, f'{self.slide.name}-raw.png'), bbox_inches='tight')
		for i in range(self.NUM_CLASSES):
			heatmap_dict[i].set_alpha(0.6)
			plt.savefig(os.path.join(self.save_folder, f'{self.slide.name}-{i}.png'), bbox_inches='tight')
			heatmap_dict[i].set_alpha(0.0)
		plt.close()
		log.complete(f"Saved heatmaps for {sfutil.green(self.slide.name)}", 1)