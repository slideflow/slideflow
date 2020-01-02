import cv2
import numpy as np
import openslide as ops
import math
import sys
import matplotlib.pylab as plt
import slideflow.util as sfutil

from os.path import join
from statistics import mean, median
from slideflow.util import log

def PolyArea(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class TMA:
	DOWNSCALE = 100
	WIDTH_MIN = 20
	HEIGHT_MIN = 20
	BLACK = (0,0,0)
	GREEN = (75,220,75)
	BLUE = (255, 100, 100)
	LIGHTBLUE = (255, 180, 180)
	RED = (100, 100, 200)

	def __init__(self, tma, save_dir):
		log.empty(f"Loading TMA at {sfutil.green(tma)}", 1)

		self.slide = ops.OpenSlide(tma)
		self.save_dir = save_dir
		self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
		self.DIM = self.slide.dimensions

		log.info(f"Full slide microns-per-pixel: {self.MPP}", 2)
		log.info(f"Full slide dimensions: {self.DIM}", 2)

	def extract_tiles(self):
		tiles_directory = join(self.save_dir, "tiles")
		log.empty(f"Extracting tiles from TMA, saving to {sfutil.green(tiles_directory)}", 1)
		img_orig = np.array(self.slide.get_thumbnail((self.DIM[0]/self.DOWNSCALE, self.DIM[1]/self.DOWNSCALE)))
		img_annotated = img_orig.copy()

		white = np.array([255,255,255])
		buffer = 28
		mask = cv2.inRange(img_orig, np.array([0,0,0]), white-buffer)

		closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		dilating_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
		closing = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, closing_kernel)
		dilated = cv2.dilate(closing, dilating_kernel)

		contours, heirarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

		box_areas = []
		num_filtered = 0
		for i, component in enumerate(zip(contours, heirarchy[0])):
			cnt = component[0]
			heir = component[1]
			rect = cv2.minAreaRect(cnt)
			width = rect[1][0]
			height = rect[1][1]
			if width > self.WIDTH_MIN and height > self.HEIGHT_MIN and heir[3] < 0:
				moment = cv2.moments(cnt)
				cX = int(moment["m10"] / moment["m00"])
				cY = int(moment["m01"] / moment["m00"])
				cv2.drawContours(img_annotated, contours, i, self.LIGHTBLUE)
				cv2.circle(img_annotated, (cX, cY), 4, self.GREEN, -1)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				area = PolyArea([b[0] for b in box], [b[1] for b in box])
				box_areas += [area]
				cv2.drawContours(img_annotated, [box], 0, self.BLUE, 2)
				num_filtered += 1   
				cv2.putText(img_annotated, f'{num_filtered}', (cX+10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.BLACK, 2)
			elif heir[3] < 0:
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(img_annotated, [box], 0, self.RED, 2)

		log.info(f"Number of detected regions: {len(contours)}", 2)
		log.info(f"Number of regions after filtering: {num_filtered}", 2)

		cv2.imwrite(join(self.save_dir, "annotated.jpg"), cv2.resize(img_annotated, (1400, 1000)))

		tile_id = 0
		for i, component in enumerate(zip(contours, heirarchy[0])):
			cnt = component[0]
			heir = component[1]
			rect = cv2.minAreaRect(cnt)
			width = rect[1][0]
			height = rect[1][1]
			if width > self.WIDTH_MIN and height > self.HEIGHT_MIN and heir[3] < 0:
				print(f"Working on tile {tile_id+1} of {num_filtered}...", end="\033[K\r")
				img_crop = self.getSubImage(rect)
				tile_id += 1
				cv2.imwrite(join(tiles_directory, f"tile{tile_id}.jpg"), img_crop)

		log.empty("Summary of box areas:", 1)
		log.info(f"Min: {min(box_areas) * self.DOWNSCALE * self.MPP:.1f}", 2)
		log.info(f"Max: {max(box_areas) * self.DOWNSCALE * self.MPP:.1f}", 2)
		log.info(f"Mean: {mean(box_areas) * self.DOWNSCALE * self.MPP:.1f}", 2)
		log.info(f"Median: {median(box_areas) * self.DOWNSCALE * self.MPP:.1f}", 2)

	def getSubImage(self, rect):
		box = cv2.boxPoints(rect) * self.DOWNSCALE
		box = np.int0(box)
		rect_width = int(rect[1][0] * self.DOWNSCALE)
		rect_height = int(rect[1][1] * self.DOWNSCALE)

		region_x_min = int(min([b[0] for b in box]))
		region_x_max = max([b[0] for b in box])
		region_y_min = int(min([b[1] for b in box]))
		region_y_max = max([b[1] for b in box])

		region_width = region_x_max - region_x_min
		region_height = region_y_max - region_y_min

		extracted = self.slide.read_region((region_x_min, region_y_min), 0, (region_width, region_height))
		relative_box = box - [region_x_min, region_y_min]

		src_pts = relative_box.astype("float32")
		dst_pts = np.array([[0, rect_height-1],
							[0, 0],
							[rect_width-1, 0],
							[rect_width-1, rect_height-1]], dtype="float32")
		P = cv2.getPerspectiveTransform(src_pts, dst_pts)
		warped=cv2.warpPerspective(np.array(extracted), P, (rect_width, rect_height))
		return warped

if __name__ == '__main__':
	sampleTMA = TMA("/home/shawarma/data/TMA/TMA_1185.svs", "/home/shawarma/data/TMA")
	sampleTMA.extract_tiles()