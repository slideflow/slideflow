import shutil

from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
import sys
from random import shuffle
import argparse
import slideflow.util as sfutil
from glob import glob
from slideflow.util import log

def make_dir(_dir):
	''' Makes a directory if one does not already exist, in a manner compatible with multithreading. '''
	if not exists(_dir):
		try:
			makedirs(_dir, exist_ok=True)
		except FileExistsError:
			pass

def split_tiles(folder, fraction, names):
	'''Split a directory of .jpg files into subdirectories.

	Args:
		folder 		folder to search for tiles
		fraction	array containing fraction of tiles to include in each subdirectory;
						remaining tiles will be split among subdirectories with fraction of -1
		names		names of subfolder to split tiles. Must be same length as fraction
	'''

	# Initial error checking
	if len(fraction) != len(names):
		log.error(f'When splitting tiles, length of "fraction" ({len(fraction)}) should equal length of "names" ({len(names)})')
		sys.exit()
	if sum([i for i in fraction if i != -1]) > 1:
		log.error(f'Unable to split tiles; Sum of fraction is greater than 1')
		sys.exit()

	# Setup directories
	slides = [_dir for _dir in listdir(folder) if isdir(join(folder, _dir))]
	num_moved = [0] * len(names)

	for slide in slides:
		slide_directory = join(folder, slide)
		slide_files = [f for f in listdir(slide_directory) 
						if (isfile(join(slide_directory, f))) and
						(sfutil.path_to_ext(f) == 'jpg')]

		shuffle(slide_files)
		num_files = len(slide_files)
		num_to_move = [0] * len(fraction)

		# First, calculate number to move for the explicitly specified fractions
		for fr in range(len(fraction)):
			if fraction[fr] != -1:
				num_leftover = num_files - sum(num_to_move)
				num_to_move[fr] = min(int(fraction[fr] * num_files), num_leftover)

		# Now, split up leftover into the other categories
		num_fr_dynamic = len([i for i in fraction if i == -1])
		if num_fr_dynamic != 0:
			num_per_dynamic = int((num_files - sum(num_to_move)) / num_fr_dynamic)
			for fr in range(len(fraction)):
				if fraction[fr] == -1:
					num_leftover = num_files - sum(num_to_move)
					num_to_move[fr] = min(num_per_dynamic, num_leftover)

		# Error checking
		if sum(num_to_move) > num_files:
			log.error(f"Error with separating tiles; tried to move {sum(num_to_move)} tiles into {len(fraction)} subfolders, only {num_files} tiles available", 1)
			sys.exit()
		if sum(num_to_move) < num_files:
			log.warn(f"Not all tiles separated into subfolders; {num_files - sum(num_to_move)} leftover tiles will be discarded.", 1)

		# Split tiles by subfolder
		for n, name in enumerate(names):
			slide_subfolder_directory = join(folder, name, slide)
			make_dir(slide_subfolder_directory)

			num = num_to_move[n]
			files_to_move = slide_files[0:num]
			slide_files = slide_files[num:]

			for f in files_to_move:
				shutil.move(join(slide_directory, f), join(slide_subfolder_directory, f))
			num_moved[n] += num
			log.empty(f"Moved {num} tiles for slide {sfutil.green(slide)} into subfolder {name}", 1)

		# Remove the empty directory
		shutil.rmtree(slide_directory)

	# Print results
	for n, name in enumerate(names):
		log.complete(f"Moved {num_moved[n]} tiles into subfolder {name}", 1)
			
def build_validation(train_dir, eval_dir, fraction = 0.1):
	total_moved = 0
	make_dir(eval_dir)
	slide_dirs = [_dir for _dir in listdir(train_dir) if isdir(join(train_dir, _dir))]
	for slide_dir in slide_dirs:
		make_dir(join(eval_dir, slide_dir))
		files = [_file for _file in listdir(join(train_dir, slide_dir)) 
					if (isfile(join(train_dir, slide_dir, _file))) and
						(sfutil.path_to_ext(_file) == "jpg")]
		shuffle(files)
		num_to_move = int(len(files)*fraction)
		total_moved += num_to_move
		for file in files[0:num_to_move]:
			shutil.move(join(train_dir, slide_dir, file), join(eval_dir, slide_dir, file))
		log.empty(f"Set aside {num_to_move} tiles for slide {sfutil.green(slide_dir)} for validation dataset", 1)
	log.complete(f"Set aside {sfutil.bold(total_moved)} tiles for validation dataset", 1)

def merge_validation(train_dir, eval_dir):
	cat_dirs = [_dir for _dir in listdir(eval_dir) if isdir(join(eval_dir, _dir))]
	for cat_dir in cat_dirs:
		print(f"Category {cat_dir}:")
		slide_dirs = [_dir for _dir in listdir(join(eval_dir, cat_dir)) if isdir(join(eval_dir, cat_dir, _dir))]
		for slide_dir in slide_dirs:
			files = [_file for _file in listdir(join(eval_dir, cat_dir, slide_dir)) 
						if (isfile(join(eval_dir, cat_dir, slide_dir, _file))) and
						   (sfutil.path_to_ext(_file) == "jpg")]
			for file in files:
				shutil.move(join(eval_dir, cat_dir, slide_dir, file), join(train_dir, cat_dir, slide_dir, file))
			print(f"  Merged {len(files)} files for slide {slide_dir}")

class Dataset:
	def __init__(self, config_file, sources):
		config = sfutil.load_json(config_file)
		try:
			self.datasets = {k:v for (k,v) in config.items() if k in sources}
		except KeyError:
			sources_list = ", ".join(sources)
			log.error(f"Unable to find datasets named {sfutil.bold(sources_list)} in config file {sfutil.green(config_file)}", 1)
			sys.exit()

	def get_tfrecords(self, ask_to_merge_subdirs=False):
		tfrecords_list = []
		folders_to_search = []
		for d in self.datasets:
			tfrecords = self.datasets[d]['tfrecords']
			label = self.datasets[d]['label']
			tfrecord_path = join(tfrecords, label)
			subdirs = [sd for sd in listdir(tfrecord_path) if isdir(join(tfrecord_path, sd))]

			# Check if given subfolder contains split data (tiles split into multiple TFRecords, likely for validation testing)
			# If true, can merge inputs and to use all data, likely for evaluation
			if len(subdirs) and ask_to_merge_subdirs:
				if sfutil.yes_no_input(f"Warning: TFRecord directory {sfutil.green(tfrecord_path)} contains data split into sub-directories ({', '.join([sfutil.green(s) for s in subdirs])}); merge and use? [y/N] ", default='no'):
					folders_to_search += [join(tfrecord_path, subdir) for subdir in subdirs]
				else:
					sys.exit()
			else:
				if len(subdirs):
					log.warn(f"Warning: TFRecord directory {sfutil.green(tfrecord_path)} contains data split into sub-directories; ignoring sub-directories", 1)
				folders_to_search += [tfrecord_path]
		for folder in folders_to_search:
			tfrecords_list += glob(join(folder, "*.tfrecords"))
		return tfrecords_list

	def get_rois(self):
		rois_list = []
		for d in self.datasets:
			rois_list += glob(join(self.datasets[d]['roi'], "*.csv"))
		return rois_list

	def get_tfrecords_by_subfolder(self, subfolder):
		tfrecords_list = []
		folders_to_search = []
		for d in self.datasets:
			base_dir = join(self.datasets[d]['tfrecords'], self.datasets[d]['label'])
			tfrecord_path = join(base_dir, subfolder)
			if not exists(tfrecord_path):
				log.error(f"Unable to find subfolder {sfutil.bold(subfolder)} in dataset {sfutil.bold(d)}, tfrecord directory: {sfutil.green(base_dir)}")
				sys.exit()
			folders_to_search += [tfrecord_path]
		for folder in folders_to_search:
			tfrecords_list += glob(join(folder, "*.tfrecords"))
		return tfrecords_list

	def get_slides_by_dataset(self, name):
		if name not in self.datasets.keys():
			log.error(f"Dataset {name} not found.")
			sys.exit()
		return sfutil.get_slide_paths(self.datasets[name]['slides'])

	def get_slide_paths(self):
		paths = []
		for d in self.datasets:
			paths += sfutil.get_slide_paths(self.datasets[d]['slides'])
		return paths

	def get_manifest(self):
		combined_manifest = {}
		for d in self.datasets:
			tfrecord_dir = join(self.datasets[d]['tfrecords'], self.datasets[d]['label'])
			combined_manifest.update(sfutil.get_global_manifest(tfrecord_dir))
		return combined_manifest

	def get_tfrecords_folders(self):
		return [join(self.datasets[d]['tfrecords'], self.datasets[d]['label']) for d in self.datasets]