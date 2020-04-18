import shutil
import sys
import os
import csv
import argparse
import copy
import slideflow.util as sfutil

from glob import glob
from random import shuffle
from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
from slideflow.util import log, TCGA, _shortname, make_dir

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
	'''Object to supervise organization of slides, tfrecords, and tiles across a one or more datasets in a stored configuration file.'''

	ANNOTATIONS = []
	filters = None
	filter_blank = None

	def __init__(self, config_file, sources):
		config = sfutil.load_json(config_file)
		sources = sources if type(sources) == list else [sources]
		try:
			self.datasets = {k:v for (k,v) in config.items() if k in sources}
		except KeyError:
			sources_list = ", ".join(sources)
			log.error(f"Unable to find datasets named {sfutil.bold(sources_list)} in config file {sfutil.green(config_file)}", 1)
			sys.exit()

	def apply_filters(self, filters=None, filter_blank=None):
		self.filters = filters
		self.filter_blank = filter_blank

	def get_global_manifest(self, directory):
		'''Loads a saved relative manifest at a directory and returns a dict containing
		absolute/global path and file names.'''
		manifest_path = join(directory, "manifest.json")
		if not exists(manifest_path):
			log.info(f"No manifest file detected in {directory}; will create now", 1)
			self.update_tfrecord_manifest(directory)
		relative_manifest = sfutil.load_json(manifest_path)
		global_manifest = {}
		for record in relative_manifest:
			global_manifest.update({join(directory, record): relative_manifest[record]})
		return global_manifest

	def get_manifest(self):
		'''Generates a manifest of all tfrecords.'''
		combined_manifest = {}
		for d in self.datasets:
			tfrecord_dir = join(self.datasets[d]['tfrecords'], self.datasets[d]['label'])
			combined_manifest.update(self.get_global_manifest(tfrecord_dir))
		return combined_manifest

	def get_rois(self):
		'''Returns a list of all ROIs.'''
		rois_list = []
		for d in self.datasets:
			rois_list += glob(join(self.datasets[d]['roi'], "*.csv"))
		return rois_list

	def get_slides(self):
		'''Returns a list of slide names from the annotations file using a given set of filters.'''
		
		# Begin filtering slides with annotations
		slides = []
		self.filter_blank = [self.filter_blank] if type(self.filter_blank) != list else self.filter_blank
		slide_patient_dict = {}
		if not len(self.ANNOTATIONS):
			print(self.ANNOTATIONS)
			log.error("No annotations loaded; is the annotations file empty?")
		for ann in self.ANNOTATIONS:
			skip_annotation = False
			if TCGA.slide not in ann.keys():
				log.error(f"{TCGA.slide} not found in annotations file.")
				sys.exit()

			# Skip missing or blank slides
			if ann[TCGA.slide] in sfutil.SLIDE_ANNOTATIONS_TO_IGNORE:
				continue

			# Ensure slides are only assigned to a single patient
			if ann[TCGA.slide] not in slide_patient_dict:
				slide_patient_dict.update({ann[TCGA.slide]: ann[TCGA.patient]})
			elif slide_patient_dict[ann[TCGA.slide]] != ann[TCGA.patient]:
				log.error(f"Multiple patients assigned to slide {sfutil.green(ann[TCGA.slide])}.")
				return None

			# Only return slides with annotation values specified in "filters"
			if self.filters:
				for filter_key in self.filters.keys():
					if filter_key not in ann.keys():
						log.error(f"Filter header {sfutil.bold(filter_key)} not found in annotations file.")
						raise IndexError(f"Filter header {filter_key} not found in annotations file.")
					if    ((type(self.filters[filter_key]) == list and ann[filter_key] not in self.filters[filter_key]) 
						or (type(self.filters[filter_key]) != list and self.filters[filter_key] != ann[filter_key])):
						skip_annotation = True
						break

			# Filter out slides that are blank in a given annotation column ("filter_blank")
			if self.filter_blank and self.filter_blank != [None]:
				for fb in self.filter_blank:
					if fb not in ann.keys():
						log.error(f"Unable to filter blank slides from header {fb}; this header was not found in the annotations file.")
						sys.exit()
					if not ann[fb] or ann[fb] == '':
						skip_annotation = True
						break
			if skip_annotation: continue
			slides += [ann[TCGA.slide]]
		
		return slides

	def get_slide_paths(self, dataset=None):
		'''Returns a list of paths to all slides.'''
		if dataset and dataset not in self.datasets.keys():
			log.error(f"Dataset {name} not found.")
			return None

		# Get unfiltered paths
		if dataset:
			paths = sfutil.get_slide_paths(self.datasets[name]['slides'])
		else:
			
			paths = []
			for d in self.datasets:
				paths += sfutil.get_slide_paths(self.datasets[d]['slides'])

		# Filter paths
		filtered_slides = self.get_slides()
		filtered_paths = [path for path in paths if sfutil.path_to_name(path) in filtered_slides]
		return filtered_paths

	def get_tfrecords(self, ask_to_merge_subdirs=False):
		'''Returns a list of all tfrecords.'''
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

		# Now filter the list
		filtered_tfrecords_list = [tfrecord for tfrecord in tfrecords_list if tfrecord.split('/')[-1][:-10] in self.get_slides()]

		return filtered_tfrecords_list

	def get_tfrecords_by_subfolder(self, subfolder):
		'''Returns a list of tfrecords in a specific subfolder.'''
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

	def get_tfrecords_folders(self):
		'''Returns folders containing tfrecords.'''
		return [join(self.datasets[d]['tfrecords'], self.datasets[d]['label']) for d in self.datasets]

	def get_outcomes_from_annotations(self, headers, use_float=False):
		'''Returns a dictionary of slide names mapping to patient id and [an] outcome variable(s).

		Args:
			headers			annotation header(s) that specifies outcome variable. May be a list.
			use_float		If true, will try to convert data into float

		Returns:
			Dictionary with slides as keys and dictionaries as values. The value dictionaries contain both "TCGA.patient" and "outcome" keys.
		'''
		slides = self.get_slides()
		filtered_annotations = [a for a in self.ANNOTATIONS if a[TCGA.slide] in slides]
		results = {}
		headers = [headers] if type(headers) != list else headers
		assigned_headers = {}
		unique_outcomes = None
		for header in headers:
			assigned_headers[header] = {}
			try:
				filtered_outcomes = [a[header] for a in filtered_annotations]
			except KeyError:
				log.error(f"Unable to find column {header} in annotation file.", 1)
				sys.exit()

			# Ensure outcomes can be converted to desired type
			if use_float:
				try:
					filtered_outcomes = [float(o) for o in filtered_outcomes]
				except ValueError:
					log.error(f"Unable to convert outcome {sfutil.bold(header)} into type 'float'.", 1)
					raise TypeError(f"Unable to convert outcome {header} into type 'float'.")
			else:
				log.info(f'Assigning outcome descriptors in column "{header}" to numerical values', 1)
				unique_outcomes = list(set(filtered_outcomes))
				unique_outcomes.sort()
				for i, uo in enumerate(unique_outcomes):
					num_matching_slides_filtered = sum(o == uo for o in filtered_outcomes)
					log.empty(f"{header} '{sfutil.info(uo)}' assigned to value '{i}' [{sfutil.bold(str(num_matching_slides_filtered))} slides]", 2)
			
			# Create function to process/convert outcome
			def _process_outcome(o):
				if use_float:
					return float(o)
				else:
					return unique_outcomes.index(o)

			# Assemble results dictionary
			patient_outcomes = {}
			for annotation in filtered_annotations:
				slide = annotation[TCGA.slide]
				patient = annotation[TCGA.patient]
				annotation_outcome = _process_outcome(annotation[header])

				# Mark this slide as having been already assigned an outcome with his header
				assigned_headers[header][slide] = True

				# Ensure patients do not have multiple outcomes
				if patient not in patient_outcomes:
					patient_outcomes[patient] = annotation_outcome
				elif patient_outcomes[patient] != annotation_outcome:
					log.error(f"Multiple different outcomes in header {header} found for patient {patient} ({patient_outcomes[patient]}, {annotation_outcome})", 1)
					sys.exit()
				elif (slide in slides) and (slide in results) and (slide in assigned_headers[header]):
					continue

				if slide in slides:
					if slide in results:
						so = results[slide]['outcome']
						results[slide]['outcome'] = [so] if type(so) != list else so
						results[slide]['outcome'] += [annotation_outcome]
					else:
						results[slide] = {'outcome': annotation_outcome if not use_float else [annotation_outcome]}
						results[slide][TCGA.patient] = patient
		return results, unique_outcomes

	def load_annotations(self, annotations_file):
		'''Load annotations from a given CSV file.'''
		# Verify annotations file exists
		if not os.path.exists(annotations_file):
			log.error(f"Annotations file {sfutil.green(annotations_file)} does not exist, unable to load")
			sys.exit()

		header, current_annotations = sfutil.read_annotations(annotations_file)

		# Check for duplicate headers in annotations file
		if len(header) != len(set(header)):
			log.error("Annotations file containers at least one duplicate header; all headers must be unique")
			sys.exit()

		# Verify there is a patient header
		try:
			patient_index = header.index(TCGA.patient)
		except:
			log.error(f"Check annotations file for header '{TCGA.patient}'.", 1)
			sys.exit()

		# Verify that a slide header exists; if not, offer to make one and automatically associate slide names with patients
		try:
			slide_index = header.index(TCGA.slide)
		except:
			log.error(f"Header column '{TCGA.slide}' not found.", 1)
			if sfutil.yes_no_input('\nSearch slides directory and automatically associate patients with slides? [Y/n] ', default='yes'):
				self.update_annotations_with_slidenames(annotations_file)
				header, current_annotations = sfutil.read_annotations(annotations_file)
			else:
				sys.exit()
		self.ANNOTATIONS = current_annotations

	def verify_annotations_slides(self):
		'''Verify that annotations are correctly loaded.'''
		slide_list = self.get_slide_paths()

		# Verify no duplicate slide names are found
		slide_list_from_annotations = self.get_slides()
		if len(slide_list_from_annotations) != len(list(set(slide_list_from_annotations))):
			log.error("Duplicate slide names detected in the annotation file.")
			sys.exit()

		# Verify all SVS files in the annotation column are valid
		num_warned = 0
		warn_threshold = 3
		for annotation in self.ANNOTATIONS:
			print_func = print if num_warned < warn_threshold else None
			slide = annotation[TCGA.slide]
			if slide == '':
				log.warn(f"Patient {sfutil.green(annotation[TCGA.patient])} has no slide assigned.", 1, print_func)
				num_warned += 1
			elif not slide in [sfutil.path_to_name(s) for s in slide_list]:
				log.warn(f"Unable to locate slide {slide}", 1, print_func)
				num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)
		if not num_warned:
			log.info(f"Slides successfully verified, no errors found.", 1)

	def update_tfrecord_manifest(self, directory, force_update=False):
		'''Log number of tiles in each TFRecord file present in the given directory and all subdirectories, 
		saving manifest to file within the parent directory.'''
		import tensorflow as tf

		slide_list = []
		manifest_path = join(directory, "manifest.json")
		manifest = {} if not exists(manifest_path) else sfutil.load_json(manifest_path)
		prior_manifest = copy.deepcopy(manifest)
		try:
			relative_tfrecord_paths = sfutil.get_relative_tfrecord_paths(directory)
		except FileNotFoundError:
			log.warn(f"Unable to find TFRecords in the directory {directory}")
			return
		slide_names_from_annotations = self.get_slides()

		slide_list_errors = []
		for rel_tfr in relative_tfrecord_paths:
			tfr = join(directory, rel_tfr)

			if (not force_update) and (rel_tfr in manifest) and ('total' in manifest[rel_tfr]):
				continue

			manifest.update({rel_tfr: {}})
			raw_dataset = tf.data.TFRecordDataset(tfr)
			print(f" + Verifying tiles in {sfutil.green(rel_tfr)}...", end="\r\033[K")
			total = 0
			for raw_record in raw_dataset:
				example = tf.train.Example()
				example.ParseFromString(raw_record.numpy())
				slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')
				if slide not in manifest[rel_tfr]:
					manifest[rel_tfr][slide] = 1
				else:
					manifest[rel_tfr][slide] += 1
				total += 1
			manifest[rel_tfr]['total'] = total
		print()

		# Find slides that have TFRecords
		for man_rel_tfr in manifest:
			try:
				for slide_key in manifest[man_rel_tfr]:
					if slide_key != 'total':
						slide_list.extend([slide_key])
						slide_list = list(set(slide_list))
						if slide_key not in slide_names_from_annotations:
							slide_list_errors.extend([slide_key])
							slide_list_errors = list(set(slide_list_errors))
			except:
				continue
		
		error_threshold = 3
		for s, slide in enumerate(slide_list_errors):
			print_func = print if s < error_threshold else None
			log.warn(f"Failed TFRecord integrity check: annotation not found for slide {sfutil.green(slide)}", 1, print_func)

		if len(slide_list_errors) >= error_threshold:
			log.warn(f"...{len(slide_list_errors)} total TFRecord integrity check failures, see {sfutil.green(log.logfile)} for details", 1)
		if len(slide_list_errors) == 0:
			log.info("TFRecords verified, no errors found.", 1)

		sys.stdout.write("\r\033[K")
		sys.stdout.flush()

		# Write manifest file
		if manifest != prior_manifest:
			sfutil.write_json(manifest, manifest_path)

		return manifest
		
	def update_annotations_with_slidenames(self, annotations_file):
		'''Attempts to automatically associate slide names from a directory with patients in a given annotations file.'''
		header, _ = sfutil.read_annotations(annotations_file)
		slide_list = self.get_slide_paths()

		# First, load all patient names from the annotations file
		try:
			patient_index = header.index(TCGA.patient)
		except:
			log.error(f"Patient header {TCGA.patient} not found in annotations file.")
			sys.exit()
		patients = []
		patient_slide_dict = {}
		with open(annotations_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			header = next(csv_reader, None)
			for row in csv_reader:
				patients.extend([row[patient_index]])
		patients = list(set(patients)) 

		# Then, check for sets of slides that would match to the same patient; due to ambiguity, these will be skipped.
		num_occurrences = {}
		for slide in slide_list:
			if _shortname(slide) not in num_occurrences:
				num_occurrences[_shortname(slide)] = 1
			else:
				num_occurrences[_shortname(slide)] += 1
		slides_to_skip = [slide for slide in slide_list if num_occurrences[_shortname(slide)] > 1]

		# Next, search through the slides folder for all SVS/JPG files
		num_warned = 0
		warn_threshold = 1
		for slide_filename in slide_list:
			slide_name = sfutil.path_to_name(slide_filename)
			print_func = print if num_warned < warn_threshold else None
			# First, skip this slide due to ambiguity if needed
			if slide_name in slides_to_skip:
				log.warn(f"Unable to associate slide {slide_name} due to ambiguity; multiple slides match to patient id {_shortname(slide_name)}; skipping.", 1)
				num_warned += 1
			# Then, make sure the shortname and long name aren't both in the annotation file
			if (slide_name != _shortname(slide_name)) and (slide_name in patients) and (_shortname(slide_name) in patients):
				log.warn(f"Unable to associate slide {slide_name} due to ambiguity; both {slide_name} and {_shortname(slide_name)} are patients in annotation file; skipping.", 1)
				num_warned += 1

			# Check if either the slide name or the shortened version are in the annotation file
			if any(x in patients for x in [slide_name, _shortname(slide_name)]):
				slide = slide_name if slide_name in patients else _shortname(slide_name)
				patient_slide_dict.update({slide: slide_name})
			else:
				#log.warn(f"Slide '{slide_name}' not found in annotations file, skipping.", 1, print_func)
				#num_warned += 1
				pass
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

		# Now, write the assocations
		num_updated_annotations = 0
		num_missing = 0
		with open(annotations_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			header = next(csv_reader, None)
			with open('temp.csv', 'w') as csv_outfile:
				csv_writer = csv.writer(csv_outfile, delimiter=',')

				# Write to existing "slide" column in the annotations file if it exists, 
				# otherwise create new column
				try:
					slide_index = header.index(TCGA.slide)
					csv_writer.writerow(header)
					for row in csv_reader:
						patient = row[patient_index]
						# Only write column if no slide is documented in the annotation
						if (patient in patient_slide_dict) and (row[slide_index] == ''):
							row[slide_index] = patient_slide_dict[patient]
							num_updated_annotations += 1
						elif (patient not in patient_slide_dict) and (row[slide_index] == ''):
							num_missing += 1
						csv_writer.writerow(row)
				except:
					header.extend([TCGA.slide])
					csv_writer.writerow(header)
					for row in csv_reader:
						patient = row[patient_index]
						if patient in patient_slide_dict:
							row.extend([patient_slide_dict[patient]])
							num_updated_annotations += 1
						else:
							row.extend([""])
							num_missing += 1
						csv_writer.writerow(row)
		log.info(f"Successfully associated slides with {num_updated_annotations} annotation entries. Slides not found for {num_missing} annotations.", 1)

		# Finally, backup the old annotation file and overwrite existing with the new data
		backup_file = f"{annotations_file}.backup"
		if exists(backup_file):
			os.remove(backup_file)
		shutil.move(annotations_file, backup_file)
		shutil.move('temp.csv', annotations_file)