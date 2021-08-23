import shutil
import os
import csv
import copy
import slideflow.util as sfutil

from glob import glob
from random import shuffle
from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
from slideflow.util import log, TCGA, _shortname, make_dir

class DatasetError(Exception):
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
		err_msg = f'When splitting tiles, length of "fraction" ({len(fraction)}) should equal length of "names" ({len(names)})'
		log.error(err_msg)
		raise DatasetError(err_msg)
	if sum([i for i in fraction if i != -1]) > 1:
		log.error('Unable to split tiles; Sum of fraction is greater than 1')
		raise DatasetError('Unable to split tiles; Sum of fraction is greater than 1')

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
			err_msg = f"Error with separating tiles; tried to move {sum(num_to_move)} tiles into {len(fraction)} subfolders, only {num_files} tiles available"
			log.error(err_msg, 1)
			raise DatasetError(err_msg)
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

	def __init__(self, config_file, sources, tile_px, tile_um, annotations=None, filters=None, filter_blank=None):
		self.ANNOTATIONS = []
		self.filter_blank = []
		self.filters = []
		config = sfutil.load_json(config_file)
		sources = sources if isinstance(sources, list) else [sources]

		try:
			self.datasets = {k:v for (k,v) in config.items() if k in sources}
		except KeyError:
			sources_list = ", ".join(sources)
			err_msg = f"Unable to find datasets named {sfutil.bold(sources_list)} in config file {sfutil.green(config_file)}"
			log.error(err_msg, 1)
			raise DatasetError(err_msg)

		if (tile_px is not None) and (tile_um is not None):
			label = f"{tile_px}px_{tile_um}um"
		else:
			label = None

		for dataset in self.datasets:
			self.datasets[dataset]['label'] = label
		
		if annotations:
			self.load_annotations(annotations)
			
		if filters or filter_blank:
			self.apply_filters(filters=filters, filter_blank=filter_blank)

	def apply_filters(self, filters=None, filter_blank=None):
		self.filters = filters
		self.filter_blank = filter_blank

	def get_manifest(self, key='path'):
		'''Generates a manifest of all tfrecords.
		
		Args:
			key:	Either 'path' (default) or 'name'. Determines key format in the manifest dictionary.
			
		Returns:
			Dictionary mapping key (path or slide name) to number of total tiles. 
		'''
		if key not in ('path', 'name'):
			raise DatasetError("'key' must be in ['path, 'name']")

		combined_manifest = {}
		for d in self.datasets:
			if self.datasets[d]['label'] is None: continue
			tfrecord_dir = join(self.datasets[d]['tfrecords'], self.datasets[d]['label'])
			manifest_path = join(tfrecord_dir, "manifest.json")
			if not exists(manifest_path):
				log.info(f"No manifest file detected in {tfrecord_dir}; will create now", 1)
				self.update_manifest_at_dir(tfrecord_dir)
			relative_manifest = sfutil.load_json(manifest_path)
			global_manifest = {}
			for record in relative_manifest:
				k = join(tfrecord_dir, record) if key == 'path' else sfutil.path_to_name(record)
				global_manifest.update({k: relative_manifest[record]})
			combined_manifest.update(global_manifest)
		
		# Now filter out any tfrecords that would be excluded by filters
		filtered_tfrecords = self.get_tfrecords() if key =='path' else [sfutil.path_to_name(tfr) for tfr in self.get_tfrecords()]
		manifest_tfrecords = list(combined_manifest.keys())
		for tfr in manifest_tfrecords:
			if tfr not in filtered_tfrecords:
				del(combined_manifest[tfr])
			
		return combined_manifest

	def get_rois(self):
		'''Returns a list of all ROIs.'''
		rois_list = []
		for d in self.datasets:
			rois_list += glob(join(self.datasets[d]['roi'], "*.csv"))
		rois_list = list(set(rois_list))
		return rois_list

	def get_slides(self):
		'''Returns a list of slide names from the annotations file using a given set of filters.'''
		
		# Begin filtering slides with annotations
		slides = []
		self.filter_blank = [self.filter_blank] if not isinstance(self.filter_blank, list) else self.filter_blank
		slide_patient_dict = {}
		if not len(self.ANNOTATIONS):
			log.error("No annotations loaded; is the annotations file empty?")
		for ann in self.ANNOTATIONS:
			skip_annotation = False
			if TCGA.slide not in ann.keys():
				err_msg = f"{TCGA.slide} not found in annotations file."
				log.error(err_msg)
				raise DatasetError(err_msg)

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

					ann_val = ann[filter_key]
					filter_vals = self.filters[filter_key]
					filter_vals = [filter_vals] if not isinstance(filter_vals, list) else filter_vals

					# Allow filtering based on shortnames if the key is a TCGA patient ID
					if filter_key == TCGA.patient:
						if ((ann_val not in filter_vals) and
							(sfutil._shortname(ann_val) not in filter_vals) and
							(ann_val not in [sfutil._shortname(fv) for fv in filter_vals]) and
							(sfutil._shortname(ann_val) not in [sfutil._shortname(fv) for fv in filter_vals])):

							skip_annotation = True
							break
					else:
						if ann_val not in filter_vals:
							skip_annotation = True
							break

			# Filter out slides that are blank in a given annotation column ("filter_blank")
			if self.filter_blank and self.filter_blank != [None]:
				for fb in self.filter_blank:
					if fb not in ann.keys():
						err_msg = f"Unable to filter blank slides from header {fb}; this header was not found in the annotations file."
						log.error(err_msg)
						raise DatasetError(err_msg)
						
					if not ann[fb] or ann[fb] == '':
						skip_annotation = True
						break
			if skip_annotation: continue
			slides += [ann[TCGA.slide]]
		return slides

	def get_slide_paths(self, dataset=None, filter=True):
		'''Returns a list of paths to all slides.'''

		if dataset and dataset not in self.datasets.keys():
			log.error(f"Dataset {dataset} not found.")
			return None

		# Get unfiltered paths
		if dataset:
			paths = sfutil.get_slide_paths(self.datasets[dataset]['slides'])
		else:
			paths = []
			for d in self.datasets:
				paths += sfutil.get_slide_paths(self.datasets[d]['slides'])

		# Remove any duplicates from shared dataset paths
		paths = list(set(paths))

		# Filter paths
		if filter:
			filtered_slides = self.get_slides()
			filtered_paths = [path for path in paths if sfutil.path_to_name(path) in filtered_slides]
			return filtered_paths
		else:
			return paths

	def get_tfrecords(self, dataset=None, merge_subdirs=False, ask_to_merge_subdirs=False):
		'''Returns a list of all tfrecords.'''
		if dataset and dataset not in self.datasets.keys():
			log.error(f"Dataset {dataset} not found.")
			return None

		datasets_to_search = list(self.datasets.keys()) if not dataset else [dataset]

		tfrecords_list = []
		folders_to_search = []
		for d in datasets_to_search:
			tfrecords = self.datasets[d]['tfrecords']
			label = self.datasets[d]['label']
			if label is None: continue
			tfrecord_path = join(tfrecords, label)
			if not exists(tfrecord_path):
				log.warn(f"TFRecords path not found: {sfutil.green(tfrecord_path)}", 1)
				return []
			subdirs = [sd for sd in listdir(tfrecord_path) if isdir(join(tfrecord_path, sd))]

			# Check if given subfolder contains split data (tiles split into multiple TFRecords, likely for validation testing)
			# If true, can merge inputs and to use all data, likely for evaluation
			if len(subdirs) and merge_subdirs:
				log.info(f"Warning: TFRecord directory {sfutil.green(tfrecord_path)} contains data split into sub-directories ({', '.join([sfutil.green(s) for s in subdirs])}); will use TFRecords from all", 1)
				folders_to_search += [join(tfrecord_path, subdir) for subdir in subdirs]
			elif len(subdirs) and ask_to_merge_subdirs:
				if sfutil.yes_no_input(f"Warning: TFRecord directory {sfutil.green(tfrecord_path)} contains data split into sub-directories ({', '.join([sfutil.green(s) for s in subdirs])}); merge and use? [y/N] ", default='no'):
					folders_to_search += [join(tfrecord_path, subdir) for subdir in subdirs]
				else:
					raise NotImplementedError("Handling not implemented.")
			else:
				if len(subdirs):
					log.warn(f"Warning: TFRecord directory {sfutil.green(tfrecord_path)} contains data split into sub-directories; ignoring sub-directories", 1)
				folders_to_search += [tfrecord_path]
		for folder in folders_to_search:
			tfrecords_list += glob(join(folder, "*.tfrecords"))

		# Now filter the list
		if self.ANNOTATIONS:
			slides = self.get_slides()
			filtered_tfrecords_list = [tfrecord for tfrecord in tfrecords_list if tfrecord.split('/')[-1][:-10] in slides]
			return filtered_tfrecords_list
		else:
			log.warn("No annotations loaded; unable to filter TFRecords list. Is the annotations file empty?", 1)
			return tfrecords_list			

	def get_tfrecords_by_subfolder(self, subfolder):
		'''Returns a list of tfrecords in a specific subfolder.'''
		tfrecords_list = []
		folders_to_search = []
		for d in self.datasets:
			if self.datasets[d]['label'] is None: continue
			base_dir = join(self.datasets[d]['tfrecords'], self.datasets[d]['label'])
			tfrecord_path = join(base_dir, subfolder)
			if not exists(tfrecord_path):
				err_msg = f"Unable to find subfolder {sfutil.bold(subfolder)} in dataset {sfutil.bold(d)}, tfrecord directory: {sfutil.green(base_dir)}"
				log.error(err_msg)
				raise DatasetError(err_msg)
			folders_to_search += [tfrecord_path]
		for folder in folders_to_search:
			tfrecords_list += glob(join(folder, "*.tfrecords"))
		return tfrecords_list

	def get_tfrecords_folders(self):
		'''Returns folders containing tfrecords.'''
		folders = []
		for d in self.datasets:
			if self.datasets[d]['label'] is None: continue
			folders += [join(self.datasets[d]['tfrecords'], self.datasets[d]['label'])]
		return folders

	def get_labels_from_annotations(self, headers, use_float=False, assigned_labels=None, key='label'):
		'''Returns a dictionary of slide names mapping to patient id and [an] label(s).

		Args:
			headers			annotation header(s) that specifies label variable. May be a list.
			use_float		Either bool, dict, or 'auto'. 
								If true, will try to convert all data into float. If unable, will raise TypeError.
								If false, will interpret all data as categorical.
								If a dict is provided, will look up each header to determine whether float should be used.
								If 'auto', will try to convert all data into float. For each header in which this fails, 
									will interpret as categorical instead.
			assigned_labels	Dictionary mapping label ids to label names. If not provided, will map
								ids to names by sorting alphabetically.
			key				Key name to use for the returned dictionary. Defaults to 'label'

		Returns:
			1) Dictionary with slides as keys and dictionaries as values. The value dictionaries contain both "TCGA.patient" and "label" (or manually specified) keys.
			2) list of unique labels
		'''
		slides = self.get_slides()
		filtered_annotations = [a for a in self.ANNOTATIONS if a[TCGA.slide] in slides]
		results = {}
		headers = [headers] if not isinstance(headers, list) else headers
		assigned_headers = {}
		unique_labels = {}
		for header in headers:
			if assigned_labels and (len(headers) > 1 or header in assigned_labels):
				assigned_labels_for_this_header = assigned_labels[header]
			elif assigned_labels:
				assigned_labels_for_this_header = assigned_labels
			else:
				assigned_labels_for_this_header = None

			unique_labels_for_this_header = []
			assigned_headers[header] = {}
			try:
				filtered_labels = [a[header] for a in filtered_annotations]
			except KeyError:
				log.error(f"Unable to find column {header} in annotation file.", 1)
				raise DatasetError(f"Unable to find column {header} in annotation file.")

			# Determine whether values should be converted into float
			if type(use_float) == dict and header not in use_float:
				raise DatasetError(f"Dict was provided to use_float, but header {header} is missing.")
			elif type(use_float) == dict:
				use_float_for_this_header = use_float[header]
			elif type(use_float) == bool:
				use_float_for_this_header = use_float
			elif use_float == 'auto':
				try:
					filtered_labels = [float(o) for o in filtered_labels]
					use_float_for_this_header = True
				except ValueError:
					use_float_for_this_header = False
			else:
				raise DatasetError(f"Invalid use_float option {use_float}")

			# Ensure labels can be converted to desired type, then assign values
			if use_float_for_this_header:
				try:
					filtered_labels = [float(o) for o in filtered_labels]
				except ValueError:
					raise TypeError(f"Unable to convert label {header} into type 'float'.")
			else:
				log.info(f'Assigning label descriptors in column "{header}" to numerical values', 1)
				unique_labels_for_this_header = list(set(filtered_labels))
				unique_labels_for_this_header.sort()
				for i, ul in enumerate(unique_labels_for_this_header):
					num_matching_slides_filtered = sum(l == ul for l in filtered_labels)
					if assigned_labels_for_this_header and ul not in assigned_labels_for_this_header:
						raise KeyError(f"assigned_labels was provided, but label {ul} not found in this dict")
					elif assigned_labels_for_this_header:
						log.empty(f"{header} '{sfutil.info(ul)}' assigned to value '{assigned_labels_for_this_header[ul]}' [{sfutil.bold(str(num_matching_slides_filtered))} slides]", 2)
					else:
						log.empty(f"{header} '{sfutil.info(ul)}' assigned to value '{i}' [{sfutil.bold(str(num_matching_slides_filtered))} slides]", 2)
			
			# Create function to process/convert label
			def _process_label(o):
				if use_float_for_this_header:
					return float(o)
				elif assigned_labels_for_this_header:
					return assigned_labels_for_this_header[o]
				else:
					return unique_labels_for_this_header.index(o)

			# Assemble results dictionary
			patient_labels = {}
			num_warned = 0
			warn_threshold = 3
			for annotation in filtered_annotations:
				slide = annotation[TCGA.slide]
				patient = annotation[TCGA.patient]
				annotation_label = _process_label(annotation[header])
				print_func = print if num_warned < warn_threshold else None

				# Mark this slide as having been already assigned a label with his header
				assigned_headers[header][slide] = True

				# Ensure patients do not have multiple labels
				if patient not in patient_labels:
					patient_labels[patient] = annotation_label
				elif patient_labels[patient] != annotation_label:
					log.error(f"Multiple different labels in header {header} found for patient {patient} ({patient_labels[patient]}, {annotation_label})", 1, print_func)
					num_warned += 1
				elif (slide in slides) and (slide in results) and (slide in assigned_headers[header]):
					continue

				if slide in slides:
					if slide in results:
						so = results[slide][key]
						results[slide][key] = [so] if not isinstance(so, list) else so
						results[slide][key] += [annotation_label]
					else:
						results[slide] = {key: annotation_label if not use_float_for_this_header else [annotation_label]}
						results[slide][TCGA.patient] = patient
			if num_warned >= warn_threshold:
				log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)
			unique_labels[header] = unique_labels_for_this_header
		if len(headers) == 1:
			unique_labels = unique_labels[headers[0]]
		return results, unique_labels

	def slide_to_label(self, headers, use_float=False, return_unique=False):
		labels, unique_labels = self.get_labels_from_annotations(headers=headers, use_float=use_float)
		if not use_float and not unique_labels:
			raise DatasetError(f"No labels were detected for header {headers} in this dataset")
		elif not use_float:
			return_dict = {k:unique_labels[v['label']] for k, v in labels.items()}
		else:
			return_dict = {k:labels[k]['label'] for k,v in labels.items()}
		if return_unique:
			return return_dict, unique_labels
		else:
			return return_dict

	def load_annotations(self, annotations_file):
		'''Load annotations from a given CSV file.'''
		# Verify annotations file exists
		if not os.path.exists(annotations_file):
			raise DatasetError(f"Annotations file {sfutil.green(annotations_file)} does not exist, unable to load")

		header, current_annotations = sfutil.read_annotations(annotations_file)

		# Check for duplicate headers in annotations file
		if len(header) != len(set(header)):
			err_msg = "Annotations file containers at least one duplicate header; all headers must be unique"
			log.error(err_msg)
			raise DatasetError(err_msg)

		# Verify there is a patient header
		try:
			patient_index = header.index(TCGA.patient)
		except:
			print(header)
			err_msg = f"Check that annotations file is formatted correctly and contains header '{TCGA.patient}'."
			log.error(err_msg, 1)
			raise DatasetError(err_msg)

		# Verify that a slide header exists; if not, offer to make one and automatically associate slide names with patients
		try:
			slide_index = header.index(TCGA.slide)
		except:
			log.info(f"Header column '{TCGA.slide}' not found.", 1)
			log.info("Attempting to automatically associate patients with slides...", 1)
			self.update_annotations_with_slidenames(annotations_file)
			header, current_annotations = sfutil.read_annotations(annotations_file)
		self.ANNOTATIONS = current_annotations

	def verify_annotations_slides(self):
		'''Verify that annotations are correctly loaded.'''

		# Verify no duplicate slide names are found
		slide_list_from_annotations = self.get_slides()
		if len(slide_list_from_annotations) != len(list(set(slide_list_from_annotations))):
			log.error("Duplicate slide names detected in the annotation file.")
			raise DatasetError("Duplicate slide names detected in the annotation file.")

		# Verify all SVS files in the annotation column are valid
		num_warned = 0
		warn_threshold = 3
		for annotation in self.ANNOTATIONS:
			print_func = print if num_warned < warn_threshold else None
			slide = annotation[TCGA.slide]
			if slide == '':
				log.warn(f"Patient {sfutil.green(annotation[TCGA.patient])} has no slide assigned.", 1, print_func)
				num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)
		if not num_warned:
			log.info(f"Slides successfully verified, no errors found.", 1)

	def update_manifest(self, force_update=False):
		tfrecords_folders = self.get_tfrecords_folders()
		for tfr_folder in tfrecords_folders:
			self.update_manifest_at_dir(directory=tfr_folder, 
										force_update=force_update)

	def update_manifest_at_dir(self, directory, force_update=False):
		'''Log number of tiles in each TFRecord file present in the given directory and all subdirectories, 
		saving manifest to file within the parent directory.'''
		import tensorflow as tf

		manifest_path = join(directory, "manifest.json")
		manifest = {} if not exists(manifest_path) else sfutil.load_json(manifest_path)
		prior_manifest = copy.deepcopy(manifest)
		try:
			relative_tfrecord_paths = sfutil.get_relative_tfrecord_paths(directory)
		except FileNotFoundError:
			log.warn(f"Unable to find TFRecords in the directory {directory}", 1)
			return
		slide_names_from_annotations = self.get_slides()

		# Verify all tfrecords in manifest exist
		for rel_tfr in prior_manifest.keys():
			tfr = join(directory, rel_tfr)
			if not exists(tfr):
				log.warn(f"TFRecord in manifest was not found at {tfr}; removing", 1)
				del(manifest[rel_tfr])

		# Verify detected TFRecords are in manifest, recording number of tiles if not
		slide_list_errors = []
		
		for rel_tfr in relative_tfrecord_paths:
			tfr = join(directory, rel_tfr)

			if (not force_update) and (rel_tfr in manifest) and ('total' in manifest[rel_tfr]):
				continue

			manifest.update({rel_tfr: {}})
			try:
				raw_dataset = tf.data.TFRecordDataset(tfr)
			except Exception as e:
				log.error(f"Unable to open TFRecords file with Tensorflow: {str(e)}")
				return
			if log.INFO_LEVEL > 0: print(f"\r\033[K + Verifying tiles in {sfutil.green(rel_tfr)}...", end="")
			total = 0
			try:
				for raw_record in raw_dataset:
					example = tf.train.Example()
					example.ParseFromString(raw_record.numpy())
					slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')
					if slide not in manifest[rel_tfr]:
						manifest[rel_tfr][slide] = 1
					else:
						manifest[rel_tfr][slide] += 1
					total += 1
			except tf.errors.DataLossError:
				print('\r\033[K', end="")
				log.error(f"Corrupt or incomplete TFRecord at {tfr}", 1)
				log.info(f"Deleting and removing corrupt TFRecord from manifest...", 1)
				del(raw_dataset)
				os.remove(tfr)
				del(manifest[rel_tfr])
				continue
			manifest[rel_tfr]['total'] = total
			print('\r\033[K', end="")

		# Write manifest file
		if (manifest != prior_manifest) or (manifest == {}):
			sfutil.write_json(manifest, manifest_path)

		return manifest
		
	def update_annotations_with_slidenames(self, annotations_file):
		'''Attempts to automatically associate slide names from a directory with patients in a given annotations file,
			skipping any slide names that are already present in the annotations file.'''
		header, _ = sfutil.read_annotations(annotations_file)
		slide_list = self.get_slide_paths(filter=False)

		# First, load all patient names from the annotations file
		try:
			patient_index = header.index(TCGA.patient)
		except:
			err_msg = f"Patient header {TCGA.patient} not found in annotations file."
			log.error(err_msg)
			raise DatasetError(f"Patient header {TCGA.patient} not found in annotations file.")
		patients = []
		patient_slide_dict = {}
		with open(annotations_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			header = next(csv_reader, None)
			for row in csv_reader:
				patients.extend([row[patient_index]])
		patients = list(set(patients))
		log.info(f"Number of patients in annotations: {len(patients)}", 1)
		log.info(f"Slides found: {len(slide_list)}", 1)

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
		if num_updated_annotations:
			log.complete(f"Successfully associated slides with {num_updated_annotations} annotation entries. Slides not found for {num_missing} annotations.", 1)
		elif num_missing:
			log.complete(f"No annotation updates performed. Slides not found for {num_missing} annotations.", 1)
		else:
			log.complete(f"Annotations up-to-date, no changes made.", 1)

		# Finally, backup the old annotation file and overwrite existing with the new data
		backup_file = f"{annotations_file}.backup"
		if exists(backup_file):
			os.remove(backup_file)
		shutil.move(annotations_file, backup_file)
		shutil.move('temp.csv', annotations_file)