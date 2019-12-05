import sys
import json
import csv

import os
from os.path import join, isdir, exists
from glob import glob
import shutil
import datetime, time

import tensorflow as tf
from tensorflow.keras import backend as K

# Enable color sequences on Windows
try:
	import ctypes
	kernel32 = ctypes.windll.kernel32
	kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except:
	pass
# ------

PROJECT_DIR = None
ANNOTATIONS = []
SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tiff', 'svslide', 'bif']

SLIDE_ANNOTATIONS_TO_IGNORE = ['', 'na', 'n/a', 'none', 'missing']

HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

FORMATTING_OPTIONS = [HEADER, BLUE, GREEN, WARNING, FAIL, ENDC, BOLD, UNDERLINE]
LOGGING_PREFIXES = ['', ' + ', '    - ']
LOGGING_PREFIXES_WARN = ['', ' ! ', '    ! ']
LOGGING_PREFIXES_EMPTY = ['', '   ', '     ']

class UpdatedBatchNormalization(tf.keras.layers.BatchNormalization):
	def call(self, inputs, training=None):
		true_phase = int(K.get_session().run(K.learning_phase()))
		trainable = int(self.trainable)
		with K.learning_phase_scope(trainable * true_phase):
			return super(tf.keras.layers.BatchNormalization, self).call(inputs, training)

def global_path(path_string):
	if not PROJECT_DIR:
		print("ERROR: No project loaded.")
		sys.exit()
	if path_string and (len(path_string) > 2) and path_string[:2] == "./":
		return os.path.join(PROJECT_DIR, path_string[2:])
	elif path_string and (path_string[0] != "/"):
		return os.path.join(PROJECT_DIR, path_string)
	else:
		return path_string		

def warn(text):
	return WARNING + str(text) + ENDC

def header(text):
	return HEADER + str(text) + ENDC

def info(text):
	return BLUE + str(text) + ENDC

def green(text):
	return GREEN + str(text) + ENDC

def fail(text):
	return FAIL + str(text) + ENDC

def bold(text):
	return BOLD + str(text) + ENDC

def underline(text):
	return UNDERLINE + str(text) + ENDC

class LOGGING_LEVEL:
	INFO = 3
	WARN = 3
	ERROR = 3
	COMPLETE = 3
	SILENT = False

class Logger:
	logfile = None
	def __init__(self):
		pass
	def info(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{info('INFO')}] {text}"
		if print_func and l <= LOGGING_LEVEL.INFO and not LOGGING_LEVEL.SILENT:
			print_func(message)
		self.log(message)
		return message
	def warn(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{warn('WARN')}] {text}"
		if print_func and l <= LOGGING_LEVEL.WARN:
			print_func(message)
		self.log(message)
		return message
	def error(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{fail('ERROR')}] {text}"
		if print_func and l <= LOGGING_LEVEL.ERROR:
			print_func(message)
		self.log(message)
		return message
	def complete(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{header('Complete')}] {text}"
		if print_func and l <= LOGGING_LEVEL.COMPLETE and not LOGGING_LEVEL.SILENT:
			print_func(message)
		self.log(message)
		return message
	def label(self, label, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{green(label)}] {text}"
		if print_func and l <= LOGGING_LEVEL.INFO and not LOGGING_LEVEL.SILENT:
			print_func(message)
		self.log(message)
		return message
	def empty(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}{text}"
		if print_func and l <= LOGGING_LEVEL.INFO and not LOGGING_LEVEL.SILENT:
			print_func(message)
		self.log(message)
		return message
	def header(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"\n{LOGGING_PREFIXES_EMPTY[l]}{bold(text)}"
		if print_func and not LOGGING_LEVEL.SILENT:
			print_func(message)
		self.log(message)
		return message
	def log(self, text):
		st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
		if self.logfile:
			for s in FORMATTING_OPTIONS:
				text = text.replace(s, "")
			outfile = open(self.logfile, 'a')
			outfile.write(f"[{st}] {text.strip()}\n")
			outfile.close()

log = Logger()

class TCGA:
	patient = 'submitter_id'
	project = 'project_id'
	slide = 'slide'

def _shortname(string):
	if len(string) == 60:
		# May be TCGA slide with long name; convert to patient name by returning first 12 characters
		return string[:12]
	else:
		return string

def yes_no_input(prompt, default='no'):
	yes = ['yes','y']
	no = ['no', 'n']
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return True if default in yes else False
		if response.lower() in yes:
			return True
		if response.lower() in no:
			return False
		print(f"Invalid response.")

def dir_input(prompt, default=None, create_on_invalid=False, absolute=False):
	while True:
		if not absolute:
			response = global_path(input(f"{prompt}"))
		else:
			response = input(f"{prompt}")
		if not response and default:
			response = global_path(default)
		if not os.path.exists(response) and create_on_invalid:
			if yes_no_input(f'Directory "{response}" does not exist. Create directory? [Y/n] ', default='yes'):
				os.makedirs(response)
				return response
			else:
				continue
		elif not os.path.exists(response):
			print(f'Unable to locate directory "{response}"')
			continue
		return response

def file_input(prompt, default=None, filetype=None, verify=True):
	while True:
		response = global_path(input(f"{prompt}"))
		if not response and default:
			response = global_path(default)
		if verify and not os.path.exists(response):
			print(f'Unable to locate file "{response}"')
			continue
		extension = path_to_ext(response)
		if filetype and (extension != filetype):
			print(f'Incorrect filetype; provided file of type "{extension}", need type "{filetype}"')
			continue
		return response

def int_input(prompt, default=None):
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		try:
			int_response = int(response)
		except ValueError:
			print("Please supply a valid number.")
			continue
		return int_response

def float_input(prompt, default=None, valid_range=None):
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		try:
			float_response = float(response)
		except ValueError:
			print("Please supply a valid number.")
			continue
		if valid_range and not (float_response >= valid_range[0] and float_response <= valid_range[1]):
			print(f"Please supply a valid numer in the range {valid_range[0]} to {valid_range[1]}")
		return float_response

def choice_input(prompt, valid_choices, default=None, multi_choice=False):
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		if not multi_choice and response not in valid_choices:
			print("Invalid option.")
			continue
		elif multi_choice:
			response = response.replace(" ", "").split(',')
			invalid = [r not in valid_choices for r in response]
			if any(invalid):
				print(f'Invalid selection (response: {response})')
				continue
		return response

def load_json(filename):
	with open(filename, 'r') as data_file:
		return json.load(data_file)

def write_json(data, filename):
	with open(filename, "w") as data_file:
		json.dump(data, data_file)

def _parse_function(example_proto):
	feature_description = {'slide':     tf.io.FixedLenFeature([], tf.string),
						   'image_raw':	tf.io.FixedLenFeature([], tf.string)}
	return tf.io.parse_single_example(example_proto, feature_description)

def get_slide_paths(slides_dir):
	num_dir = len(slides_dir.split('/'))
	slide_list = [i for i in glob(join(slides_dir, '**/*.jpg'))
					if i.split('/')[num_dir] != 'thumbs']
	
	for filetype in SUPPORTED_FORMATS:
		slide_list.extend( [i for i in glob(join(slides_dir, f'**/*.{filetype}'))
							if i.split('/')[num_dir] != 'thumbs'] )

		slide_list.extend(glob(join(slides_dir, f'*.{filetype}')))

	return slide_list

def filter_slide_paths(slide_list, filters, filter_blank=[]):
	filtered_slide_names = get_slides_from_annotations(filters=filters, filter_blank=filter_blank)
	filtered_slide_list = [slide for slide in slide_list if path_to_name(slide) in filtered_slide_names]
	return filtered_slide_list

def filter_tfrecords_paths(tfrecords_list, filters, filter_blank=[]):
	filtered_slide_names = get_slides_from_annotations(filters=filters, filter_blank=filter_blank)
	filtered_tfrecords_list = [tfrecord for tfrecord in tfrecords_list if tfrecord.split('/')[-1][:-10] in filtered_slide_names]
	return filtered_tfrecords_list

def get_slides_from_annotations(filters=None, filter_blank=[]):
	'''Returns a list of slide names from the annotations file using a given set of filters.'''
	global ANNOTATIONS
	result = []
	filter_blank = [filter_blank] if type(filter_blank) != list else filter_blank
	patient_slide_dict = {}
	if not len(ANNOTATIONS):
		log.error("No annotations loaded; is the annotations file empty?")
	for ann in ANNOTATIONS:
		skip_annotation = False
		if TCGA.slide not in ann.keys():
			log.error(f"{TCGA.slide} not found in annotations file.")
			sys.exit()

		# Skip missing or blank slides
		if ann[TCGA.slide] in SLIDE_ANNOTATIONS_TO_IGNORE:
			continue

		# Ensure slides are only assigned to a single patient
		if ann[TCGA.patient] not in patient_slide_dict:
			patient_slide_dict.update({ann[TCGA.patient]: ann[TCGA.slide]})
		elif patient_slide_dict[ann[TCGA.patient]] != ann[TCGA.slide]:
			log.error(f"Multiple patients assigned to slide {green(ann[TCGA.slide])}.")
			sys.exit()

		# Only return slides with annotation values specified in "filters"
		if filters:
			for filter_key in filters.keys():
				if filter_key not in ann.keys():
					log.error(f"Filter header {filter_key} not found in annotations file.")
					sys.exit()
				if    ((type(filters[filter_key]) == list and ann[filter_key] not in filters[filter_key]) 
					or (type(filters[filter_key]) != list and filters[filter_key] != ann[filter_key])):
					skip_annotation = True
					break

		# Filter out slides that are blank in a given annotation column ("filter_blank")
		if filter_blank:
			for fb in filter_blank:
				if fb not in ann.keys():
					log.error(f"Unable to filter blank slides from header {fb}; this header was not found in the annotations file.")
					sys.exit()
				if not ann[fb] or ann[fb] == '':
					skip_annotation = True
					break
		if skip_annotation: continue
		result += [ann[TCGA.slide]]
	return result

def get_outcomes_from_annotations(headers, filters=None, filter_blank=[], use_float=False):
	'''Returns a dictionary of slide names mapping to patient id and [an] outcome variable(s).

	Args:
		headers			annotation header(s) that specifies outcome variable. May be a list.
		filters			dictionary of filters to use when selecting slides from annotations file
		filter_blank	will filter out slides that are blank in the specified column(s)
		use_float		If true, will try to convert data into float

	Returns:
		Dictionary with slides as keys and dictionaries as values. The value dictionaries contain the following keys/values:
			TCGA.patient	patient name
			outcome			if only one header is supplied, this is a single value containing the processed outcome for this slide
								if multiple headers were supplied, this is a list of processed outcomes, one for each header
								if use_float is specified, the value will always be a list
	'''
	global ANNOTATIONS
	slides = get_slides_from_annotations(filters, filter_blank)
	filtered_annotations = [a for a in ANNOTATIONS if a[TCGA.slide] in slides]
	results = {}
	headers = [headers] if type(headers) != list else headers
	filter_blank = [filter_blank] if type(filter_blank) != list else filter_blank
	assigned_headers = {}
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
				log.error(f"Unable to convert outcome {header} into type 'float'.", 1)
				sys.exit()
		else:
			log.info(f'Assigning outcome descriptors in column "{header}" to numerical values', 1)
			unique_outcomes = list(set(filtered_outcomes))
			unique_outcomes.sort()
			for i, uo in enumerate(unique_outcomes):
				num_matching_slides_filtered = sum(o == uo for o in filtered_outcomes)
				log.empty(f"{header} '{info(uo)}' assigned to value '{i}' [{bold(str(num_matching_slides_filtered))} slides]", 2)
		
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
	return results

def update_annotations_with_slidenames(annotations_file, dataset):
	'''Attempts to automatically associate slide names from a directory with patients in a given annotations file.'''
	header, _ = read_annotations(annotations_file)
	slide_list = dataset.get_slide_paths()

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
		slide_name = path_to_name(slide_filename)
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
			log.warn(f"Slide '{slide_name}' not found in annotations file, skipping.", 1, print_func)
			num_warned += 1
	if num_warned >= warn_threshold:
		log.warn(f"...{num_warned} total warnings, see {green(log.logfile)} for details", 1)

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
					if patient in patient_slide_dict:
						row[slide_index] = patient_slide_dict[patient]
						num_updated_annotations += 1
					else:
						row[slide_index] = ''
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

def read_annotations(annotations_file):
	results = []
	# Open annotations file and read header
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		# First, try to open file
		try:
			header = next(csv_reader, None)
		except OSError:
			log.error(f"Unable to open annotations file {green(annotations_file)}, is it open in another program?")
			sys.exit()

		for row in csv_reader:
			row_dict = {}
			for i, key in enumerate(header):
				row_dict[key] = row[i]
			results += [row_dict]
	return header, results
			
def load_annotations(annotations_file, dataset):
	global ANNOTATIONS
	# Verify annotations file exists
	if not os.path.exists(annotations_file):
		log.error(f"Annotations file {green(annotations_file)} does not exist, unable to load")
		sys.exit()

	header, current_annotations = read_annotations(annotations_file)

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
		if dataset and yes_no_input('\nSearch slides directory and automatically associate patients with slides? [Y/n] ', default='yes'):
			update_annotations_with_slidenames(annotations_file, dataset)
			header, current_annotations = read_annotations(annotations_file)
		else:
			sys.exit()
	ANNOTATIONS = current_annotations

def verify_annotations_slides(dataset):
	global ANNOTATIONS
	slide_list = dataset.get_slide_paths()

	# Verify no duplicate slide names are found
	slide_list_from_annotations = get_slides_from_annotations()
	if len(slide_list_from_annotations) != len(list(set(slide_list_from_annotations))):
		log.error("Duplicate slide names detected in the annotation file.")
		sys.exit()

	# Verify all SVS files in the annotation column are valid
	num_warned = 0
	warn_threshold = 3
	for annotation in ANNOTATIONS:
		print_func = print if num_warned < warn_threshold else None
		slide = annotation[TCGA.slide]
		if slide == '':
			log.warn(f"Patient {green(annotation[TCGA.patient])} has no slide assigned.", 1, print_func)
			num_warned += 1
		elif not slide in [path_to_name(s) for s in slide_list]:
			log.warn(f"Unable to locate slide {slide}", 1, print_func)
			num_warned += 1
	if num_warned >= warn_threshold:
		log.warn(f"...{num_warned} total warnings, see {green(log.logfile)} for details", 1)
	if not num_warned:
		log.info(f"Slides successfully verified, no errors found.", 1)

def get_relative_tfrecord_paths(root, directory=""):
	tfrecords = [join(directory, f) for f in os.listdir(join(root, directory)) if (not isdir(join(root, directory, f)) and len(f) > 10 and f[-10:] == ".tfrecords")]
	subdirs = [f for f in os.listdir(join(root, directory)) if isdir(join(root, directory, f))]
	for sub in subdirs:
		tfrecords += get_relative_tfrecord_paths(root, join(directory, sub))
	return tfrecords

def get_global_manifest(directory):
	'''Loads a saved relative manifest at a directory and returns a dict containing
	absolute/global path and file names.'''
	manifest_path = join(directory, "manifest.json")
	if not exists(manifest_path):
		log.error(f"No manifest file detected in {directory}; will create now")
		update_tfrecord_manifest(directory)
	relative_manifest = load_json(manifest_path)
	global_manifest = {}
	for record in relative_manifest:
		global_manifest.update({join(directory, record): relative_manifest[record]})
	return global_manifest

def update_tfrecord_manifest(directory, force_update=False):
	'''Log number of tiles in each TFRecord file present in the given directory and all subdirectories, 
	saving manifest to file within the parent directory.
	
	Additionally, verify all TFRecords have an associated annotation.'''
	global ANNOTATIONS
	slide_list = []
	manifest_path = join(directory, "manifest.json")
	manifest = {} if not exists(manifest_path) else load_json(manifest_path)
	try:
		relative_tfrecord_paths = get_relative_tfrecord_paths(directory)
	except FileNotFoundError:
		log.warn(f"Unable to find TFRecords in the directory {directory}")
		return
	slide_names_from_annotations = get_slides_from_annotations()

	slide_list_errors = []
	for rel_tfr in relative_tfrecord_paths:
		tfr = join(directory, rel_tfr)

		if (not force_update) and (rel_tfr in manifest) and ('total' in manifest[rel_tfr]):
			continue

		manifest.update({rel_tfr: {}})
		raw_dataset = tf.data.TFRecordDataset(tfr)
		sys.stdout.write(f"\r + Verifying tiles in {green(rel_tfr)}...")
		sys.stdout.flush()
		total = 0
		for raw_record in raw_dataset:
			example = tf.train.Example()
			example.ParseFromString(raw_record.numpy())
			slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')
			slide_list.extend([slide])
			slide_list = list(set(slide_list))
			if slide not in manifest[rel_tfr]:
				manifest[rel_tfr][slide] = 1
			else:
				manifest[rel_tfr][slide] += 1
			if slide not in slide_names_from_annotations:
				slide_list_errors.extend([slide])
				slide_list_errors = list(set(slide_list_errors))
			total += 1
		manifest[rel_tfr]['total'] = total

	for slide in slide_list_errors:
		log.error(f"Failed TFRecord integrity check: annotation not found for slide {green(slide)}", 1)

	sys.stdout.write("\r\033[K")
	sys.stdout.flush()

	# Write manifest file
	write_json(manifest, manifest_path)

	# Now, check to see if all annotations have a corresponding set of tiles
	if len(ANNOTATIONS):
		num_warned = 0
		warn_threshold = 3
		for annotation in ANNOTATIONS:
			print_func = print if num_warned < warn_threshold else None
			if annotation[TCGA.slide] == '':
				log.warn(f"Patient {green(annotation[TCGA.patient])} has no slide assigned.", 1, print_func)
				num_warned += 1
			elif annotation[TCGA.slide] not in slide_list:
				log.warn(f"Slide {green(annotation[TCGA.slide])} in annotation file has no image tiles.", 1, print_func)
				num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {green(log.logfile)} for details", 1)

	return manifest
	
def contains_nested_subdirs(directory):
	subdirs = [_dir for _dir in os.listdir(directory) if isdir(join(directory, _dir))]
	for subdir in subdirs:
		contents = os.listdir(join(directory, subdir))
		for c in contents:
			if isdir(join(directory, subdir, c)):
				return True
	return False

def path_to_name(path):
	_file = path.split('/')[-1]
	if len(_file.split('.')) == 1:
		return _file
	else:
		return '.'.join(_file.split('.')[:-1])

def path_to_ext(path):
	_file = path.split('/')[-1]
	if len(_file.split('.')) == 1:
		return ''
	else:
		return _file.split('.')[-1]