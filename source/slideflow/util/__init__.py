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
	INFO = 0
	WARN = 3
	ERROR = 3
	COMPLETE = 3

class Logger:
	logfile = None
	def __init__(self):
		pass
	def info(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{info('INFO')}] {text}"
		if print_func and l <= LOGGING_LEVEL.INFO:
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
		if print_func and l <= LOGGING_LEVEL.COMPLETE:
			print_func(message)
		self.log(message)
		return message
	def label(self, label, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{green(label)}] {text}"
		if print_func and l <= LOGGING_LEVEL.INFO:
			print_func(message)
		self.log(message)
		return message
	def empty(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}{text}"
		if print_func and l <= LOGGING_LEVEL.INFO:
			print_func(message)
		self.log(message)
		return message
	def header(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"\n{LOGGING_PREFIXES_EMPTY[l]}{bold(text)}"
		if print_func:
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

class TCGAAnnotations:
	case = 'submitter_id'
	project = 'project_id'
	slide = 'slide'

def _shortname(string):
	if len(string) == 60:
		# May be TCGA SVS file with long name; convert to case name by returning first 12 characters
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
				os.mkdir(response)
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
		extension = response.split('.')[-1]
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

def choice_input(prompt, valid_choices, default=None):
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		if response not in valid_choices:
			print("Invalid option.")
			continue
		return response

def load_json(filename):
	with open(filename, 'r') as data_file:
		return json.load(data_file)

def write_json(data, filename):
	with open(filename, "w") as data_file:
		json.dump(data, data_file)

def _parse_function(example_proto):
	feature_description = {'category': tf.io.FixedLenFeature([], tf.int64),
						   'case':     tf.io.FixedLenFeature([], tf.string),
						   'image_raw':tf.io.FixedLenFeature([], tf.string)}
	return tf.io.parse_single_example(example_proto, feature_description)

def get_slide_paths(slides_dir):
	num_dir = len(slides_dir.split('/'))
	slide_list = [i for i in glob(join(slides_dir, '**/*.svs'))
					if i.split('/')[num_dir] != 'thumbs']

	slide_list.extend( [i for i in glob(join(slides_dir, '**/*.jpg'))
						if i.split('/')[num_dir] != 'thumbs'] )

	slide_list.extend( [i for i in glob(join(slides_dir, '**/*.tiff'))
						if i.split('/')[num_dir] != 'thumbs'] )

	slide_list.extend(glob(join(slides_dir, '*.svs')))
	slide_list.extend(glob(join(slides_dir, '*.jpg')))
	slide_list.extend(glob(join(slides_dir, '*.tiff')))
	return slide_list

def get_filtered_slide_paths(slides_dir, annotations_file, filter_header, filter_values):
	slide_list = get_slide_paths(slides_dir)
	if not os.path.exists(annotations_file):
		log.error(f"Unable to find annotations file at {green(annotations_file)}; unable to filter slides.")
		return slide_list
	filtered_annotation_dict = get_annotations_dict(annotations_file, TCGAAnnotations.slide, TCGAAnnotations.case, filter_header=filter_header, filter_values=filter_values, use_encode=False)
	filtered_slide_names = list(filtered_annotation_dict.keys())
	filtered_slide_list = [slide for slide in slide_list if slide.split('/')[-1].split(".")[0] in filtered_slide_names]
	return filtered_slide_list

def get_filtered_tfrecords_paths(tfrecords_dir, annotations_file, filter_header, filter_values):
	tfrecord_list = glob(join(tfrecords_dir, "*.tfrecords"))
	filtered_annotation_dict = get_annotations_dict(annotations_file, TCGAAnnotations.slide, TCGAAnnotations.case, filter_header=filter_header, filter_values=filter_values, use_encode=False)
	filtered_slide_names = list(filtered_annotation_dict.keys())
	filtered_tfrecord_list = [tfrecord for tfrecord in tfrecord_list if tfrecord.split('/')[-1][:-10] in filtered_slide_names]
	return filtered_tfrecord_list

def get_annotations_dict(annotations_file, key_name, value_name, filter_header=None, filter_values=None, use_encode=True, use_float=False):
	if filter_header and not filter_values:
		log.error("If supplying a filter header, you must also supply filter_values")
		sys.exit() 
	if type(filter_header) != list and filter_header:
		filter_header = [filter_header]
		filter_values = [filter_values]
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader, None)
		encode = False
		key_dict_int = {}
		key_dict_str = {}
		try:
			value_index = header.index(value_name)
			key_index = header.index(key_name)
			#if filter_header: filter_index = header.index(filter_header)
			if filter_header:
				filter_indices = [header.index(name) for name in filter_header]
		except:
			column_names = f'"{key_name}", "{value_name}"'
			 
			if filter_header:
				header_names = ", ".join(filter_header) 
				column_names += f', "{header_names}"'
			log.error(f"Unable to find columns {column_names} in annotation file; confirm file format is correct and headers exist", 1)
			sys.exit()
		for row in csv_reader:
			value = row[value_index]
			key = row[key_index]
			if key in key_dict_str.keys():
				log.error(f"Multiple values of '{key}' found in annotation column '{key_name}'", 1)
				sys.exit()
			if filter_header:
				should_skip = False
				for i in range(len(filter_header)):
					observed_value = row[filter_indices[i]]
					# Check if this slide should be skipped
					if (type(filter_values[i])==str and observed_value!=filter_values[i]) or observed_value not in filter_values[i]:
						should_skip = True
				if should_skip:
					continue
			key_dict_str.update({key: value})
			if use_encode:
				try:
					int_value = int(value)
					key_dict_int.update({key: int_value})
				except:
					if use_encode and not encode:
						log.info(f"Non-integer in '{value_name}' header, encoding with integer values", 1)
						encode = True

		# Raise an error if no tiles were selected based on criteria
		if not len(key_dict_str):
			log.error(f"No tiles were selected based on filtering criteria.")
			sys.exit()

		if use_encode and encode:
			values = list(set(key_dict_str.values()))
			values.sort()
			values_str_to_int = {}
			for i, c in enumerate(values):
				values_str_to_int.update({c: i})
				number_of_slides = sum(x == c for x in key_dict_str.values())
				log.empty(f"{value_name} '{info(c)}' assigned to value '{i}' [{number_of_slides} slides]", 2)
			for value_string in key_dict_str.keys():
				key_dict_str[value_string] = values_str_to_int[key_dict_str[value_string]]
			return key_dict_str
		elif use_encode:
			return key_dict_int
		elif use_float:
			for key in key_dict_str:
				try:
					key_dict_str[key] = float(key_dict_str[key])
				except:
					log.error(f'Unable to convert data ("{key_dict_str[key]}") into float; please check data integrity and chosen annotation column')
					sys.exit()
			return key_dict_str
		else:
			return key_dict_str

def verify_annotations(annotations_file, slides_dir=None):
	if not os.path.exists(annotations_file):
		log.error(f"Annotations file {green(annotations_file)} does not exist, unable to verify")
		return
	slide_list = get_slide_paths(slides_dir)
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		# First, verify case, category, and slide headers exist
		try:
			header = next(csv_reader, None)
		except OSError:
			log.error(f"Unable to open annotations file {green(annotations_file)}, is it open in another program?")
			return
	try:
		case_index = header.index(TCGAAnnotations.case)
	except:
		log.error(f"Check annotations file for header '{TCGAAnnotations.case}'.", 1)
		sys.exit()
	try:
		slide_index = header.index(TCGAAnnotations.slide)
	except:
		log.error(f"Header column 'slide' not found.", 1)
		if slides_dir and yes_no_input('\nSearch slides directory and automatically associate cases with slides? [Y/n] ', default='yes'):
			# First, load all case names from the annotations file
			slide_index = len(header)
			cases = []
			case_slide_dict = {}
			with open(annotations_file) as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=',')
				header = next(csv_reader, None)
				for row in csv_reader:
					cases.extend([row[case_index]])
			cases = list(set(cases)) # remove duplicates
			# Next, search through the slides folder for all SVS/JPG files
			print(f" + Searching {slides_dir}...")
			skip_missing = False
			for slide_filename in slide_list:
				slide_name = slide_filename.split('/')[-1].split(".")[0]
				# First, make sure the shortname and long name aren't both in the annotation file
				if (slide_name != _shortname(slide_name)) and (slide_name in cases) and (_shortname(slide_name) in cases):
					log.error(f"Both slide name {slide_name} and shorthand {_shortname(slide_name)} in annotation file; please remove one.", 1)
					sys.exit()
				# Check if either the slide name or the shortened version are in the annotation file
				if any(x in cases for x in [slide_name, _shortname(slide_name)]):
					slide = slide_name if slide_name in cases else _shortname(slide_name)
					case_slide_dict.update({slide: slide_name})
				elif not skip_missing:
					if not yes_no_input(f" + [{warn('WARN')}] Case '{_shortname(slide_name)}' not found in annotation file, skip this slide? [Y/n] ", default='yes'):
						sys.exit()
					else:
						skip_missing = True
			
			# Now, write the assocations
			with open(annotations_file) as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=',')
				header = next(csv_reader, None)
				with open('temp.csv', 'w') as csv_outfile:
					csv_writer = csv.writer(csv_outfile, delimiter=',')
					header.extend([TCGAAnnotations.slide])
					csv_writer.writerow(header)
					for row in csv_reader:
						try:
							case = row[case_index]
							row.extend([case_slide_dict[case]])
							csv_writer.writerow(row)
						except KeyError:
							log.error(f"Unable to locate slide {case} in annotation file")
			# Finally, overwrite the previous annotation file
			os.remove(annotations_file)
			shutil.move('temp.csv', annotations_file)
		else:
			sys.exit()

	# Verify all SVS files in the annotation column are valid
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader, None)
		skip_warn = False
		num_warned = 0
		warn_threshold = 3
		for row in csv_reader:
			if not row[slide_index] in [s.split('/')[-1].split(".")[0] for s in slide_list]:
				if not skip_warn and yes_no_input(f" + [{warn('WARN')}] Unable to locate slide {row[slide_index]}. Quit? [y/N] ", default='no'):
					sys.exit()
				else:
					print_func = print if num_warned < warn_threshold else None
					log.warn(f"Unable to locate slide {row[slide_index]}", 1, print_func)
					skip_warn = True
					num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {green(log.logfile)} for details", 1)

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

def update_tfrecord_manifest(directory, annotations=None, force_update=False):
	'''Log number of tiles in each TFRecord file present in the given directory and all subdirectories, 
	saving manifest to file within the parent directory.
	
	Additionally, if annotations are provided, verify all TFRecords have a valid annotation.'''

	case_list = []
	manifest_path = join(directory, "manifest.json")
	manifest = {} if not exists(manifest_path) else load_json(manifest_path)
	relative_tfrecord_paths = get_relative_tfrecord_paths(directory)

	case_list_errors = []
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
			case = example.features.feature['case'].bytes_list.value[0].decode('utf-8')
			case_list.extend([case])
			case_list = list(set(case_list))
			if case not in manifest[rel_tfr]:
				manifest[rel_tfr][case] = 1
			else:
				manifest[rel_tfr][case] += 1
			if annotations and case not in annotations:
				case_list_errors.extend([case])
				case_list_errors = list(set(case_list_errors))
				success = False
			total += 1
		manifest[rel_tfr]['total'] = total

	for case in case_list_errors:
		log.error(f"Failed TFRecord integrity check: annotation not found for case {green(case)}", 1)

	sys.stdout.write("\r\033[K")
	sys.stdout.flush()

	# Write manifest file
	write_json(manifest, manifest_path)

	# Now, check to see if all annotations have a corresponding set of tiles
	if annotations:
		num_warned = 0
		warn_threshold = 3
		for annotation_case in annotations.keys():
			if annotation_case not in case_list:
				print_func = print if num_warned < warn_threshold else None
				log.warn(f"Case {green(annotation_case)} in annotation file has no image tiles", 2, print_func)
				num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {green(log.logfile)} for details", 2)

	return manifest
	
def contains_nested_subdirs(directory):
	subdirs = [_dir for _dir in os.listdir(directory) if isdir(join(directory, _dir))]
	for subdir in subdirs:
		contents = os.listdir(join(directory, subdir))
		for c in contents:
			if isdir(join(directory, subdir, c)):
				return True
	return False