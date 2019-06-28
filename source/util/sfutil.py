import sys
import json
import csv

import os
from os.path import join
from glob import glob
import shutil

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

LOGGING_PREFIXES = ['', ' + ', '    - ']
LOGGING_PREFIXES_WARN = ['', ' ! ', '    ! ']

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
	LABEL = INFO
	EMPTY = INFO

class log:
	def info(text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{info('INFO')}] {text}"
		if print_func and l <= LOGGING_LEVEL.INFO:
			print_func(message)
		return message
	def warn(text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{warn('WARN')}] {text}"
		if print_func and l <= LOGGING_LEVEL.WARN:
			print_func(message)
		return message
	def error(text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{fail('ERROR')}] {text}"
		if print_func and l <= LOGGING_LEVEL.ERROR:
			print_func(message)
		return message
	def complete(text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{header('Complete')}] {text}"
		if print_func and l <= LOGGING_LEVEL.COMPLETE:
			print_func(message)
		return message
	def label(label, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{green(label)}] {text}"
		if print_func and l <= LOGGING_LEVEL.LABEL:
			print_func(message)
		return message
	def empty(text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]} {text}"
		if print_func and l <= LOGGING_LEVEL.EMPTY:
			print_func(message)
		return message

class UpdatedBatchNormalization(tf.keras.layers.BatchNormalization):
	def call(self, inputs, training=None):
		true_phase = int(K.get_session().run(K.learning_phase()))
		trainable = int(self.trainable)
		with K.learning_phase_scope(trainable * true_phase):
			return super(tf.keras.layers.BatchNormalization, self).call(inputs, training)

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

def global_path(path_string):
	if not PROJECT_DIR:
		log.error("No project loaded.", 1)
		sys.exit()
	if path_string and (len(path_string) > 2) and path_string[:2] == "./":
		return os.path.join(PROJECT_DIR, path_string[2:])
	elif path_string and (path_string[0] != "/"):
		return os.path.join(PROJECT_DIR, path_string)
	else:
		return path_string		

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

	slide_list.extend(glob(join(slides_dir, '*.svs')))
	slide_list.extend(glob(join(slides_dir, '*.jpg')))
	return slide_list

def get_filtered_slide_paths(slides_dir, annotations_file, filter_header, filter_values):
	slide_list = get_slide_paths(slides_dir)
	filtered_annotation_dict = get_annotations_dict(annotations_file, TCGAAnnotations.slide, TCGAAnnotations.case, filter_header=filter_header, filter_values=filter_values)
	filtered_slide_names = list(filtered_annotation_dict.keys())
	filtered_slide_list = [slide for slide in slide_list if slide.split('/')[-1][:-4] in filtered_slide_names]
	return filtered_slide_list

def get_annotations_dict(annotations_file, key_name, value_name, filter_header=None, filter_values=None, use_encode=True):
	if filter_header and not filter_values:
		log.error("If supplying a filter header, you must also supply filter_values")
		sys.exit() 
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader, None)
		encode = False
		key_dict_int = {}
		key_dict_str = {}
		try:
			value_index = header.index(value_name)
			key_index = header.index(key_name)
			if filter_header: filter_index = header.index(filter_header)
		except:
			column_names = f'"{key_name}", "{value_name}"'
			if filter_header: column_names += f', "{filter_header}"'
			log.error(f"Unable to find columns {column_names} in annotation file", 1)
			
			sys.exit()
		for row in csv_reader:
			value = row[value_index]
			key = row[key_index]
			if key in key_dict_str.keys():
				log.error(f"Multiple values of '{key}' found in annotation column '{key_name}'", 1)
				sys.exit()
			if filter_header:
				filter_value = row[filter_index]
				# Check if this slide should be skipped
				if (type(filter_values)==str and filter_value!=filter_values) or filter_value not in filter_values:
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
		if use_encode and encode:
			values = list(set(key_dict_str.values()))
			values.sort()
			values_str_to_int = {}
			for i, c in enumerate(values):
				values_str_to_int.update({c: i})
				log.empty(f"{value_name} '{info(c)}' assigned to value '{i}'", 2)
			for value_string in key_dict_str.keys():
				key_dict_str[value_string] = values_str_to_int[key_dict_str[value_string]]
			return key_dict_str
		elif use_encode:
			return key_dict_int
		else:
			return key_dict_str

def verify_annotations(annotations_file, slides_dir=None):
	slide_list = get_slide_paths(slides_dir)
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		# First, verify case, category, and slide headers exist
		header = next(csv_reader, None)
	try:
		case_index = header.index(TCGAAnnotations.case)
		category_index = header.index('category')
	except:
		log.error(f"Check annotations file for headers '{TCGAAnnotations.case}' and 'category'.", 1)
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
			
			for slide_filename in slide_list:
				slide_name = slide_filename.split('/')[-1][:-4]
				# First, make sure the shortname and long name aren't both in the annotation file
				if (slide_name != _shortname(slide_name)) and (slide_name in cases) and (_shortname(slide_name) in cases):
					log.error(f"Both slide name {slide_name} and shorthand {_shortname(slide_name)} in annotation file; please remove one.", 1)
					sys.exit()
				# Check if either the slide name or the shortened version are in the annotation file
				if any(x in cases for x in [slide_name, _shortname(slide_name)]):
					slide = slide_name if slide_name in cases else _shortname(slide_name)
					case_slide_dict.update({slide: slide_name})
				else:
					if not yes_no_input(f" + [{warn('WARN')}] Case '{_shortname(slide_name)}' not found in annotation file, skip this slide? [Y/n] ", default='yes'):
						sys.exit()
			# Now, write the assocations
			with open(annotations_file) as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=',')
				header = next(csv_reader, None)
				with open('temp.csv', 'w') as csv_outfile:
					csv_writer = csv.writer(csv_outfile, delimiter=',')
					header.extend([TCGAAnnotations.slide])
					csv_writer.writerow(header)
					for row in csv_reader:
						case = row[case_index]
						row.extend([case_slide_dict[case]])
						csv_writer.writerow(row)
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
		for row in csv_reader:
			if not row[slide_index] in [s.split('/')[-1][:-4] for s in slide_list]:
				if not skip_warn and yes_no_input(f" + [{warn('WARN')}] Unable to locate slide {row[slide_index]}. Quit? [y/N] ", default='no'):
					sys.exit()
				else:
					log.warn(f"Unable to locate slide {row[slide_index]}", 1)
					skip_warn = True

def verify_tiles(annotations, input_dir, tfrecord_files=[]):
	'''Iterate through folders if using raw images and verify all have an annotation;
	if using TFRecord, iterate through all records and verify all entries for valid annotation.
	
	Additionally, generate a manifest to log the number of tiles for each slide.'''
	print(f" + Verifying tiles and annotations...")
	success = True
	case_list = []
	manifest = {'train_data': {},
				'eval_data': {}}
	if tfrecord_files:
		case_list_errors = []
		for tfrecord_file in tfrecord_files:
			manifest.update({tfrecord_file: {}})
			raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
			for i, raw_record in enumerate(raw_dataset):
				sys.stdout.write(f"\r + Verifying tile...{i}")
				sys.stdout.flush()
				example = tf.train.Example()
				example.ParseFromString(raw_record.numpy())
				case = example.features.feature['case'].bytes_list.value[0].decode('utf-8')
				case_list.extend([case])
				case_list = list(set(case_list))
				if case not in manifest[tfrecord_file]:
					manifest[tfrecord_file][case] = 1
				else:
					manifest[tfrecord_file][case] += 1
				if case not in annotations:
					case_list_errors.extend([case])
					case_list_errors = list(set(case_list_errors))
					success = False
			total = 0
			for case in manifest[tfrecord_file].keys():
				total += manifest[tfrecord_file][case]
			manifest[tfrecord_file]['total'] = total
		for case in case_list_errors:
			log.error(f"Failed TFRecord integrity check: annotation not found for case {green(case)}", 1)
	else:
		manifest['total_train_tiles'] = len(glob(os.path.join(input_dir, "train_data/**/*.jpg")))
		train_case_list = [i.split('/')[-1] for i in glob(os.path.join(input_dir, "train_data/*"))]
		eval_case_list = [i.split('/')[-1] for i in glob(os.path.join(input_dir, "eval_data/*"))]
		for case in train_case_list:
			manifest['train_data'][case] = len(glob(os.path.join(input_dir, f"train_data/{case}/*.jpg")))
		for case in eval_case_list:
			manifest['eval_data'][case] = len(glob(os.path.join(input_dir, f"eval_data/{case}/*.jpg")))
		case_list = list(set(train_case_list + eval_case_list))
		for case in case_list:
			if case not in annotations:
				log.error(f"Failed image tile integrity check: annotation not found for case {green(case)}", 1)
				success = False
	print(f" ...complete.")
	# Now, check to see if all annotations have a corresponding set of tiles
	for annotation_case in annotations.keys():
		if annotation_case not in case_list:
			log.warn(f"Case {green(annotation_case)} in annotation file has no image tiles", 2)
	if not success:
		sys.exit()
	else:
		return manifest
