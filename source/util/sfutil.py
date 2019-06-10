import sys
import json
import csv

import os
from os.path import join
from glob import glob
import shutil

import tensorflow as tf

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

class TCGAAnnotations:
	case = 'submitter_id'
	project = 'project_id'
	slide = 'slide'

def warn(text):
	return WARNING + text + ENDC

def header(text):
	return HEADER + text + ENDC

def info(text):
	return BLUE + text + ENDC

def green(text):
	return GREEN + text + ENDC

def fail(text):
	return FAIL + text + ENDC

def bold(text):
	return BOLD + text + ENDC

def underline(text):
	return UNDERLINE + text + ENDC

def _shortname(string):
	if len(string) == 60:
		# May be TCGA SVS file with long name; convert to case name by returning first 12 characters
		return string[:12]
	else:
		return string

def create_global_path(path_string):
	if not PROJECT_DIR:
		print(f" + [{fail('ERROR')}] No project loaded.")
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

def dir_input(prompt, default=None, create_on_invalid=False):
	while True:
		response = create_global_path(input(f"{prompt}"))
		if not response and default:
			response = create_global_path(default)
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
		response = create_global_path(input(f"{prompt}"))
		if not response and default:
			response = create_global_path(default)
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

def parse_config(config_file):
	with open(config_file, 'r') as data_file:
		return json.load(data_file)

def write_config(data, config_file):
	with open(config_file, "w") as data_file:
		json.dump(data, data_file)

def get_slide_paths(slides_dir):
	num_dir = len(slides_dir.split('/'))
	slide_list = [i for i in glob(join(slides_dir, '**/*.svs'))
					if i.split('/')[num_dir] != 'thumbs']

	slide_list.extend( [i for i in glob(join(slides_dir, '**/*.jpg'))
						if i.split('/')[num_dir] != 'thumbs'] )

	slide_list.extend(glob(join(slides_dir, '*.svs')))
	slide_list.extend(glob(join(slides_dir, '*.jpg')))
	return slide_list

def get_annotations_dict(annotations_file, key_name, value_name, use_encode=True):
	# TODO: save new annotations file if encoding needs to be done
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader, None)
		encode = False
		key_dict_int = {}
		key_dict_str = {}
		try:
			value_index = header.index(value_name)
			key_index = header.index(key_name)
		except:
			print(f" + [{fail('ERROR')}] Unable to find '{key_name}' and/or '{value_name}' headers in annotation file")
			sys.exit()
		for row in csv_reader:
			value = row[value_index]
			key = row[key_index]
			key_dict_str.update({key: value})
			if use_encode:
				try:
					int_value = int(value)
					key_dict_int.update({key: int_value})
				except:
					if use_encode and not encode:
						print(f" + [{info('INFO')}] Non-integer in '{value_name}' header, encoding with integer values")
						encode = True
		if use_encode and encode:
			values = set(key_dict_str.values())
			values_str_to_int = {}
			for i, c in enumerate(values):
				values_str_to_int.update({c: i})
				print(f"   - {value_name} '{info(c)}' assigned to value '{i}'")
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
		print(f" + [{fail('ERROR')}] Check annotations file for headers 'case' and 'category'.")
		sys.exit()
	try:
		slide_index = header.index(TCGAAnnotations.slide)
	except:
		print(f" + [{fail('ERROR')}] Header column 'slide' not found.")
		if slides_dir and yes_no_input('\nSearch slides directory and automatically associate cases with slides? [Y/n] ', default='yes'):
			# First, load all case names from the annotations file
			cases = []
			case_slide_dict = {}
			with open(annotations_file) as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=',')
				header = next(csv_reader, None)
				for row in csv_reader:
					cases.extend([row[case_index]])
			cases = list(set(cases)) # remove duplicates
			print(cases)
			# Next, search through the slides folder for all SVS/JPG files
			print(f" + Searching {slides_dir}...")
			
			for slide_filename in slide_list:
				slide_name = slide_filename.split('/')[-1][:-4]
				# First, make sure the shortname and long name aren't both in the annotation file
				if (slide_name in cases) and (_shortname(slide_name) in cases):
					print(f" + [{fail('ERROR')}] Both slide name {slide_name} and shorthand {_shortname(slide_name)} in annotation file; please remove one.")
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
				print(header)
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
		for row in csv_reader:
			if not row[slide_index] in [s.split('/')[-1][:-4] for s in slide_list]:
				if yes_no_input(f" + [{warn('WARN')}] Unable to locate slide {row[slide_index]}. Quit? [y/N] ", default='no'):
					sys.exit()

def verify_tiles(annotations, input_dir, tfrecord_files=[]):
	'''Iterate through folders if using raw images and verify all have an annotation;
	if using TFRecord, iterate through all records and verify all entries for valid annotation.'''
	print(f" + Verifying tiles and annotations...")
	success = True
	case_list = []
	if tfrecord_files:
		case_list_errors = []
		for tfrecord_file in tfrecord_files:
			tfrecord_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file)
			for string_record in tfrecord_iterator:
				example = tf.train.Example()
				example.ParseFromString(string_record)
				case = example.features.feature['case'].bytes_list.value[0].decode('utf-8')
				case_list.extend([case])
				case_list = list(set(case_list))
				if case not in annotations:
					case_list_errors.extend([case])
					case_list_errors = list(set(case_list_errors))
					success = False
		for case in case_list_errors:
			print(f" + [{fail('ERROR')}] Failed TFRecord integrity check: annotation not found for case {green(case)}")

	else:
		case_list = [i.split('/')[-1] for i in glob(os.path.join(input_dir, "train_data/*"))]
		case_list.extend([i.split('/')[-1] for i in glob(os.path.join(input_dir, "eval_data/*"))])
		case_list = list(set(case_list))
		for case in case_list:
			if case not in annotations:
				print(f" + [{fail('ERROR')}] Failed image tile integrity check: annotation not found for case {green(case)}")
				success = False
	# Now, check to see if all annotations have a corresponding set of tiles
	for annotation_case in annotations.keys():
		if annotation_case not in case_list:
			print(f"   - [{warn('WARN')}] Case {green(annotation_case)} in annotation file has no image tiles")
	if not success:
		sys.exit()
	else:
		print(f"   ...complete.")
