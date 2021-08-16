import sys
import json
import types
import csv
import time
import os
import io
import shutil
import datetime
import threading
import cv2

from glob import glob
from tensorflow.keras import backend as K
from os.path import join, isdir, exists
from PIL import Image
import multiprocessing as mp
import numpy as np

# TODO: re-enable logging with maximum log file size

# Enable color sequences on Windows
try:
	import ctypes
	kernel32 = ctypes.windll.kernel32
	kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except:
	pass
# ------

SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tiff', 'svslide', 'bif', 'jpg']
SLIDE_ANNOTATIONS_TO_IGNORE = ['', 'na', 'n/a', 'none', 'missing']

HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
PURPLE = '\033[38;5;5m'
UNDERLINE = '\033[4m'

FORMATTING_OPTIONS = [HEADER, BLUE, GREEN, WARNING, FAIL, ENDC, BOLD, UNDERLINE]
LOGGING_PREFIXES = ['', ' + ', '    - ']
LOGGING_PREFIXES_WARN = ['', ' ! ', '    ! ']
LOGGING_PREFIXES_EMPTY = ['', '   ', '     ']

# Old BatchNorm fix for bug in TF v1.14
#class UpdatedBatchNormalization(tf.keras.layers.BatchNormalization):
#	def call(self, inputs, training=None):
#		true_phase = int(K.get_session().run(K.learning_phase()))
#		trainable = int(self.trainable)
#		with K.learning_phase_scope(trainable * true_phase):
#			return super(tf.keras.layers.BatchNormalization, self).call(inputs, training)

class StainNormalizer:
	'''Object to supervise stain normalization for images and 
	efficiently convert between common image types.'''

	def __init__(self, method='macenko', source=None):
		'''Initializer. Establishes normalization method.

		Args:
			method:		Either 'macenko', 'reinhard', or 'vahadane'.
			source:		Path to source image for normalizer. 
							If not provided, defaults to an internal example image.
		'''
		from slideflow.slide import stainNorm_Macenko, stainNorm_Reinhard, stainNorm_Vahadane, stainNorm_Augment, stainNorm_Reinhard_Mask

		self.normalizers = {
			'macenko':  stainNorm_Macenko.Normalizer,
			'reinhard': stainNorm_Reinhard.Normalizer,
			'reinhard_mask': stainNorm_Reinhard_Mask.Normalizer,
			'vahadane': stainNorm_Vahadane.Normalizer,
			'augment': stainNorm_Augment.Normalizer
		}

		if not source:
			package_directory = os.path.dirname(os.path.abspath(__file__))
			source = join(package_directory, 'norm_tile.jpg')
		self.n = self.normalizers[method]()
		self.n.fit(cv2.imread(source))

	def pil_to_pil(self, image):
		'''Non-normalized PIL.Image -> normalized PIL.Image'''
		cv_image = np.array(image.convert('RGB'))
		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
		cv_image = self.n.transform(cv_image)
		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
		return Image.fromarray(cv_image)

	def tf_to_rgb(self, image):
		'''Non-normalized tensorflow image array -> normalized RGB numpy array'''
		return self.rgb_to_rgb(np.array(image))

	def rgb_to_rgb(self, image):
		'''Non-normalized RGB numpy array -> normalized RGB numpy array'''
		cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv_image = self.n.transform(cv_image)
		return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

	def jpeg_to_rgb(self, jpeg_string):
		'''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
		cv_image = cv2.imdecode(np.fromstring(jpeg_string, dtype=np.uint8), cv2.IMREAD_COLOR)
		cv_image = self.n.transform(cv_image)
		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
		return cv_image

	def jpeg_to_jpeg(self, jpeg_string):
		'''Non-normalized compressed JPG string data -> normalized compressed JPG string data'''
		cv_image = self.jpeg_to_rgb(jpeg_string)
		with io.BytesIO() as output:
			Image.fromarray(cv_image).save(output, format="JPEG", quality=75)
			return output.getvalue()

class DummyLock:
	def __init__(self, *args): pass
	def __enter__(self, *args): pass
	def __exit__(self, *args): pass

class Bar:
	def __init__(self, ending_value, starting_value=0, bar_length=20, label='',
					show_eta=False, show_counter=False, counter_text='', update_interval=1,
					mp_counter=None, mp_lock=None):

		if mp_counter is not None:
			self.counter = mp_counter
			self.mp_lock = mp_lock
		else:
			manager = mp.Manager()
			self.counter = manager.Value('i', 0)
			self.mp_lock = manager.Lock()

		# Setup timing
		self.starttime = None
		self.lastupdated = None
		self.checkpoint_time = None
		self.checkpoint_val = starting_value

		# Other initializing variables
		self.counter.value = starting_value
		self.end_value = ending_value
		self.bar_length = bar_length
		self.label = label
		self.show_counter = show_counter
		self.counter_text = '' if not counter_text else " " + counter_text
		self.show_eta = show_eta
		self.text = ''
		self.num_per_sec = 0
		self.update_interval = update_interval

	def get_text(self):
		current_time = int(time.time())
		if not self.starttime:
			self.starttime = current_time
			self.checkpoint_time = current_time
			self.checkpoint_val = self.counter.value
			self.lastupdated = self.starttime
		elif current_time == self.lastupdated:
			return self.text
		else:
			current_time

		timediff = int(time.time())-self.starttime

		# Checkpoint every 5 seconds
		if (current_time - self.checkpoint_time) > self.update_interval:
			self.num_per_sec = (self.counter.value - self.checkpoint_val) / (current_time - self.checkpoint_time)
			# Reset checkpoint
			self.checkpoint_val = self.counter.value
			self.checkpoint_time = current_time

		percent = float(self.counter.value) / self.end_value
		arrow = chr(0x2588) * int(round(percent * self.bar_length))
		spaces = u'-' * (self.bar_length - len(arrow))

		self.text = u"\u007c{0}\u007c {1:.1f}%{2}".format(arrow + spaces, 
													 (float(self.counter.value) / self.end_value)*100, 
													 f' ({self.label})' if self.label else '')
		if self.show_counter and self.num_per_sec:
			num_per_sec_str = "?" if timediff == 0 else f'{self.num_per_sec:.1f}'
			self.text += f" {num_per_sec_str}{self.counter_text}/sec"
		if self.show_eta and timediff and self.num_per_sec:
			eta_sec = (self.end_value - self.counter.value) / self.num_per_sec
			self.text += f" (ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))})"
		elif self.show_eta:
			self.text += f" (ETA: ?)"

		return self.text

class ProgressBar:
	'''Flexible progress bar with dynamic ETA monitoring.'''
	tail = ''
	text = ''

	def __init__(self, ending_val, starting_val=0, bar_length=20, endtext='', show_eta=False, 
					show_counter=False, counter_text='', leadtext='', mp_counter=None, mp_lock=None):
		
		self.leadtext = leadtext
		self.refresh_thread = None
		self.live = True
		self.BARS = [Bar(ending_val, starting_val, bar_length, endtext, show_eta, show_counter, counter_text, mp_counter=mp_counter, mp_lock=mp_lock)]
		self.refresh()

	def add_bar(self, val, endval, bar_length=20, endtext='', show_eta=False,
					show_counter=False, counter_text=''):

		self.BARS += [Bar(val, endval, bar_length, endtext, show_eta, show_counter, counter_text)]
		self.refresh()
		return len(self.BARS)-1

	def increase_bar_value(self, amount=1, id=0):
		with self.BARS[id].mp_lock:
			self.BARS[id].counter.value = min(self.BARS[id].counter.value + amount, self.BARS[id].end_value)
		self.refresh()

	def get_counter(self, id=0):
		return self.BARS[id].counter

	def get_lock(self, id=0):
		return self.BARS[id].mp_lock

	def set_bar_value(self, value, id=0):
		with self.BARS[id].mp_lock:
			self.BARS[id].counter.value = min(value, self.BARS[id].end_value)
		self.refresh()

	def set_bar_text(self, text, id=0):
		self.BARS[id].text = text
		self.refresh()

	def auto_refresh(self, freq=0.2):
		def auto_refresh_worker():
			while self.live:
				self.refresh()
				time.sleep(freq)
			
		self.refresh_thread = threading.Thread(target=auto_refresh_worker, daemon=True)
		self.refresh_thread.start()

	def refresh(self):
		if len(self.BARS) == 0:
			sys.stdout.write("\r\033[K")
			sys.stdout.flush()
			return
		new_text = f"\r\033[K{self.leadtext}"
		for bar in self.BARS:
			new_text += bar.get_text()
			if len(self.BARS) > 1:
				new_text += "  "
		new_text += self.tail
		if new_text != self.text:
			sys.stdout.write(new_text)
			sys.stdout.flush()
			self.text = new_text

	def end(self, id=-1):
		if id == -1:
			self.BARS = []
			print(f"\r\033[K", end="")
		else:
			del(self.BARS[id])
			print(f"\r\033[K{self.text}", end="")
		self.live = False

	def print(self, string):
		sys.stdout.write(f"\r\033[K{string}\n")
		sys.stdout.flush()
		sys.stdout.write(self.text)
		sys.stdout.flush()

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

def purple(text):
	return PURPLE + str(text) + ENDC

class Logger:
	'''Logging class to handle console and file logging output.'''

	def __init__(self):
		self.logfile = None
		self.INFO_LEVEL = 3
		self.WARN_LEVEL = 3
		self.ERROR_LEVEL = 3
		self.COMPLETE_LEVEL = 3
		self.SILENT = False

	def configure(self, **kwargs):
		'''Configures logger to record the designated logging levels, overriding defaults.'''

		for arg in kwargs:
			if arg not in ('filename', 'levels'):
				raise TypeError(f"Unknown argument '{arg}'")
		
		if 'filename' in kwargs: self.logfile = kwargs['filename']

		if 'levels' in kwargs:
			levels = kwargs['levels']
			if 'info' in levels: self.INFO_LEVEL = levels['info']
			if 'warn' in levels: self.WARN_LEVEL = levels['warn']
			if 'error' in levels: self.ERROR_LEVEL = levels['error']
			if 'complete' in levels: self.COMPLETE_LEVEL = levels['complete']
			if 'silent' in levels: self.SILENT = levels['silent']

	def info(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{info('INFO')}] {text}"
		if print_func and l <= self.INFO_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message

	def warn(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{warn('WARN')}] {text}"
		if print_func and l <= self.WARN_LEVEL:
			print_func(message)
		self.log(message)
		return message

	def error(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{fail('ERROR')}] {text}"
		if print_func and l <= self.ERROR_LEVEL:
			print_func(message)
		self.log(message)
		return message

	def complete(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{header('Complete')}] {text}"
		if print_func and l <= self.COMPLETE_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message

	def label(self, label, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{green(label)}] {text}"
		if print_func and l <= self.INFO_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message

	def empty(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}{text}"
		if print_func and l <= self.INFO_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message

	def header(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"\n{LOGGING_PREFIXES_EMPTY[l]}{bold(text)}"
		if print_func and not self.SILENT:
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

class ThreadSafeList:
	def __init__(self):
		self.lock = threading.Lock()
		self.items = []
	def add(self, item):
		with self.lock:
			self.items.append(item)
	def getAll(self):
		with self.lock:
			items, self.items = self.items, []
		return items

def make_dir(_dir):
	'''Makes a directory if one does not already exist, in a manner compatible with multithreading. '''
	if not exists(_dir):
		try:
			os.makedirs(_dir, exist_ok=True)
		except FileExistsError:
			pass

def global_path(root, path_string):
	'''Returns global path from a local path.'''
	if not root: root = ""
	if path_string and (len(path_string) > 2) and path_string[:2] == "./":
		return os.path.join(root, path_string[2:])
	elif path_string and (path_string[0] != "/"):
		return os.path.join(root, path_string)
	else:
		return path_string		

def _shortname(string):
	if len(string) == 60:
		# May be TCGA slide with long name; convert to patient name by returning first 12 characters
		return string[:12]
	else:
		return string

def yes_no_input(prompt, default='no'):
	'''Prompts user for yes/no input.'''
	yes = ['yes','y']
	no = ['no', 'n']
	while True:
		response = input(prompt)
		if not response and default:
			return True if default in yes else False
		if response.lower() in yes:
			return True
		if response.lower() in no:
			return False
		print(f"Invalid response.")

def dir_input(prompt, root, default=None, create_on_invalid=False, absolute=False):
	'''Prompts user for directory input.'''
	while True:
		if not absolute:
			response = global_path(root, input(f"{prompt}"))
		else:
			response = input(f"{prompt}")
		if not response and default:
			response = global_path(root, default)
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

def file_input(prompt, root, default=None, filetype=None, verify=True):
	'''Prompts user for file input.'''
	while True:
		response = global_path(root, input(f"{prompt}"))
		if not response and default:
			response = global_path(root, default)
		if verify and not os.path.exists(response):
			print(f'Unable to locate file "{response}"')
			continue
		extension = path_to_ext(response)
		if filetype and (extension != filetype):
			print(f'Incorrect filetype; provided file of type "{extension}", need type "{filetype}"')
			continue
		return response

def int_input(prompt, default=None):
	'''Prompts user for int input.'''
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
	'''Prompts user for float input.'''
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

def choice_input(prompt, valid_choices, default=None, multi_choice=False, input_type=str):
	'''Prompts user for multi-choice input.'''
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		if not multi_choice and response not in valid_choices:
			print("Invalid option.")
			continue
		elif multi_choice:
			try:
				response = [input_type(r) for r in response.replace(" ", "").split(',')]
			except:
				print(f"Invalid selection (response: {response})")
				continue
			invalid = [r not in valid_choices for r in response]
			if any(invalid):
				print(f'Invalid selection (response: {response})')
				continue
		return response

def load_json(filename):
	'''Reads JSON data from file.'''
	with open(filename, 'r') as data_file:
		return json.load(data_file)

def write_json(data, filename):
	'''Writes data to JSON file.'''
	with open(filename, "w") as data_file:
		json.dump(data, data_file, indent=1)

def get_slide_paths(slides_dir):
	'''Get all slide paths from a given directory containing slides.'''
	slide_list = [i for i in glob(join(slides_dir, '**/*.*')) if path_to_ext(i).lower() in SUPPORTED_FORMATS]
	slide_list.extend([i for i in glob(join(slides_dir, '*.*')) if path_to_ext(i).lower() in SUPPORTED_FORMATS])
	return slide_list

def read_annotations(annotations_file):
	'''Read an annotations file.'''
	results = []
	# Open annotations file and read header
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		# First, try to open file
		try:
			header = next(csv_reader, None)
		except OSError:
			err_msg = f"Unable to open annotations file {green(annotations_file)}, is it open in another program?"
			log.error(err_msg)
			raise OSError(err_msg)

		for row in csv_reader:
			row_dict = {}
			for i, key in enumerate(header):
				row_dict[key] = row[i]
			results += [row_dict]
	return header, results
			
def get_relative_tfrecord_paths(root, directory=""):
	tfrecords = [join(directory, f) for f in os.listdir(join(root, directory)) if (not isdir(join(root, directory, f)) and len(f) > 10 and f[-10:] == ".tfrecords")]
	subdirs = [f for f in os.listdir(join(root, directory)) if isdir(join(root, directory, f))]
	for sub in subdirs:
		tfrecords += get_relative_tfrecord_paths(root, join(directory, sub))
	return tfrecords

def contains_nested_subdirs(directory):
	subdirs = [_dir for _dir in os.listdir(directory) if isdir(join(directory, _dir))]
	for subdir in subdirs:
		contents = os.listdir(join(directory, subdir))
		for c in contents:
			if isdir(join(directory, subdir, c)):
				return True
	return False

def path_to_name(path):
	'''Returns name of a file, without extension, from a given full path string.'''
	_file = path.split('/')[-1]
	if len(_file.split('.')) == 1:
		return _file
	else:
		return '.'.join(_file.split('.')[:-1])

def path_to_ext(path):
	'''Returns extension of a file path string.'''
	_file = path.split('/')[-1]
	if len(_file.split('.')) == 1:
		return ''
	else:
		return _file.split('.')[-1]

def update_results_log(results_log_path, model_name, results_dict):
	'''Dynamically update results_log when recording training metrics.'''
	# First, read current results log into a dictionary
	results_log = {}
	if exists(results_log_path):
		with open(results_log_path, "r") as results_file:
			reader = csv.reader(results_file)
			headers = next(reader)
			try:
				model_name_i = headers.index('model_name')
				result_keys = [k for k in headers if k != 'model_name']
			except ValueError:
				model_name_i = headers.index('epoch')
				result_keys = [k for k in headers if k != 'epoch']
			for row in reader:
				name = row[model_name_i]
				results_log[name] = {}
				for result_key in result_keys:
					result = row[headers.index(result_key)]
					results_log[name][result_key] = result
		# Move the current log file into a temporary file
		shutil.move(results_log_path, f"{results_log_path}.temp")

	# Next, update the results log with the new results data
	for epoch in results_dict:
		results_log.update({f'{model_name}-{epoch}': results_dict[epoch]})

	# Finally, create a new log file incorporating the new data
	with open(results_log_path, "w") as results_file:
		writer = csv.writer(results_file)
		result_keys = []
		# Search through results to find all results keys
		for model in results_log:
			result_keys += list(results_log[model].keys())
		# Remove duplicate result keys
		result_keys = list(set(result_keys))
		result_keys.sort()
		# Write header labels
		writer.writerow(['model_name'] + result_keys)
		# Iterate through model results and record
		for model in results_log:
			row = [model]
			# Include all saved metrics
			for result_key in result_keys:
				if result_key in results_log[model]:
					row += [results_log[model][result_key]]
				else:
					row += [""]
			writer.writerow(row)

	# Delete the old results log file
	if exists(f"{results_log_path}.temp"):
		os.remove(f"{results_log_path}.temp")

def read_predictions_from_csv(path, outcome_labels=None, restrict_outcomes=None, outcome_type='categorical'):
	'''Function to assist with loading predictions from files.

	Returns:
		Dictionary mapping slides to nested dictionary, which maps outcome labels to predictions.'''

	predictor_label = 'percent_tiles_positive' if outcome_type == 'categorical' else 'average'

	predictions = {}

	with open(path, 'r') as csv_file:
		reader = csv.reader(csv_file)
		header = next(reader)
		slide_i = header.index('slide')

		outcomes = restrict_outcomes if restrict_outcomes else [int(h.split('y_true')[-1]) for h in header if 'y_true' in h]

		for row in reader:
			slide = row[slide_i]
			predictions_dict = {}
			for outcome in outcomes:
				prediction_index = header.index(f'{predictor_label}{outcome}')
				outcome_label = outcome if not outcome_labels else outcome_labels[outcome]
				predictions_dict.update({outcome_label: row[prediction_index]})
			predictions.update({slide: predictions_dict})

	return predictions
