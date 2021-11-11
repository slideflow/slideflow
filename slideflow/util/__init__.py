import sys
import json
import csv
import time
import os
import io
import shutil
import threading
import logging
import cv2
#import multiprocessing_logging
import importlib

import multiprocessing as mp
import numpy as np
import slideflow as sf

from glob import glob
from os.path import join, isdir, exists, dirname
from PIL import Image
from tqdm import tqdm

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
SLIDE_ANNOTATIONS_TO_IGNORE = ['', ' ']
LOGGING_PREFIXES = ['', ' + ', '    - ']
LOGGING_PREFIXES_WARN = ['', ' ! ', '    ! ']
LOGGING_PREFIXES_EMPTY = ['', '   ', '     ']
CPLEX_AVAILABLE = (importlib.util.find_spec('cplex') is not None)

def dim(text):        return '\033[2m' + str(text) + '\033[0m'
def yellow(text):     return '\033[93m' + str(text) + '\033[0m'
def cyan(text):       return '\033[96m' + str(text) + '\033[0m'
def blue(text):       return '\033[94m' + str(text) + '\033[0m'
def green(text):      return '\033[92m' + str(text) + '\033[0m'
def red(text):        return '\033[91m' + str(text) + '\033[0m'
def bold(text):       return '\033[1m' + str(text) + '\033[0m'
def underline(text):  return '\033[4m' + str(text) + '\033[0m'
def purple(text):     return '\033[38;5;5m' + str(text) + '\033[0m'

log = logging.getLogger('slideflow')
if 'SF_LOGGING_LEVEL' in os.environ:
    try:
        intLevel = int(os.environ['SF_LOGGING_LEVEL'])
        log.setLevel(intLevel)
    except:
        pass
else:
    log.setLevel(logging.INFO)

class UserError(Exception):
    pass

class CPLEXError(Exception):
    pass

class LogFormatter(logging.Formatter):
    MSG_FORMAT = "%(asctime)s [%(levelname)s] - %(message)s"
    LEVEL_FORMATS = {
        logging.DEBUG: dim(MSG_FORMAT),
        logging.INFO: MSG_FORMAT,
        logging.WARNING: yellow(MSG_FORMAT),
        logging.ERROR: red(MSG_FORMAT),
        logging.CRITICAL: bold(red(MSG_FORMAT))
    }

    def format(self, record):
        log_fmt = self.LEVEL_FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt, '%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class FileFormatter(logging.Formatter):
    MSG_FORMAT = "%(asctime)s [%(levelname)s] - %(message)s"
    FORMAT_CHARS = ['\033[1m', '\033[2m', '\033[4m', '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[38;5;5m', '\033[0m']
    def format(self, record):
        formatter = logging.Formatter(fmt=self.MSG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
        formatted = formatter.format(record)
        for char in self.FORMAT_CHARS:
            formatted = formatted.replace(char, '')
        return formatted

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flush_line = False

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.flush_line:
                msg = '\r\033[K' + msg
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except:
            print(f"problems with msg {record}")
            self.handleError(record)

#multiprocessing_logging.install_mp_handler(log)
ch = TqdmLoggingHandler()
ch.setFormatter(LogFormatter())
ch.setLevel(log.level)
log.addHandler(ch)
# ------------------------------------------------------------

class DummyLock:
    def __init__(self, *args): pass
    def __enter__(self, *args): pass
    def __exit__(self, *args): pass

class DummyCounter:
    def __init__(self, value):
        self.value = value

class Bar:
    def __init__(self, ending_value, starting_value=0, bar_length=20, label='',
                    show_eta=False, show_counter=False, counter_text='', update_interval=1,
                    mp_counter=None, mp_lock=None):

        if mp_counter is not None:
            self.counter = mp_counter
            self.mp_lock = mp_lock
        else:
            try:
                manager = mp.Manager()
                self.counter = manager.Value('i', 0)
                self.mp_lock = manager.Lock()
            except AssertionError:
                self.counter = DummyCounter(0)
                self.mp_lock = DummyLock()

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
    '''Flexible progress bar with dynamic ETA monitoring and multiprocessing support.'''
    tail = ''
    text = ''

    def __init__(self, ending_val, starting_val=0, bar_length=20, endtext='', show_eta=False,
                    show_counter=False, counter_text='', leadtext='', mp_counter=None, mp_lock=None):

        self.leadtext = leadtext
        self.refresh_thread = None
        self.live = True
        self.BARS = [Bar(ending_val,
                         starting_val,
                         bar_length,
                         endtext,
                         show_eta,
                         show_counter,
                         counter_text,
                         mp_counter=mp_counter,
                         mp_lock=mp_lock)]
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

    def auto_refresh(self, freq=0.1):
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

class TCGA:
    patient = 'patient'
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

def to_onehot(val, max):
    """Converts value to one-hot encoding

    Args:
        val (int): Value to encode
        max (int): Maximum value (length of onehot encoding)
    """

    onehot = np.zeros(max, dtype=np.int64)
    onehot[val] = 1
    return onehot

def clear_console():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

def make_dir(_dir):
    '''Makes a directory if one does not already exist, in a manner compatible with multithreading. '''
    if not exists(_dir):
        try:
            os.makedirs(_dir, exist_ok=True)
        except FileExistsError:
            pass

def relative_path(path, root):
    if path[0] == '.':
        return join(root, path[2:])
    elif path[:5] == '$ROOT':
        log.warn('Deprecation warning: invalid path prefix $ROOT, please update project settings ' + \
                    '(use "." for relative paths)')
        return join(root, path[6:])
    else:
        return path

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

def path_input(prompt, root, default=None, create_on_invalid=False, filetype=None, verify=True):
    '''Prompts user for directory input.'''
    while True:
        relative_response = input(f"{prompt}")
        global_response = global_path(root, relative_response)
        if not relative_response and default:
            relative_response = default
            global_response = global_path(root, relative_response)
        if verify and not os.path.exists(global_response):
            if not filetype and create_on_invalid:
                if yes_no_input(f'Directory "{global_response}" does not exist. Create directory? [Y/n] ', default='yes'):
                    os.makedirs(global_response)
                    return relative_response
                else:
                    continue
            elif filetype:
                print(f'Unable to locate file "{global_response}"')
                continue
        elif not filetype and not os.path.exists(global_response):
            print(f'Unable to locate directory "{global_response}"')
            continue
        if filetype and (path_to_ext(global_response) != filetype):
            print(f'Incorrect filetype; provided file of type "{path_to_ext(global_response)}", need type "{filetype}"')
            continue
        return relative_response

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

def get_slides_from_model_manifest(model_path, dataset=None):
    """Get list of slides from a model manifest.

    Args:
        model_path (str): Path to model from which to load the model manifest.
        dataset (str):  'training' or 'validation'. Will return only slides from this dataset. Defaults to None (all).

    Returns:
        list(str): List of slide names.
    """

    slides = []
    if exists(join(model_path, 'slide_manifest.csv')):
        manifest = join(model_path, 'slide_manifest.csv')
    elif exists(join(dirname(model_path), 'slide_manifest.csv')):
        log.warning("Slide manifest file not found in model directory; loading from parent directory. " + \
                    "Please move slide_manifest.csv into model folder.")
        manifest = join(dirname(model_path), 'slide_manifest.csv')
    else:
        log.error('Slide manifest file not found (could not find "slide_manifest.csv" in model folder)')
        return None
    with open(manifest, 'r') as manifest_file:
        reader = csv.reader(manifest_file)
        header = next(reader)
        dataset_index = header.index('dataset')
        slide_index = header.index('slide')
        for row in reader:
            dataset_name = row[dataset_index]
            slide_name = row[slide_index]
            if dataset_name == dataset or not dataset:
                slides += [slide_name]
    return slides

def get_model_config(model_path):
    """Loads model configuration JSON file."""

    if exists(join(model_path, 'params.json')):
        return load_json(join(model_path, 'params.json'))
    elif exists(join(dirname(model_path), 'params.json')):
        if sf.backend() == 'tensorflow':
            log.warning("Hyperparameters file not found in model directory; loading from parent directory. " + \
                        "Please move params.json into model folder.")
        return load_json(join(dirname(model_path), 'params.json'))
    else:
        log.warning("Hyperparameters file not found.")
        return None

def get_slide_paths(slides_dir):
    '''Get all slide paths from a given directory containing slides.'''
    slide_list = [i for i in glob(join(slides_dir, '**/*.*')) if path_to_ext(i).lower() in SUPPORTED_FORMATS]
    slide_list.extend([i for i in glob(join(slides_dir, '*.*')) if path_to_ext(i).lower() in SUPPORTED_FORMATS])
    return slide_list

def read_annotations(annotations_file):
    '''Read an annotations file.'''
    results = []
    # Open annotations file and read header
    with open(annotations_file, 'r') as csv_file:
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
    '''Returns relative tfrecord paths with respect to the given directory.'''

    tfrecords = [join(directory, f) for f in os.listdir(join(root, directory))
                                    if (not isdir(join(root, directory, f))
                                        and len(f) > 10 and f[-10:] == ".tfrecords")]
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