import atexit
import csv
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import requests
import tarfile
import hashlib
import pandas as pd
import tempfile
import threading
import multiprocessing as mp
import time
from rich import progress
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn
from contextlib import contextmanager
from functools import partial
from glob import glob
from os.path import dirname, exists, isdir, join
from packaging import version
from tqdm import tqdm
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Iterator
)

import numpy as np
import slideflow as sf
from slideflow import errors
from . import example_pb2, log_utils
from .colors import *  # noqa F403,F401 - Here for compatibility
from .smac_utils import (broad_search_space, shallow_search_space,
                         create_search_space)

tf_available = importlib.util.find_spec('tensorflow')
torch_available = importlib.util.find_spec('torch')

# Enable color sequences on Windows
try:
    import ctypes.windll
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except Exception:
    pass


# --- Global vars -------------------------------------------------------------

SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs',
                     'tiff', 'svslide', 'bif', 'jpg', 'jpeg', 'png',
                     'ome.tif', 'ome.tiff']
EMPTY = ['', ' ', None, np.nan]
CPLEX_AVAILABLE = (importlib.util.find_spec('cplex') is not None)
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    opt = SolverFactory('bonmin', validate=False)
    if not opt.available():
        raise errors.SolverNotFoundError
except Exception:
    BONMIN_AVAILABLE = False
else:
    BONMIN_AVAILABLE = True


# --- Commonly used types -----------------------------------------------------

# Outcome labels
Labels = Union[Dict[str, str], Dict[str, int], Dict[str, List[float]]]

# Normalizer fit keyword arguments
NormFit = Union[Dict[str, np.ndarray], Dict[str, List]]

# --- Detect CPU cores --------------------------------------------------------

def num_cpu(default: Optional[int] = None) -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))
    except Exception as e:
        count = os.cpu_count()
        if count is None and default is not None:
            return default
        else:
            return count

# --- Configure logging--------------------------------------------------------

log = logging.getLogger('slideflow')
log.setLevel(logging.DEBUG)


def setLoggingLevel(level):
    """Set the logging level.

    Uses standard python logging levels:

    - 50: CRITICAL
    - 40: ERROR
    - 30: WARNING
    - 20: INFO
    - 10: DEBUG
    - 0:  NOTSET

    Args:
        level (int): Logging level numeric value.

    """
    log.handlers[0].setLevel(level)


def getLoggingLevel():
    """Return the current logging level."""
    return log.handlers[0].level


@contextmanager
def logging_level(level: int):
    _initial = getLoggingLevel()
    setLoggingLevel(level)
    try:
        yield
    finally:
        setLoggingLevel(_initial)


def addLoggingFileHandler(path):
    fh = logging.FileHandler(path)
    fh.setFormatter(log_utils.FileFormatter())
    handler = log_utils.MultiProcessingHandler(
        "mp-file-handler-{0}".format(len(log.handlers)),
        sub_handler=fh
    )
    log.addHandler(handler)
    atexit.register(handler.close)


# Add tqdm-friendly stream handler
#ch = log_utils.TqdmLoggingHandler()
ch = RichHandler(
    markup=True,
    log_time_format="[%X]",
    show_path=False,
    highlighter=NullHighlighter(),
    rich_tracebacks=True
)
ch.setFormatter(log_utils.LogFormatter())
if 'SF_LOGGING_LEVEL' in os.environ:
    try:
        intLevel = int(os.environ['SF_LOGGING_LEVEL'])
        ch.setLevel(intLevel)
    except ValueError:
        pass
else:
    ch.setLevel(logging.INFO)
log.addHandler(ch)

# Add multiprocessing-friendly file handler
addLoggingFileHandler("slideflow.log")

# Workaround for duplicate logging with TF 2.9
log.propagate = False


class TileExtractionSpeedColumn(progress.ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task: "progress.Task") -> progress.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return progress.Text("?", style="progress.data.speed")
        data_speed = f'{int(speed)} img'
        return progress.Text(f"{data_speed}/s", style="progress.data.speed")

class ImgBatchSpeedColumn(progress.ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def render(self, task: "progress.Task") -> progress.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return progress.Text("?", style="progress.data.speed")
        data_speed = f'{int(speed * self.batch_size)} img'
        return progress.Text(f"{data_speed}/s", style="progress.data.speed")


class TileExtractionProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == 'speed':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TileExtractionSpeedColumn(),)
            if task.fields.get("progress_type") == 'slide_progress':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    progress.TaskProgressColumn(),
                    progress.MofNCompleteColumn(),
                    "●",
                    progress.TimeRemainingColumn(),
                )
            yield self.make_tasks_table([task])


def set_ignore_sigint():
    """Ignore keyboard interrupts."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class MultiprocessProgressTracker:
    """Wrapper for a rich.progress tracker that can be shared across processes."""

    def __init__(self, tasks):
        ctx = mp.get_context('spawn')
        self.mp_values = {
            task.id: ctx.Value('i', task.completed)
            for task in tasks
        }

    def advance(self, id, amount):
        with self.mp_values[id].get_lock():
            self.mp_values[id].value += amount

    def __getitem__(self, id):
        return self.mp_values[id].value

class MultiprocessProgress:
    """Wrapper for a rich.progress bar that can be shared across processes."""

    def __init__(self, pb):
        self.pb = pb
        self.tracker = MultiprocessProgressTracker(self.pb.tasks)
        self.should_stop = False

    def _update_progress(self):
        while not self.should_stop:
            for task in self.pb.tasks:
                self.pb.update(task.id, completed=self.tracker[task.id])
            time.sleep(0.1)

    def __enter__(self):
        self._thread = threading.Thread(target=self._update_progress)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.should_stop = True
        self._thread.join()


# --- Slideflow header --------------------------------------------------------

def about(console=None) -> None:
    """Print a summary of the slideflow version and active backends.

    Example
        >>> sf.about()
        ╭=======================╮
        │       Slideflow       │
        │    Version: 2.1.0     │
        │  Backend: tensorflow  │
        │ Slide Backend: cucim  │
        │ https://slideflow.dev │
        ╰=======================╯

    Args:
        console (rich.console.Console, optional): Active console, if one exists.
            Defaults to None.
    """
    if console is None:
        console = Console()
    col1 = 'yellow' if sf.backend() == 'tensorflow' else 'purple'
    if sf.slide_backend() == 'libvips':
        try:
            import pyvips
            _version = '{}.{}.{}'.format(
                pyvips.major, pyvips.minor, pyvips.micro
            )
        except Exception:
            _version = 'unknown'
        col2 = 'cyan'
        slide_backend = 'libvips ({})'.format(_version)
    else:
        slide_backend = sf.slide_backend()
        col2 = 'green'
    console.print(
        Panel(f"[white bold]Slideflow[/]"
              f"\nVersion: {sf.__version__}"
              f"\nBackend: [{col1}]{sf.backend()}[/]"
              f"\nSlide Backend: [{col2}]{slide_backend}[/]"
              "\n[blue]https://slideflow.dev[/]",
              border_style='purple'),
        justify='left')


# --- Data download functions -------------------------------------------------

def download_from_tcga(
    uuid: str,
    dest: str,
    message: str = 'Downloading...'
) -> None:
    """Download a file from TCGA (GDC) by UUID."""
    data_endpt = f"https://api.gdc.cancer.gov/data/"
    response = requests.post(
        data_endpt,
        data=json.dumps({'ids': [uuid]}),
        headers={"Content-Type": "application/json"},
        stream=True
    )
    response_head_cd = response.headers["Content-Disposition"]
    block_size = 4096
    block_per_mb = block_size / 1000000
    file_size = int(response.headers.get('Content-Length', ''))
    file_size_mb = file_size / 1000000
    running_total_mb = 0
    file_name = join(dest, re.findall("filename=(.+)", response_head_cd)[0])
    pbar = tqdm(desc=message,
                total=file_size_mb, unit='MB',
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "{n:.2f}/{total:.2f} [{elapsed}<{remaining}] "
                           "{rate_fmt}{postfix}")

    with open(file_name, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=block_size):
            output_file.write(chunk)
            if block_per_mb + running_total_mb < file_size_mb:
                running_total_mb += block_per_mb  # type: ignore
                pbar.update(block_per_mb)
            else:
                running_total_mb += file_size_mb - running_total_mb  # type: ignore
                pbar.update(file_size_mb - running_total_mb)


def make_cache_dir_path(path: str) -> str:
    if 'HOME' in os.environ:
        dest = os.path.join(os.environ['HOME'], '.cache', 'slideflow', path)
    elif 'USERPROFILE' in os.environ:
        dest = os.path.join(os.environ['USERPROFILE'], '.cache', 'slideflow', path)
    else:
        dest = os.path.join(tempfile.gettempdir(), '.cache', 'slideflow', path)
    os.makedirs(dest, exist_ok=True)
    return dest


def get_gdc_manifest() -> pd.DataFrame:
    sf_cache = make_cache_dir_path('gdc')
    manifest = join(sf_cache, 'gdc_manifest.tsv')
    if not exists(manifest):
        tar = 'gdc_manifest.tar.xz'
        r = requests.get(f'https://raw.githubusercontent.com/jamesdolezal/slideflow/1.4.0/datasets/{tar}')
        open(join(sf_cache, tar), 'wb').write(r.content)
        tarfile.open(join(sf_cache, tar)).extractall(sf_cache)
        os.remove(join(sf_cache, tar))
        if not exists(manifest):
            log.error("Failed to download GDC manifest.")
    return pd.read_csv(manifest, delimiter='\t')


# --- Utility functions and classes -------------------------------------------

class no_scope():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access
    with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def zip_allowed() -> bool:
    return not ('SF_ALLOW_ZIP' in os.environ and os.environ['SF_ALLOW_ZIP'] == '0')

@contextmanager
def enable_zip(enable: bool) -> Iterator[None]:
    _zip_allowed = zip_allowed()
    os.environ['SF_ALLOW_ZIP'] = '1' if enable else '0'
    yield
    os.environ['SF_ALLOW_ZIP'] = '0' if not _zip_allowed else '1'

def md5(path: str) -> str:
    """Calculate and return MD5 checksum for a file."""
    m = hashlib.md5()
    with open(path, 'rb') as f:
        chunk = f.read(4096)
        # No walrus for Python 3.7 :(
        while chunk:
            m.update(chunk)
            chunk = f.read(4096)
    return m.hexdigest()

def allow_gpu_memory_growth() -> None:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

def model_backend(model):
    if sf.util.torch_available and 'torch' in sys.modules:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'torch'
    if sf.util.tf_available and 'tensorflow' in sys.modules:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return 'tensorflow'
        from tensorflow.lite.python.interpreter import SignatureRunner
        if isinstance(model, SignatureRunner):
            return 'tflite'
    raise ValueError(f"Unable to interpret model {model}")


def detuple(arg1: Any, args: tuple) -> Any:
    if len(args):
        return tuple([arg1] + list(args))
    else:
        return arg1

def _as_list(arg1: Any) -> List[Any]:
    if isinstance(arg1, np.ndarray):
        return arg1.tolist()
    else:
        return arg1

def batch(iterable: List, n: int = 1) -> Iterable:
    """Separates an interable into batches of maximum size `n`."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def batch_generator(iterable: Iterable, n: int = 1) -> Iterable:
    """Separates an interable into batches of maximum size `n`."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if len(batch):
        yield batch
    return


def as_list(arg1: Any) -> List[Any]:
    if not isinstance(arg1, list):
        return [arg1]
    else:
        return arg1


def isnumeric(val: Any) -> bool:
    """Check if the given value is numeric (numpy or python).

    Tensors will return False.

    Specifically checks if the value is a python int or float,
    or if the value is a numpy array with a numeric dtype (int or float).

    """
    float_np_types = (np.int32, np.int64, np.uint8, np.float16,
                      np.float32, np.float64)
    if isinstance(val, (int, float)):
            return True
    if isinstance(val, np.ndarray):
            return val.dtype in float_np_types
    return type(val) in float_np_types


def is_mag(arg1: str) -> bool:
    arg1_split = arg1.lower().split('x')
    if (len(arg1_split) != 2) or (arg1_split[1] != ''):
        return False
    try:
        mag = float(arg1_split[0])
    except ValueError:
        return False
    return True


def is_model(path: str) -> bool:
    """Checks if the given path is a valid Slideflow model."""
    return is_tensorflow_model_path(path) or is_torch_model_path(path)


def is_project(path: str) -> bool:
    """Checks if the given path is a valid Slideflow project."""
    return isdir(path) and exists(join(path, 'settings.json'))


def is_slide(path: str) -> bool:
    """Checks if the given path is a supported slide."""
    return (os.path.isfile(path)
            and sf.util.path_to_ext(path).lower() in SUPPORTED_FORMATS)


def is_tensorflow_model_path(path: str) -> bool:
    """Checks if the given path is a valid Slideflow/Tensorflow model."""
    return (isdir(path)
            and (exists(join(path, 'params.json'))
                 or exists(join(dirname(path), 'params.json'))))


def is_torch_model_path(path: str) -> bool:
    """Checks if the given path is a valid Slideflow/PyTorch model."""
    return (os.path.isfile(path)
            and sf.util.path_to_ext(path).lower() == 'zip'
            and exists(join(dirname(path), 'params.json')))


def is_simclr_model_path(path: Any) -> bool:
    """Checks if the given path is a valid SimCLR model or checkpoint."""
    is_model =  (isinstance(path, str)
                 and isdir(path)
                 and exists(join(path, 'args.json')))
    is_checkpoint = (isinstance(path, str)
                     and path.endswith('.ckpt')
                     and exists(join(dirname(path), 'args.json')))
    return is_model or is_checkpoint


def is_uq_model(model_path: str) -> bool:
    """Checks if the given model path points to a UQ-enabled model."""
    is_model_path = (is_tensorflow_model_path(model_path)
                     or is_torch_model_path(model_path))
    if not is_model_path:
        return False
    config = get_model_config(model_path)
    return config['hp']['uq']


def assert_is_mag(arg1: str):
    if not isinstance(arg1, str) or not is_mag(arg1):
        raise ValueError(
            f'Invalid magnification {arg1}. Must be of format'
            f' [int/float]x, such as "10x", "20X", or "2.5x"'
        )


def to_mag(arg1: str) -> Union[int, float]:
    assert_is_mag(arg1)
    try:
        return int(arg1.lower().split('x')[0])
    except ValueError:
        return float(arg1.lower().split('x')[0])


def is_tile_size_compatible(
    tile_px1: int,
    tile_um1: Union[str, int],
    tile_px2: int,
    tile_um2: Union[str, int]
) -> bool:
    """Check whether tile sizes are compatible.

    Compatibility is defined as:
        - Equal size in pixels
        - If tile width (tile_um) is defined in microns (int) for both, these must be equal
        - If tile width (tile_um) is defined as a magnification (str) for both, these must be equal
        - If one is defined in microns and the other as a magnification, the calculated magnification must be +/- 2.

    Example 1:
    - tile_px1=299, tile_um1=302
    - tile_px2=299, tile_um2=304
    - Incompatible (unequal micron width)

    Example 2:
    - tile_px1=299, tile_um1=10x
    - tile_px2=299, tile_um2=9x
    - Incompatible (unequal magnification)

    Example 3:
    - tile_px1=299, tile_um1=302
    - tile_px2=299, tile_um2=10x
    - Compatible (first has an equivalent magnification of 9.9x, which is +/- 2 compared to 10x)


    Args:
        tile_px1 (int): Tile size (in pixels) of first slide.
        tile_um1 (int or str): Tile size (in microns) of first slide.
            Can also be expressed as a magnification level, e.g. ``'10x'``
        tile_px2 (int): Tile size (in pixels) of second slide.
        tile_um2 (int or str): Tile size (in microns) of second slide.
            Can also be expressed as a magnification level, e.g. ``'10x'``

    Returns:
        bool: Whether the tile sizes are compatible.

    """
    # Type checks
    if not isinstance(tile_px1, int):
        raise ValueError("Expected tile_px1 to be an int, got: {}".format(type(tile_px1)))
    if not isinstance(tile_um1, (str, int)):
        raise ValueError("Expected tile_um1 to be a str or int, got: {}".format(type(tile_um1)))
    if not isinstance(tile_px2, int):
        raise ValueError("Expected tile_px2 to be an int, got: {}".format(type(tile_px2)))
    if not isinstance(tile_um2, (str, int)):
        raise ValueError("Expected tile_um2 to be a str or int, got: {}".format(type(tile_um2)))

    # Enforce equivalent pixel size
    if tile_px1 != tile_px2:
        return False
    # If both are defined as a magnification, check if these are equal
    if isinstance(tile_um1, str) and isinstance(tile_um2, str):
        return tile_um1 == tile_um2
    # If both are defined in microns, check if these are equal
    if isinstance(tile_um1, int) and isinstance(tile_um2, int):
        return tile_um1 == tile_um2
    # If one is defined in microns and the other as magnification,
    # check if they are compatible.
    if isinstance(tile_um1, str) and isinstance(tile_um2, int):
        mag2 = 10 / (tile_um2 / tile_px2)
        return abs(mag2 - to_mag(tile_um1)) <= 2
    if isinstance(tile_um1, int) and isinstance(tile_um2, str):
        mag1 = 10 / (tile_um1 / tile_px1)
        return abs(mag1 - to_mag(tile_um2)) <= 2
    else:
        raise ValueError("Error assessing tile size compatibility between px={}, um={} and px={}, um={}".format(
            tile_px1, tile_um1, tile_px2, tile_um2
        ))


def multi_warn(arr: List, compare: Callable, msg: Union[Callable, str]) -> int:
    """Logs multiple warning

    Args:
        arr (List): Array to compare.
        compare (Callable): Comparison to perform on array. If True, will warn.
        msg (str): Warning message.

    Returns:
        int: Number of warnings.
    """
    num_warned = 0
    warn_threshold = 3
    for item in arr:
        if compare(item):
            fn = log.warn if num_warned < warn_threshold else log.debug
            if isinstance(msg, str):
                fn(msg.format(item))
            elif callable(msg):
                fn(msg(item))
            num_warned += 1
    if num_warned >= warn_threshold:
        log.warn(f'...{num_warned} total warnings, see log for details')
    return num_warned


def to_onehot(val: int, max: int) -> np.ndarray:
    """Converts value to one-hot encoding

    Args:
        val (int): Value to encode
        max (int): Maximum value (length of onehot encoding)
    """

    onehot = np.zeros(max, dtype=np.int64)
    onehot[val] = 1
    return onehot


def clear_console() -> None:
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def make_dir(_dir: str) -> None:
    """Makes a directory if one does not already exist,
    in a manner compatible with multithreading.
    """
    if not exists(_dir):
        try:
            os.makedirs(_dir, exist_ok=True)
        except FileExistsError:
            pass


def relative_path(path: str, root: str):
    """Returns a relative path, from a given root directory."""
    if path[0] == '.':
        return join(root, path[2:])
    elif path.startswith('$ROOT'):
        raise ValueError("Invalid path prefix $ROOT; update project settings")
    else:
        return path


def global_path(root: str, path_string: str):
    '''Returns global path from a local path.'''
    if not root:
        root = ""
    if path_string and (len(path_string) > 2) and path_string[:2] == "./":
        return os.path.join(root, path_string[2:])
    elif path_string and (path_string[0] != "/"):
        return os.path.join(root, path_string)
    else:
        return path_string


def _shortname(string: str):
    if len(string) == 60:
        # May be TCGA slide with long name; convert to
        # patient name by returning first 12 characters
        return string[:12]
    else:
        return string


def yes_no_input(prompt: str, default: str = 'no') -> bool:
    '''Prompts user for yes/no input.'''
    while True:
        response = input(prompt)
        if not response and default:
            return (default in ('yes', 'y'))
        elif response.lower() in ('yes', 'no', 'y', 'n'):
            return (response.lower() in ('yes', 'y'))
        else:
            print("Invalid response.")


def path_input(
    prompt: str,
    root: str,
    default: Optional[str] = None,
    create_on_invalid: bool = False,
    filetype: Optional[str] = None,
    verify: bool = True
) -> str:
    '''Prompts user for directory input.'''
    while True:
        relative_response = input(f"{prompt}")
        reponse = global_path(root, relative_response)
        if not relative_response and default:
            relative_response = default
            reponse = global_path(root, relative_response)
        if verify and not os.path.exists(reponse):
            if not filetype and create_on_invalid:
                prompt = f'Path "{reponse}" does not exist. Create? [Y/n] '
                if yes_no_input(prompt, default='yes'):
                    os.makedirs(reponse)
                    return relative_response
                else:
                    continue
            elif filetype:
                print(f'Unable to locate file "{reponse}"')
                continue
        elif not filetype and not os.path.exists(reponse):
            print(f'Unable to locate directory "{reponse}"')
            continue
        resp_type = path_to_ext(reponse)
        if filetype and (resp_type != filetype):
            print(f'Incorrect filetype "{resp_type}", expected "{filetype}"')
            continue
        return relative_response


def choice_input(prompt, valid_choices, default=None, multi_choice=False,
                 input_type=str):
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
                replaced = response.replace(" ", "")
                response = [input_type(r) for r in replaced.split(',')]
            except ValueError:
                print(f"Invalid selection (response: {response})")
                continue
            invalid = [r not in valid_choices for r in response]
            if any(invalid):
                print(f'Invalid selection (response: {response})')
                continue
        return response


def load_json(filename: str) -> Any:
    '''Reads JSON data from file.'''
    with open(filename, 'r') as data_file:
        return json.load(data_file)


class ValidJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return "<unknown>"


def write_json(data: Any, filename: str) -> None:
    """Write data to JSON file.

    Args:
        data (Any): Data to write.
        filename (str): Path to JSON file.

    """
    # First, remove any invalid entries that are not serializable
    with open(filename, "w") as data_file:
        json.dump(data, data_file, indent=1, cls=ValidJSONEncoder)


def log_manifest(
    train_tfrecords: Optional[List[str]] = None,
    val_tfrecords: Optional[List[str]] = None,
    *,
    labels: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    remove_extension: bool = True
) -> str:
    """Saves the training manifest in CSV format and returns as a string.

    Args:
        train_tfrecords (list(str)], optional): List of training TFRecords.
            Defaults to None.
        val_tfrecords (list(str)], optional): List of validation TFRecords.
            Defaults to None.

    Keyword args:
        labels (dict, optional): TFRecord outcome labels. Defaults to None.
        filename (str, optional): Path to CSV file to save. Defaults to None.
        remove_extension (bool, optional): Remove file extension from slide
            names. Defaults to True.

    Returns:
        str: Saved manifest in str format.
    """
    out = ''
    has_labels = (isinstance(labels, dict) and len(labels))
    if filename:
        save_file = open(os.path.join(filename), 'w')
        writer = csv.writer(save_file)
        writer.writerow(['slide', 'dataset', 'outcome_label'])
    if train_tfrecords or val_tfrecords:
        if train_tfrecords:
            for tfrecord in train_tfrecords:
                if remove_extension:
                    slide = sf.util.path_to_name(tfrecord)
                else:
                    slide = tfrecord
                outcome_label = labels[slide] if has_labels else 'NA'
                out += ' '.join([slide, 'training', str(outcome_label)])
                if filename:
                    writer.writerow([slide, 'training', outcome_label])
        if val_tfrecords:
            for tfrecord in val_tfrecords:
                if remove_extension:
                    slide = sf.util.path_to_name(tfrecord)
                else:
                    slide = tfrecord
                outcome_label = labels[slide] if has_labels else 'NA'
                out += ' '.join([slide, 'validation', str(outcome_label)])
                if filename:
                    writer.writerow([slide, 'validation', outcome_label])
    if filename:
        save_file.close()
    return out


def get_slides_from_model_manifest(
    model_path: str,
    dataset: Optional[str] = None
) -> List[str]:
    """Get list of slides from a model manifest.

    Args:
        model_path (str): Path to model from which to load the model manifest.
        dataset (str):  'training' or 'validation'. Will return only slides
            from this dataset. Defaults to None (all).

    Returns:
        list(str): List of slide names.
    """

    slides = []
    if exists(join(model_path, 'slide_manifest.csv')):
        manifest = join(model_path, 'slide_manifest.csv')
    elif exists(join(dirname(model_path), 'slide_manifest.csv')):
        log.debug("Slide manifest not found in model directory")
        log.debug("Loading manifest from parent directory.")
        manifest = join(dirname(model_path), 'slide_manifest.csv')
    else:
        log.error('Slide manifest not found in model folder')
        return []
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


def get_gan_config(model_path: str) -> Dict:
    """Loads a GAN training_options.json for an associated network PKL."""

    if exists(join(dirname(model_path), 'training_options.json')):
        return load_json(join(dirname(model_path), 'training_options.json'))
    else:
        raise errors.ModelParamsNotFoundError


def get_model_config(model_path: str) -> Dict:
    """Loads model configuration JSON file."""

    if exists(join(model_path, 'params.json')):
        config = load_json(join(model_path, 'params.json'))
    elif exists(model_path) and exists(join(dirname(model_path), 'params.json')):
        if not (sf.util.torch_available
                and sf.util.path_to_ext(model_path) == 'zip'):
            log.warning(
                "Hyperparameters not in model directory; loading from parent"
                " directory. Please move params.json into model folder."
            )
        config = load_json(join(dirname(model_path), 'params.json'))
    else:
        raise errors.ModelParamsNotFoundError
    # Compatibility for pre-1.1
    if 'norm_mean' in config:
        config['norm_fit'] = {
            'target_means': config['norm_mean'],
            'target_stds': config['norm_std'],
        }
    if 'outcome_label_headers' in config:
        log.debug("Replacing outcome_label_headers in params.json -> outcomes")
        config['outcomes'] = config.pop('outcome_label_headers')
    return config


def get_ensemble_model_config(model_path: str) -> Dict:
    """Loads ensemble model configuration JSON file."""

    if exists(join(model_path, 'ensemble_params.json')):
        config = load_json(join(model_path, 'ensemble_params.json'))
    elif exists(join(dirname(model_path), 'ensemble_params.json')):
        if not (sf.util.torch_available
                and sf.util.path_to_ext(model_path) == 'zip'):
            log.warning(
                "Hyperparameters not in model directory; loading from parent"
                " directory. Please move ensemble_params.json into model folder."
            )
        config = load_json(join(dirname(model_path), 'params.json'))
    else:
        raise errors.ModelParamsNotFoundError
    # Compatibility for pre-1.1
    if 'norm_mean' in config:
        config['norm_fit'] = {
            'target_means': config['norm_mean'],
            'target_stds': config['norm_std'],
        }
    if 'outcome_label_headers' in config:
        log.debug("Replacing outcome_label_headers in params.json -> outcomes")
        config['outcomes'] = config.pop('outcome_label_headers')
    return config


def get_model_normalizer(
    model_path: str
) -> Optional["sf.norm.StainNormalizer"]:
    """Loads and fits normalizer using configuration at a model path."""

    config = sf.util.get_model_config(model_path)
    if is_torch_model_path(model_path):
        backend = 'torch'
    elif is_tensorflow_model_path(model_path):
        backend = 'tensorflow'
    else:
        log.warn(f"Unable to determine backend for model at {model_path}")
        backend = None

    if not config['hp']['normalizer']:
        return None

    if ('slideflow_version' in config
       and version.parse(config['slideflow_version']) <= version.parse("1.2.2")
       and config['hp']['normalizer'] in ('vahadane', 'macenko')):
        log.warn("Detected model trained with Macenko or Vahadane "
                 "normalization with Slideflow version <= 1.2.2. Macenko "
                 "and Vahadane algorithms were optimized in 1.2.3 and may "
                 "now yield slightly different results. ")

    normalizer = sf.norm.autoselect(
        config['hp']['normalizer'],
        config['hp']['normalizer_source'],
        backend=backend
    )
    if 'norm_fit' in config and config['norm_fit'] is not None:
        normalizer.set_fit(**config['norm_fit'])
    return normalizer


def get_preprocess_fn(model_path: str):
    """Returns a function which preprocesses a uint8 image for a model.

    Args:
        model_path (str): Path to a saved Slideflow model.

    Returns:
        A function which accepts a single image or batch of uint8 images,
        and returns preprocessed (and stain normalized) float32 images.

    """
    normalizer = get_model_normalizer(model_path)
    if is_torch_model_path(model_path):
        from slideflow.io.torch import preprocess_uint8
        return partial(preprocess_uint8, normalizer=normalizer)
    elif is_tensorflow_model_path(model_path):
        from slideflow.io.tensorflow import preprocess_uint8
        return partial(preprocess_uint8, normalizer=normalizer, as_dict=False)
    else:
        raise ValueError(f"Unrecognized model: {model_path}")


def get_slide_paths(slides_dir: str) -> List[str]:
    '''Get all slide paths from a given directory containing slides.'''
    slide_list = [i for i in glob(join(slides_dir, '**/*.*')) if is_slide(i)]
    slide_list.extend([i for i in glob(join(slides_dir, '*.*')) if is_slide(i)])
    return slide_list


def read_annotations(path: str) -> Tuple[List[str], List[Dict]]:
    '''Read an annotations file.'''
    results = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # First, try to open file
        try:
            header = next(csv_reader, None)
        except OSError:
            raise OSError(
                f"Failed to open annotations file {path}"
            )
        assert isinstance(header, list)
        for row in csv_reader:
            row_dict = {}
            for i, key in enumerate(header):
                row_dict[key] = row[i]
            results += [row_dict]
    return header, results


def get_relative_tfrecord_paths(root: str, directory: str = "") -> List[str]:
    '''Returns relative tfrecord paths with respect to the given directory.'''

    tfrecords = [
        join(directory, f) for f in os.listdir(join(root, directory))
        if (not isdir(join(root, directory, f))
            and len(f) > 10 and f[-10:] == ".tfrecords")
    ]
    subdirs = [
        f for f in os.listdir(join(root, directory))
        if isdir(join(root, directory, f))
    ]
    for sub in subdirs:
        tfrecords += get_relative_tfrecord_paths(root, join(directory, sub))
    return tfrecords


def contains_nested_subdirs(directory: str) -> bool:
    subdirs = [
        _dir for _dir in os.listdir(directory)
        if isdir(join(directory, _dir))
    ]
    for subdir in subdirs:
        contents = os.listdir(join(directory, subdir))
        for c in contents:
            if isdir(join(directory, subdir, c)):
                return True
    return False


def path_to_name(path: str) -> str:
    '''Returns name of a file, without extension,
    from a given full path string.'''
    _file = path.split('/')[-1]
    dot_split = _file.split('.')
    if len(dot_split) == 1:
        return _file
    elif len(dot_split) > 2 and '.'.join(dot_split[-2:]) in SUPPORTED_FORMATS:
        return '.'.join(dot_split[:-2])
    else:
        return '.'.join(dot_split[:-1])


def path_to_ext(path: str) -> str:
    '''Returns extension of a file path string.'''
    _file = path.split('/')[-1]
    dot_split = _file.split('.')
    if len(dot_split) == 1:
        return ''
    elif len(dot_split) > 2 and '.'.join(dot_split[-2:]) in SUPPORTED_FORMATS:
        return '.'.join(dot_split[-2:])
    else:
        return dot_split[-1]


def update_results_log(
    results_log_path: str,
    model_name: str,
    results_dict: Dict
) -> None:
    '''Dynamically update results_log when recording training metrics.'''
    # First, read current results log into a dictionary
    results_log = {}  # type: Dict[str, Any]
    if exists(results_log_path):
        with open(results_log_path, "r") as results_file:
            reader = csv.reader(results_file)
            try:
                headers = next(reader)
            except StopIteration:
                pass
            else:
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


def map_values_to_slide_grid(
    locations: np.ndarray,
    values: np.ndarray,
    wsi: "sf.WSI",
    background: str = 'min',
    *,
    interpolation: Optional[str] = 'bicubic',
) -> np.ndarray:
    """Map heatmap values to a slide grid, using tile location information.

    Args:
        locations (np.ndarray): Array of shape ``(n_tiles, 2)`` containing x, y
            coordinates for all image tiles. Coordinates represent the center
            for an associated tile, and must be in a grid.
        values (np.ndarray): Array of shape ``(n_tiles,)`` containing heatmap
            values for each tile.
        wsi (slideflow.wsi.WSI): WSI object.

    Keyword args:
        background (str, optional): Background strategy for heatmap. Can be
            'min', 'mean', 'median', 'max', or 'mask'. Defaults to 'min'.
        interpolation (str, optional): Interpolation strategy for smoothing
            heatmap. Defaults to 'bicubic'.

    """
    no_interpolation = (interpolation is None or interpolation == 'nearest')

    # Slide coordinate information
    loc_grid_dict = {(c[0], c[1]): (c[2], c[3]) for c in wsi.coord}

    # Determine the heatmap background
    grid = np.empty((wsi.grid.shape[1], wsi.grid.shape[0]))
    if background == 'mask' and not no_interpolation:
        raise ValueError(
            "'mask' background is not compatible with interpolation method "
            "'{}'. Expected: None or 'nearest'".format(interpolation)
        )
    elif background == 'mask':
        grid[:] = np.nan
    elif background == 'min':
        grid[:] = np.min(values)
    elif background == 'mean':
        grid[:] = np.mean(values)
    elif background == 'median':
        grid[:] = np.median(values)
    elif background == 'max':
        grid[:] = np.max(values)
    else:
        raise ValueError(f"Unrecognized value for background: {background}")

    if not isinstance(locations, np.ndarray):
        locations = np.array(locations)

    # Transform from coordinates as center locations to top-left locations.
    locations = locations - int(wsi.full_extract_px/2)

    for i, wsi_dim in enumerate(locations):
        try:
            idx = loc_grid_dict[tuple(wsi_dim)]
        except (IndexError, KeyError):
            raise errors.CoordinateAlignmentError(
                "Error plotting value at location {} for slide {}. The heatmap "
                "grid is not aligned to the slide coordinate grid. Ensure "
                "that tile_px (got: {}) and tile_um (got: {}) match the given "
                "location values. If you are using data stored in TFRecords, "
                "verify that the TFRecord was generated using the same "
                "tile_px and tile_um.".format(
                    tuple(wsi_dim), wsi.path, wsi.tile_px, wsi.tile_um
                )
            )
        grid[idx[1]][idx[0]] = values[i]

    # Mask out background, if interpolation is not used and background == 'mask'
    if no_interpolation and background == 'mask':
        masked_grid = np.ma.masked_invalid(grid)
    else:
        masked_grid = grid
    return masked_grid


def bin_values_to_slide_grid(
    locations: np.ndarray,
    values: np.ndarray,
    wsi: "sf.WSI",
    background: str = 'min',
) -> np.ndarray:
    """Bin heatmap values to a slide grid, using tile location information.

    Args:
        locations (np.ndarray): Array of shape ``(n_tiles, 2)`` containing x, y
            coordinates for all image tiles. Coordinates represent the center
            for an associated tile, and must be in a grid.
        values (np.ndarray): Array of shape ``(n_tiles,)`` containing heatmap
            values for each tile.
        wsi (slideflow.wsi.WSI): WSI object.

    Keyword args:
        background (str, optional): Background strategy for heatmap. Can be
            'min', 'mean', 'median', 'max', or 'mask'. Defaults to 'min'.

    """
    from scipy.stats import binned_statistic_2d
    masked_grid, *_ = binned_statistic_2d(
        locations[:, 0],
        locations[:, 1],
        values,
        bins=wsi.grid.shape,
        range=[[0, wsi.dimensions[0]], [0, wsi.dimensions[1]]]
    )
    masked_grid = masked_grid.T
    nan_idx = np.where(np.isnan(masked_grid))

    if background == 'mask':
        # No action needed
        pass
    elif background == 'min':
        masked_grid[nan_idx] = np.min(values)
    elif background == 'mean':
        masked_grid[nan_idx] = np.mean(values)
    elif background == 'median':
        masked_grid[nan_idx] = np.median(values)
    elif background == 'max':
        masked_grid[nan_idx] = np.max(values)
    else:
        raise ValueError(f"Unrecognized value for background: {background}")

    return masked_grid


def infer_stride(locations, wsi):
    """Infer the stride of a grid of locations from a set of locations.

    Args:
        locations (np.ndarray): Nx2 array of locations
        wsi (slideflow.wsi.WSI): WSI object

    Returns:
        float: inferred stride divisor in pixels

    """
    sort_unique_x = np.sort(np.unique(locations[:, 0]))
    sort_unique_y = np.sort(np.unique(locations[:, 1]))
    min_stride_x = (sort_unique_x[1:] - sort_unique_x[:-1]).min()
    min_stride_y = (sort_unique_y[1:] - sort_unique_y[:-1]).min()
    inferred_stride_px = min(min_stride_x, min_stride_y)
    return wsi.full_extract_px / inferred_stride_px


def location_heatmap(
    locations: np.ndarray,
    values: np.ndarray,
    slide: str,
    tile_px: int,
    tile_um: Union[int, str],
    outdir: str,
    *,
    interpolation: Optional[str] = 'bicubic',
    cmap: str = 'inferno',
    norm: Optional[str] = None,
    background: str = 'min'
) -> Dict[str, Dict[str, float]]:
    """Generate a heatmap for a slide.

    Args:
        locations (np.ndarray): Array of shape ``(n_tiles, 2)`` containing x, y
            coordinates for all image tiles. Coordinates represent the center
            for an associated tile, and must be in a grid.
        values (np.ndarray): Array of shape ``(n_tiles,)`` containing heatmap
            values for each tile.
        slide (str): Path to corresponding slide.
        tile_px (int): Tile pixel size.
        tile_um (int, str): Tile micron or magnification size.
        outdir (str): Directory in which to save heatmap.

    Keyword args:
        interpolation (str, optional): Interpolation strategy for smoothing
            heatmap. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcol

    slide_name = sf.util.path_to_name(slide)
    log.info(f'Generating heatmap for [green]{slide}[/]...')
    log.debug(f"Plotting {len(values)} values")
    wsi = sf.WSI(slide, tile_px, tile_um, verbose=False)
    stride = infer_stride(locations, wsi)
    if stride > 32:
        # Large inferred strides are likely due to unaligned grid.
        # Rather than attempting to build a coordinate grid for verifying
        # grid alignment, we will assume that the grid is unaligned and
        # use the default stride (1). This will cause map_values_to_slide_grid
        # to recognize that the grid is unaligned, and the heatmap will be built
        # using histogram2d.
        log.debug(f"Failed sanity check for inferred stride ({stride})")
    elif stride != 1:
        log.debug(f"Inferred stride: {stride}")
        wsi = sf.WSI(slide, tile_px, tile_um, stride_div=stride, verbose=False)

    stats = {
        slide_name: {
            'mean': np.mean(values),
            'median': np.median(values)
        }
    }

    try:
        masked_grid = map_values_to_slide_grid(
            locations, values, wsi, background=background, interpolation=interpolation
        )
    except errors.CoordinateAlignmentError as e:
        log.debug("Coordinate alignment error: {}".format(e))
        log.info("Unable to align grid for plotting heatmap. Heatmap will be "
                  "binned with a stride of 1.")
        masked_grid = bin_values_to_slide_grid(
            locations, values, wsi, background=background
        )

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.25, top=0.95)
    gca = plt.gca()
    gca.tick_params(
        axis='x',
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False
    )
    thumb = wsi.thumb(mpp=5)
    ax.imshow(thumb, zorder=0)

    # Calculate overlay offset
    extent = sf.heatmap.calculate_heatmap_extent(wsi, thumb, masked_grid)

    # Plot
    if norm == 'two_slope':
        norm = mcol.TwoSlopeNorm(
            vmin=min(-0.01, min(values)),
            vcenter=0,
            vmax=max(0.01, max(values))
        )
    ax.imshow(
        masked_grid,
        zorder=10,
        alpha=0.6,
        extent=extent,
        interpolation=interpolation,
        cmap=cmap,
        norm=norm
    )
    ax.set_xlim(0, thumb.size[0])
    ax.set_ylim(thumb.size[1], 0)
    log.debug('Saving figure...')
    plt.savefig(join(outdir, f'{slide_name}_attn.png'), bbox_inches='tight')
    plt.close(fig)
    del wsi
    del thumb
    return stats


def tfrecord_heatmap(
    tfrecord: str,
    slide: str,
    tile_px: int,
    tile_um: Union[int, str],
    tile_dict: Dict[int, float],
    outdir: str,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """Creates a tfrecord-based WSI heatmap using a dictionary of tile values
    for heatmap display.

    Args:
        tfrecord (str): Path to tfrecord.
        slide (str): Path to whole-slide image.
        tile_dict (dict): Dictionary mapping tfrecord indices to a
            tile-level value for display in heatmap format.
        tile_px (int): Tile width in pixels.
        tile_um (int or str): Tile width in microns (int) or magnification
            (str, e.g. "20x").
        outdir (str): Path to directory in which to save images.

    Returns:
        Dictionary mapping slide names to dict of statistics
        (mean, median)
    """
    locations = sf.io.get_locations_from_tfrecord(tfrecord)
    if len(tile_dict) != len(locations):
        raise errors.TFRecordsError(
            f'tile_dict length ({len(tile_dict)}) != TFRecord length '
            f'({len(locations)}).'
        )

    return location_heatmap(
        locations=np.array(locations),
        values=np.array([tile_dict[loc] for loc in range(len(locations))]),
        slide=slide,
        tile_px=tile_px,
        tile_um=tile_um,
        outdir=outdir,
        **kwargs
    )


def tile_size_label(tile_px: int, tile_um: Union[str, int]) -> str:
    """Return the string label of the given tile size."""
    if isinstance(tile_um, str):
        return f"{tile_px}px_{tile_um.lower()}"
    else:
        return f"{tile_px}px_{tile_um}um"


def get_valid_model_dir(root: str) -> List:
    '''
    This function returns the path of the first indented directory from root.
    This only works when the indented folder name starts with a 5 digit number,
    like "00000%".

    Examples
        If the root has 3 files:
        root/00000-foldername/
        root/00001-foldername/
        root/00002-foldername/

        The function returns "root/00000-foldername/"
    '''

    prev_run_dirs = [
        x for x in os.listdir(root)
        if isdir(join(root, x))
    ]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    return prev_run_ids, prev_run_dirs


def get_new_model_dir(root: str, model_name: str) -> str:
    prev_run_ids, prev_run_dirs = get_valid_model_dir(root)
    cur_id = max(prev_run_ids, default=-1) + 1
    model_dir = os.path.join(root, f'{cur_id:05d}-{model_name}')
    assert not os.path.exists(model_dir)
    os.makedirs(model_dir)
    return model_dir


def create_new_model_dir(root: str, model_name: str) -> str:
    path = get_new_model_dir(root, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def split_list(a: List, n: int) -> List[List]:
    '''Function to split a list into n components'''
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
            for i in range(n)]


# --- TFRecord utility functions ----------------------------------------------

def process_feature(
    feature: example_pb2.Feature,  # type: ignore
    typename: str,
    typename_mapping: Dict,
    key: str
) -> np.ndarray:
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]  # type: ignore
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(
                f"Incompatible type '{typename}' for `{key}` "
                f"(should be '{reversed_mapping[inferred_typename]}')."
            )

    if inferred_typename == "bytes_list":
        value = np.frombuffer(value[0], dtype=np.uint8)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int64)
    return value


def extract_feature_dict(
    features: Union[example_pb2.FeatureLists,  # type: ignore
                    example_pb2.Features],  # type: ignore
    description: Optional[Union[List, Dict]],
    typename_mapping: Dict
) -> Dict[str, Any]:
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list  # type: ignore

        def get_value(typename, typename_mapping, key):
            feature = features[key].feature
            fn = partial(
                process_feature,
                typename=typename,
                typename_mapping=typename_mapping,
                key=key
            )
            return list(map(fn, feature))
    elif isinstance(features, example_pb2.Features):
        features = features.feature  # type: ignore

        def get_value(typename, typename_mapping, key):
            return process_feature(features[key], typename,
                                   typename_mapping, key)
    else:
        raise TypeError(f"Incompatible type: features should be either of type "
                        f"example_pb2.Features or example_pb2.FeatureLists and "
                        f"not {type(features)}")

    all_keys = list(features.keys())  # type: ignore

    if description is None or len(description) == 0:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

        processed_features[key] = get_value(typename, typename_mapping, key)

    return processed_features


def load_predictions(path: str, **kwargs) -> pd.DataFrame:
    """Loads a 'csv', 'parquet' or 'feather' file to a pandas dataframe.

    Args:
        path (str): Path to the file to be read.

    Returns:
        df (pd.DataFrame): The dataframe read from the path.
    """
    if path.endswith("csv"):
        return pd.read_csv(f"{path}", **kwargs)
    elif path.endswith("parquet") or path.endswith("gzip"):
        return pd.read_parquet(f"{path}", **kwargs)
    elif path.endswith("feather"):
        return pd.read_feather(f"{path}", **kwargs)
    else:
        raise ValueError(f'Unrecognized extension "{path_to_ext(path)}"')


@contextmanager
def cleanup_progress(pb: Optional["Progress"]):
    try:
        yield
    finally:
        if pb is not None:
            pb.refresh()
            pb.stop()


def create_triangles(vertices, hole_vertices=None, hole_points=None):
    """
    Tessellate a complex polygon, possibly with holes.

    :param vertices: A list of vertices [(x1, y1), (x2, y2), ...] defining the polygon boundary.
    :param holes: An optional list of points [(hx1, hy1), (hx2, hy2), ...] inside each hole in the polygon.
    :return: A numpy array of vertices for the tessellated triangles.
    """
    import triangle as tr

    # Prepare the segment information for the exterior boundary
    segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])

    # Prepare the polygon for Triangle
    polygon = {'vertices': np.array(vertices), 'segments': segments}

    # If there are holes and hole boundaries, add them to the polygon definition
    if hole_points is not None and hole_vertices is not None and len(hole_vertices):
        polygon['holes'] = np.array(hole_points).astype(np.float32)

        # Start adding hole segments after the exterior segments
        start_idx = len(vertices)
        for hole in hole_vertices:
            hole_segments = [[start_idx + i, start_idx + (i + 1) % len(hole)] for i in range(len(hole))]
            segments = np.vstack([segments, hole_segments])
            start_idx += len(hole)

        # Update the vertices and segments in the polygon
        all_vertices = np.vstack([vertices] + hole_vertices)
        polygon['vertices'] = all_vertices
        polygon['segments'] = segments

    # Tessellate the polygon
    tess = tr.triangulate(polygon, 'pF')

    # Extract tessellated triangle vertices
    if 'triangles' not in tess:
        return None

    tessellated_vertices = np.array([tess['vertices'][t] for t in tess['triangles']]).reshape(-1, 2)

    # Convert to float32
    tessellated_vertices = tessellated_vertices.astype('float32')

    return tessellated_vertices

def get_splits(splits_file, id=None):
    """
    Load splits from a file, returning the split corresponding to the given ID.
    :param splits_file: Path to the splits file.
    :param id: The ID of the split to return. Default to None, which returns all splits.
    :return: The splits dictionary or the split corresponding to the given ID.
    """
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"File not found: {splits_file}")
    splits = load_json(splits_file)
    if id is not None and id >= len(splits):
        raise IndexError(f"Split ID {id} is out of range (0-{len(splits)-1}).")
    return splits if id is None else splits[id]