# +-----------------------------------------+
# | Written and maintained by James Dolezal |
# | james.dolezal@uchospitals.edu           |
# +-----------------------------------------+

import os

__version__ = "1.12.3"

if 'SF_BACKEND' not in os.environ:
    os.environ['SF_BACKEND'] = 'tensorflow'

import slideflow.io
import slideflow.model
from slideflow.activations import ActivationsInterface, ActivationsVisualizer, Heatmap
from slideflow.dataset import Dataset
from slideflow.mosaic import Mosaic
from slideflow.project import Project
from slideflow.slide import WSI, TMA
from slideflow.statistics import SlideMap

def backend():
    return os.environ['SF_BACKEND']

# Style information
# =================
# General style format should conform to Google Python best practices
# (http://google.github.io/styleguide/pyguide.html), with the exception of a
# maximum line length of 120. Docstrings should also conform with Google Style.
# (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
# A pylintrc file is included is the root directory to assist with formatting.

# Version planning (v1.12)
#TODO: finish a couple more tutorials
#       - Tutorial 2: Heatmaps, Mosaic maps, ActivationsVisualizer
#       - Tutorial 3: CLAM
#       - Tutorial 4: Cancer non-cancer
#       - Tutorial 5: Comparing normalizers
#       - Tutorial 6: Multi-outcome models
#       - Tutorial 7: Clinical models, CPH outcome, permutation feature importance
#       - Tutorial 8: Hyperparameter sweeps
#       - Tutorial 9: Class-conditional GAN with StyleGAN2
#TODO: label_parser in dataset.tfrecords()
#TODO: consistent model name strings in tensorflow and pytorch versions
#TODO: easier validation plan sharing
#TODO: more clear logging information regarding validation plans
#TODO: show tile extraction grid on thumbnail, optional
#TODO: consistent "decode" for get_tfrecord_parser in tensorflow/torch (decode should decode images + slide name)
#TODO: ActivationsInterface compatibility for multiple outcomes
#TODO: log normalization as hyperparameter
#TODO: custom models in Tensorflow & PyTorch (attention to ActivationsInterface)
#TODO: remove as many calls to sf.backend() as possible. Ideally the API should be unified/consistent
#TODO: improve tile verification speed in PyTorch
#TODO: ensure format of results_log is the same for train(), evaluate(), tensorflow & pytorch

# PyTorch implementation
# ======================
#
# Core features / high priority updates
# -------------------------------------
#TODO: full Trainer features, parameters, etc
#    - log_frequency, ema_observations, ema_smoothing, use_tensorboard, resume_training, checkpoint

# May be delayed:
# ---------------
#TODO: CPH outcomes
#TODO: slide-level input
#   - TODO: statistics.permutation_feature_importance (-> ActivationsInterface)
#   - TODO: statistics.predict_from_layer (used for permutation_feature_importance)
#TODO: multi-GPU support
#
# Low priority consistency/style changes:
# ---------------------
#TODO: PyTorch ModelParams get_loss -> @property
#TODO: consider pytorch to_numpy=False returns tensor objects
#TODO: for tfrecord parser, combine utf-8 and image decoding into single `decode` argument (rather than decode_images)
#TODO: filter.py script
#TODO: update.py script
#TODO: resize_tfrecords()
#TODO: merge annotated_thumb and thumb

# Future updates
# ===============
#TODO: implement native TF normalizers to improve realtime normalization speed
#TODO: put tfrecord report in tfrecord directories & include information
#         on normalization, filtering, slideflow version, etc
#TODO: consider multithreading sf.tfrecord.reader:255-258 (parsing of records)