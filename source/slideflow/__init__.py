# +-----------------------------------------+
# | Written and maintained by James Dolezal |
# | james.dolezal@uchospitals.edu           |
# +-----------------------------------------+

import os

__version__ = "1.12.3"

if 'SF_BACKEND' not in os.environ:
    os.environ['SF_BACKEND'] = 'tensorflow'
from slideflow.project import Project

def backend():
    return os.environ['SF_BACKEND']

def set_backend(b):
    """Sets the slideflow backend to either tensorflow or pytorch using
    the environmental variable SF_BACKEND

    Args:
        backend (str): Either 'tensorflow' or 'torch'.
    """

    if b not in ('tensorflow', 'torch'):
        raise ValueError(f'Unknown backend {b}')
    os.environ['SF_BACKEND'] = b

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
#TODO: merge annotated_thumb and thumb
#TODO: consistent "decode" for get_tfrecord_parser in tensorflow/torch (decode should decode images + slide name)
#TODO: ActivationsInterface compatibility for multiple outcomes
#TODO: implement __repr__ methods for CLI usability
#TODO: log normalization as hyperparameter
#TODO: custom models in Tensorflow & PyTorch (attention to ActivationsInterface)
#TODO: improved module loading. Look into importlib lazy loading
#TODO: remove as many calls to sf.backend() as possible. Ideally the API should be unified/consistent
#TODO: improve tile verification speed in PyTorch
#TODO: pytorch dataloader memory efficiency & performance
#TODO: ensure format of results_log is the same for train(), evaluate(), tensorflow & pytorch

# PyTorch implementation
# ======================
#
# Core features / high priority updates
# -------------------------------------
#TODO: full Trainer features, parameters, etc
#    - log_frequency, ema_observations, ema_smoothing, use_tensorboard, resume_training, checkpoint
#TODO: implement clipping for tfrecord interleaving in pytorch
#
# Slide processing (tf.data.TFRecordDataset & tf.data.TFRecordWriter)
# -------------------------------------------------------------------
#TODO: dataset.split_tfrecords_by_roi()
#TODO: dataset.tfrecord_report()
#
# Low priority updates:
# ---------------------
#TODO: filter.py script
#TODO: update.py script
#TODO: PyTorch ModelParams get_loss -> @property
#
# May be delayed:
# ---------------
#TODO: CPH outcomes
#TODO: slide-level input
#   - TODO: statistics.permutation_feature_importance (-> ActivationsInterface)
#   - TODO: statistics.predict_from_layer (used for permutation_feature_importance)
#TODO: multi-GPU support

# Future updates
# ===============
#TODO: implement native TF normalizers to improve realtime normalization speed
#TODO: put tfrecord report in tfrecord directories & include information
#         on normalization, filtering, slideflow version, etc