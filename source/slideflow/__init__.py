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
# General style format should conform to Google Python best-practices
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
#TODO: make thumbnail caching optional
#TODO: show tile extraction grid on thumbnail, optional
#TODO: merge annotated_thumb and thumb
#TODO: consistent "decode" for get_tfrecord_parser in tensorflow/torch (decode should decode images + slide name)
#TODO: ActivationsInterface compatibility for multiple outcomes
#TODO: implement __repr__ methods for CLI usability
#TODO: log normalization as hyperparameter
#TODO: custom models in Tensorflow & PyTorch (attention to ActivationsInterface)
#TODO: improved module loading. Look into importlib lazy loading
#TODO: remove as many calls to sf.backend() as possible. Ideally the API should be unified/consistent
#TODO: improve tile verification speech in PyTorch

# PyTorch implementation
# ======================
#
# Core features / high priority updates
# -------------------------------------
#TODO: full Trainer features, parameters, etc
#    - multiple outcomes
#    - CPH outcomes
#    - slide-level input
#    - multi-GPU support
#    - log_frequency, ema_observations, ema_smoothing, use_tensorboard, skip_metrics, resume_training, checkpoint
#TODO: results_dict from torch training
#TODO: statistics.permutation_feature_importance (-> ActivationsInterface)
#TODO: statistics.predict_from_layer (used for permutation_feature_importance)
#TODO: fix pytorch validate_on_batch (should skip if == 0, and should use val_steps not the whole dataset)
#TODO: implement clipping for tfrecord interleaving in pytorch
#TODO: pytorch implementation does not log results to results_log
#TODO: test suite GPU availability test
#TODO: move hyperparameters file into model ZIP or suppress warning
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

# Future updates
# ===============
#TODO: implement native TF normalizers to improve realtime normalization speed
#TODO: put tfrecord report in tfrecord directories & include information
#         on normalization, filtering, slideflow version, etc