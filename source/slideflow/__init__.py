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
#TODO: pytorch backend
#TODO: finish a couple more tutorials
#       - Tutorial 2: Heatmaps, Mosaic maps, ActivationsVisualizer
#       - Tutorial 3: CLAM
#       - Tutorial 4: Cancer non-cancer
#       - Tutorial 5: Comparing normalizers
#       - Tutorial 6: Multi-outcome models
#       - Tutorial 7: Clinical models, CPH outcome, permutation feature importance
#       - Tutorial 8: Hyperparameter sweeps
#       - Tutorial 9: Class-conditional GAN with StyleGAN2
#TODO: implement clipping for tfrecord interleaving in pytorch
#TODO: label_parser in dataset.tfrecords()
#TODO: consistent model name strings in tensorflow and pytorch versions
#TODO: fix pytorch validate_on_batch (should skip if == 0, and should use val_steps not the whole dataset)
#TODO: easier validation plan sharing
#TODO: more clear logging information regarding validation plans
#TODO: pytorch implementation does not log results to results_log
#TODO: remove tensorflow requirement from slideflow.io.tensorflow.update_manifest_at_dir() and dataset.update_manifest()
#TODO: make thumbnail caching optional
#TODO: show tile extraction grid on thumbnail, optional
#TODO: merge annotated_thumb and thumb

# PyTorch implementation
# ======================
#TODO: full Trainer features

# Future updates
# ===============
#TODO: implement native TF normalizers to improve realtime normalization speed
#TODO: put tfrecord report in tfrecord directories & include information
#         on normalization, filtering, slideflow version, etc
#TODO: neptune integration