import os

__version__ = "1.0.1"

if 'SF_BACKEND' not in os.environ:
    os.environ['SF_BACKEND'] = 'tensorflow'

import slideflow.io
import slideflow.model
from slideflow.heatmap import Heatmap
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
# maximum line length of 120 where possible. Docstrings should also conform with Google Style.
# (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
# A pylintrc file is included is the root directory to assist with formatting.

# Version planning (v1.13)
#TODO: finish a couple more tutorials / jupyter notebooks
#       - Tutorial 2: Heatmaps, Mosaic maps, DatasetFeatures
#       - Tutorial 3: CLAM
#       - Tutorial 4: Cancer non-cancer
#       - Tutorial 5: Comparing normalizers
#       - Tutorial 6: Multi-outcome models
#       - Tutorial 7: Clinical models, CPH outcome, permutation feature importance
#       - Tutorial 8: Hyperparameter sweeps
#       - Tutorial 9: Class-conditional GAN with StyleGAN2

# Low priority consistency/style changes:
# ---------------------
#TODO: PyTorch ModelParams get_loss -> @property
#TODO: consider pytorch to_numpy=False returns tensor objects
#TODO: for tfrecord parser, combine utf-8 and image decoding into single `decode` argument (rather than decode_images)
#TODO: filter.py script
#TODO: update.py script
#TODO: resize_tfrecords()

# Future updates
# ===============
#TODO: Features compatibility for multiple outcomes
#TODO: PyTorch CPH outcomes
#TODO: statistics.permutation_feature_importance (-> Features)
#TODO: statistics.predict_from_layer (used for permutation_feature_importance)
#TODO: implement native TF normalizers to improve realtime normalization speed