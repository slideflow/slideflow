# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import os

__author__ = 'James Dolezal'
__license__ = 'GNU General Public License v3.0'
__version__ = "1.0.3"

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

# Future updates
# ===============
#TODO: Features compatibility for multiple outcomes
#TODO: PyTorch CPH outcomes
#TODO: PyTorch statistics.permutation_feature_importance (-> Features)
#TODO: PyTorch statistics.predict_from_layer (used for permutation_feature_importance)
#TODO: implement native TF/PyTorch normalizers to improve realtime normalization speed
#TODO: improve estimated_num_tiles when doing tile extraction & no ROI (or QC)
#TODO: for tfrecord parser, combine utf-8 and image decoding into single `decode` argument (rather than decode_images)
#TODO: consider pytorch to_numpy=False returns tensor objects