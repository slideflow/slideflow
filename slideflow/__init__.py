# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

__author__ = 'James Dolezal'
__license__ = 'GNU General Public License v3.0'
__version__ = "1.2.3"

import os

if 'SF_BACKEND' not in os.environ:
    os.environ['SF_BACKEND'] = 'tensorflow'


def backend():
    return os.environ['SF_BACKEND']

# Import logging functions required for other submodules
from slideflow.util import getLoggingLevel, log, setLoggingLevel

...
from slideflow import io, model, norm, stats
from slideflow.dataset import Dataset
from slideflow.heatmap import Heatmap
from slideflow.model import DatasetFeatures, ModelParams
from slideflow.mosaic import Mosaic
from slideflow.project import Project
from slideflow.slide import TMA, WSI
from slideflow.stats import SlideMap
