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
__version__ = "1.1.0-rc1"

if 'SF_BACKEND' not in os.environ:
    os.environ['SF_BACKEND'] = 'tensorflow'

def backend():
    return os.environ['SF_BACKEND']

from slideflow import io
from slideflow import model
from slideflow import norm
from slideflow.heatmap import Heatmap
from slideflow.dataset import Dataset
from slideflow.mosaic import Mosaic
from slideflow.project import Project
from slideflow.slide import WSI, TMA
from slideflow.stats import SlideMap
