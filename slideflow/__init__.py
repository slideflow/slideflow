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
__version__ = "1.1.2"

import os

if 'SF_BACKEND' not in os.environ:
    os.environ['SF_BACKEND'] = 'tensorflow'


def backend():
    return os.environ['SF_BACKEND']


from slideflow import io  # noqa # pylint: disable=unused-import
from slideflow import model  # noqa # pylint: disable=unused-import
from slideflow import norm  # noqa # pylint: disable=unused-import
from slideflow.model import ModelParams  # noqa # pylint: disable=unused-import
from slideflow.heatmap import Heatmap  # noqa # pylint: disable=unused-import
from slideflow.dataset import Dataset  # noqa # pylint: disable=unused-import
from slideflow.mosaic import Mosaic  # noqa # pylint: disable=unused-import
from slideflow.project import Project  # noqa # pylint: disable=unused-import
from slideflow.slide import WSI, TMA  # noqa # pylint: disable=unused-import
from slideflow.stats import SlideMap  # noqa # pylint: disable=unused-import
