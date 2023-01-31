# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from ._version import get_versions
import os
import importlib

__author__ = 'James Dolezal'
__license__ = 'GNU General Public License v3.0'
__version__ = get_versions()['version']
__gitcommit__ = get_versions()['full-revisionid']
__github__ = 'https://github.com/jamesdolezal/slideflow'

# --- Backend configuration ---------------------------------------------------

# Deep learning backend - use Tensorflow if available.
_valid_backends = ('tensorflow', 'torch')
if 'SF_BACKEND' not in os.environ:
    if importlib.util.find_spec('tensorflow'):
        os.environ['SF_BACKEND'] = 'tensorflow'
    elif importlib.util.find_spec('torch'):
        os.environ['SF_BACKEND'] = 'torch'
    else:
        os.environ['SF_BACKEND'] = 'tensorflow'
elif os.environ['SF_BACKEND'] not in _valid_backends:
    raise ValueError("Unrecognized backend set via environmental variable "
                     "SF_BACKEND: {}. Expected one of: {}".format(
                        os.environ['SF_BACKEND'],
                        ', '.join(_valid_backends)
                     ))

# Slide backend - use cuCIM if available.
_valid_slide_backends = ('cucim', 'libvips')
if 'SF_SLIDE_BACKEND' not in os.environ:
    os.environ['SF_SLIDE_BACKEND'] = 'libvips'
    if importlib.util.find_spec('cucim'):
        import cucim
        if cucim.is_available():
            os.environ['SF_SLIDE_BACKEND'] = 'cucim'
elif os.environ['SF_SLIDE_BACKEND'] not in _valid_slide_backends:
    raise ValueError("Unrecognized slide backend set via environmental variable"
                     " SF_SLIDE_BACKEND: {}. Expected one of: {}".format(
                        os.environ['SF_SLIDE_BACKEND'],
                        ', '.join(_valid_slide_backends)
                     ))

def backend():
    return os.environ['SF_BACKEND']

def slide_backend():
    return os.environ['SF_SLIDE_BACKEND']

# -----------------------------------------------------------------------------

# Import logging functions required for other submodules
from slideflow.util import getLoggingLevel, log, setLoggingLevel, about

...
from slideflow import io, model, norm, stats
from slideflow.dataset import Dataset
from slideflow.heatmap import Heatmap
from slideflow.model import DatasetFeatures, ModelParams
from slideflow.mosaic import Mosaic
from slideflow.project import Project
from slideflow.slide import TMA, WSI
from slideflow.stats import SlideMap
