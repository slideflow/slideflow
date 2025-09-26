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

__author__ = 'James Dolezal'
__license__ = 'GNU General Public License v3.0'
__version__ = get_versions()['version']
__gitcommit__ = get_versions()['full-revisionid']
__github__ = 'https://github.com/slideflow/slideflow'

# Configure deep learning and slide backends
from ._backend import backend, slide_backend

# Import logging functions required for other submodules
from slideflow.util import getLoggingLevel, log, setLoggingLevel, about

...
from slideflow import io, model, norm, stats, gan
from slideflow.dataset import Dataset
from slideflow.heatmap import Heatmap
from slideflow.model import DatasetFeatures, ModelParams
from slideflow.model import (
    list_extractors, list_torch_extractors, list_tensorflow_extractors,
    is_extractor, is_torch_extractor, is_tensorflow_extractor,
    build_feature_extractor, build_torch_feature_extractor,
    build_tensorflow_feature_extractor, rebuild_extractor
)
from slideflow.mosaic import Mosaic
from slideflow.project import Project
from slideflow.project import create as create_project
from slideflow.project import load as load_project
from slideflow.slide import WSI
from slideflow.stats import SlideMap
from slideflow.tfrecord import TFRecord, tfrecord_loader, multi_tfrecord_loader
from slideflow.plugin import load_plugins
