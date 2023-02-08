'''This module contains low-level utilies for interfacing with TFRecords.

Code from this module was inspired in part by https://github.com/vahidk/tfrecord.
'''

from slideflow.tfrecord import \
    iterator_utils  # noqa # pylint: disable=unused-import
from slideflow.tfrecord import reader  # noqa # pylint: disable=unused-import
from slideflow.tfrecord import tools  # noqa # pylint: disable=unused-import
from slideflow.tfrecord import torch  # noqa # pylint: disable=unused-import
from slideflow.tfrecord import writer  # noqa # pylint: disable=unused-import
from slideflow.tfrecord.iterator_utils import *  # noqa # pylint: disable=unused-import
from slideflow.tfrecord.reader import *  # noqa # pylint: disable=unused-import
from slideflow.tfrecord.writer import *  # noqa # pylint: disable=unused-import
from slideflow.util import example_pb2  # noqa # pylint: disable=unused-import
