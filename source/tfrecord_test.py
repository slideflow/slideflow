#from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as numpy

def _float_feature(value):
    """Returns a bytes_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_image_tile(category_label, case_label, float_image):
    """ Creates a tf.Example message from a single image tile, ready to be written to a file."""

    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
        'category': _int64_feature(category_label),
        'case':     _bytes_feature(case_label),
        'image':    _float_feature(float_image)
    }

    # Create a Features message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

example_observation = []
#image = [[[2, 3, 4], [5, 4, 2]], 
#         [[2, 3, 4], [5, 4, 2]]]
serialized_example = serialize_image_tile(0, b'234758', 4)
print(serialized_example)