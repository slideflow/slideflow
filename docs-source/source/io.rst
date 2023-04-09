.. currentmodule:: slideflow.io

slideflow.io
============

This module contains utility functions for working with TFRecords, cross-compatible
with both Tensorflow and PyTorch.

Functions included in this module assist with processing TFRecords, detecting image and data format,
extracting tiles, splitting and merging TFrecords, and a variety of other manipulations.

Additional Tensorflow-specific TFRecord reading/writing utility functions are
available in :py:mod:`slideflow.io.tensorflow`, and additional PyTorch-specific
functions are in :py:mod:`slideflow.io.torch`.

.. autofunction:: convert_dtype
.. autofunction:: detect_tfrecord_format
.. autofunction:: extract_tiles
.. autofunction:: get_locations_from_tfrecord
.. autofunction:: get_tfrecord_by_index
.. autofunction:: get_tfrecord_by_location
.. autofunction:: get_tfrecord_parser
.. autofunction:: get_tfrecord_length
.. autofunction:: read_and_return_record
.. autofunction:: serialized_record
.. autofunction:: tfrecord_has_locations
.. autofunction:: update_manifest_at_dir
.. autofunction:: write_tfrecords_multi
.. autofunction:: write_tfrecords_single
.. autofunction:: write_tfrecords_merge

slideflow.io.preservedsite
**************************
.. autofunction:: slideflow.io.preservedsite.generate_crossfolds