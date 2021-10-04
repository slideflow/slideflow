.. currentmodule:: slideflow.io.tensorflow

slideflow.io.tensorflow
=======================

This module contains functions for processing TFRecords, including detecting contents and image format of saved
TFRecords, extracting tiles from TFRecords, splitting and merging TFRecrds, and a variety of other manipulations.

The more important compontent of this module, however, is the :func:`slideflow.io.tensorflow.interleave` function,
which interleaves a set of tfrecords together into a :class:`tf.data.Datasets` object that can be used for training.
This interleaving can include patient or category-level balancing for returned batches (see :ref:`balancing`).

.. note::
    The TFRecord reading and interleaving implemented in this module is only compatible with Tensorflow models.
    The :mod:`slideflow.io.torch` module includes an optimized, PyTorch-specific TFRecord reader based on a modified
    version of the tfrecord reader/writer at: https://github.com/vahidk/tfrecord.

.. automodule:: slideflow.io.tensorflow
    :members: