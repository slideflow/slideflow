.. currentmodule:: slideflow.io.tfrecords

slideflow.io.tfrecords
=======================

This module contains functions for processing TFRecords, including detecting contents and image format of saved
TFRecords, extracting tiles from TFRecords, splitting and merging TFRecrds, and a variety of other manipulations.

The more important compontent of this module, however, is the :func:`slideflow.io.tfrecords.interleave` function,
which interleaves a set of tfrecords together into a :class:`tf.data.Datasets` object that can be used for training.
This interleaving can include patient or category-level balancing for returned batches (see :ref:`balancing`).

.. note::
    The TFRecord reading and interleaving implemented in this module is only compatible with Tensorflow models.
    The :mod:`slideflow.io.reader` module includes a backend-agnostic TFRecord reader based on
    `dareblopy <https://github.com/podgorskiy/DareBlopy>`_ which can be used as input for Torch models.

.. automodule:: slideflow.io.tfrecords
    :members: