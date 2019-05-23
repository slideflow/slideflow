Introduction
============

``histcon`` is a Python package which aims to provide an easy and intuitive way of building and testing convolutional neural networks (CNNs) for use in histology image analysis. It is built using ``Tensorflow`` (see `here <https://www.tensorflow.org/>`_) and currently utilizes Google's `Inception-v4 <https://github.com/tensorflow/models/tree/master/research/slim>`_ network architecture.

The ``histcon`` object will initialize data input streams, build an Inception-V4 model, and initiate training upon calling ``histcon.train``. Re-training (transfer learning) is available through ``histcon.retrain``. After a model has been trained, whole-slide-image predictions can be generated and visualized with heatmap overlays using ``convoluter``. Other bundled objects, including ``data_utils`` and ``nconvert_util``, provide easy-to-use tools for generating datasets from annotated whole-slide images.

The current implementation has been developed in Python 3 using Tensorflow 1.12 and tested in Ubuntu 18.04, but should work on most Debian-based OSs.

In the following section, you will be guided through the workflow and data pipeline and provided with examples.