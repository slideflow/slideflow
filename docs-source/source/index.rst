.. slideflow documentation master file, created by
   sphinx-quickstart on Mon Jan 14 19:42:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Slideflow Documentation
=======================

``slideflow`` is a computational pathology Python package which aims to provide an easy and intuitive way of building and testing deep learning models in histology image analysis. It is built to support both `Tensorflow/Keras <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org>`_ backends (full PyTorch support coming soon), with support for all Tensorflow/Keras architectures, as well as `CLAM <https://github.com/mahmoodlab/CLAM>`_. The overarching goal of the package is to provide tools to train and test models on histology slides, apply these models to new slides, and assess performance using analytical tools including predictive heatmaps, mosaic maps, ROCs, and more.

The ``slideflow`` package includes a ``Project`` class to help coordinate project organization and supervise execution of the pipeline.  This documentation starts with a high-level overview of the pipeline, and will include examples of how to execute pipeline functions using the ``Project`` class. We will end by describing some of the source code components in more detail and provide examples for how you may further customize this pipeline according to your needs.

The current implementation has been developed in Python 3.7 and 3.8, using Tensorflow 2.5, in Ubuntu 20.04, CentOS 7, and CentOS 8.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   pipeline
   project_setup
   validation
   extract_tiles
   training
   evaluation
   layer_activations
   clam
   troubleshooting
   appendix

.. toctree::
   :maxdepth: 1
   :caption: Source

   project
   activations
   dataset
   tfrecords
   reader
   model
   mosaic
   slide
   statistics
   util

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial1

Indices and tables
==================

* :ref:`genindex`
