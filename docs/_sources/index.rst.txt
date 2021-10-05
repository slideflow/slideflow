.. slideflow documentation master file, created by
   sphinx-quickstart on Mon Jan 14 19:42:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Slideflow Documentation
=======================

``slideflow`` is a computational pathology Python package which aims to provide an easy and intuitive way of building and testing deep learning models in histology image analysis. It is built with a `Tensorflow/Keras <https://www.tensorflow.org/>`_ backend, but also includes an optimized `PyTorch <https://pytorch.org>`_ dataloader which can read and serve images stored in cross-compatible TFRecord format (full PyTorch backend support in development), as well as `CLAM <https://github.com/mahmoodlab/CLAM>`_. The overarching goal of the package is to provide tools to efficiently process histology slides in tile-wise fashion suitable for deep learning models; enable easy model training and testing from extracted tiles; and supply functional tools to assist with model analysis and interpretability, including predictive heatmaps, mosaic maps, and more.

The ``slideflow`` package includes a ``Project`` class to help coordinate project organization and supervise execution of the pipeline.  This documentation starts with a high-level overview of the pipeline, and will include examples of how to execute functions using the ``Project`` class. We will end by describing some of the source code components in more detail and provide examples for how you may further customize this pipeline according to your needs.

The current implementation has been developed and tested in Python 3.7/3.8, using Tensorflow 2.5/2.6 and PyTorch 1.19, in Ubuntu 20.04, CentOS 7, and CentOS 8.

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
   torch
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
