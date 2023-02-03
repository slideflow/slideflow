.. slideflow documentation master file

.. figure:: https://i.imgur.com/YrsKN4I.jpeg



Slideflow Documentation
=======================

``slideflow`` is a Python package that provides a unified API for building and testing deep learning models for histopathology, supporting both Tensorflow/Keras and PyTorch.

Slideflow includes tools for efficient whole-slide image processing, easy and highly customizable model training with uncertainty quantification (UQ), and a number of functional tools to assist with analysis and interpretability, including predictive heatmaps, mosaic maps, GANs, saliency maps, and more. It is built with both `Tensorflow/Keras <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org>`_ backends, with fully cross-compatible TFRecord data storage.

The ``slideflow`` package includes a ``Project`` class to help coordinate project organization and supervise execution of the pipeline.  This documentation starts with a high-level overview of the pipeline, and will include examples of how to perform common tasks using the ``Project`` class. We also provide several tutorials with examples of how Slideflow can be used on your own data.

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   installation
   overview
   quickstart
   project_setup
   validation
   extract_tiles
   training
   evaluation
   layer_activations
   workbench_tools
   custom_loops
   uq
   stylegan
   clam
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: API

   project
   dataset
   heatmap
   io
   io_tensorflow
   io_torch
   gan
   grad
   model
   mosaic
   norm
   slide
   slide_qc
   stats
   util
   workbench

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial1
   tutorial2
   tutorial3
   tutorial4
   tutorial5
   tutorial6