.. slideflow documentation master file

.. figure:: https://i.imgur.com/YrsKN4I.jpeg



Slideflow Documentation
=======================

Slideflow is a Python package that provides a unified API for building and testing deep learning models for histopathology, supporting both Tensorflow/Keras and PyTorch.

Slideflow includes tools for efficient whole-slide image processing, easy and highly customizable model training with uncertainty quantification (UQ), and a number of functional tools to assist with analysis and interpretability, including predictive heatmaps, mosaic maps, GANs, saliency maps, and more. It is built with both `Tensorflow/Keras <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org>`_ backends, with fully cross-compatible TFRecord data storage.

This documentation starts with a high-level overview of the pipeline and includes examples of how to perform common tasks using the ``Project`` helper class. We also provide several tutorials with examples of how Slideflow can be used and extended for additional functionality.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   installation
   overview
   quickstart
   project_setup
   datasets_and_val
   slide_processing
   training
   evaluation
   posthoc
   uq
   features
   mil
   ssl
   stylegan
   saliency
   segmentation
   cellseg
   custom_loops
   studio
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Developer Notes

   tfrecords
   dataloaders
   custom_extractors
   tile_labels
   plugins

.. toctree::
   :maxdepth: 1
   :caption: API

   slideflow
   project
   dataset
   dataset_features
   heatmap
   model_params
   mosaic
   slidemap
   biscuit
   slideflow_cellseg
   io
   io_tensorflow
   io_torch
   gan
   grad
   mil_module
   model
   model_tensorflow
   model_torch
   norm
   simclr
   slide
   slide_qc
   stats
   util
   studio_module

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial1
   tutorial2
   tutorial3
   tutorial4
   tutorial5
   tutorial6
   tutorial7
   tutorial8