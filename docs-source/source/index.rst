Introduction
============

``slideflow`` is a computational pathology Python package which aims to provide an easy and intuitive way of building and testing deep learning models for use in histology image analysis. It is built using Keras (with `Tensorflow <https://www.tensorflow.org/>`_ backend) and supports all Keras architectures, as well as `CLAM <https://github.com/mahmoodlab/CLAM>`_. The overarching goal of the package is to provide tools to train and test models on histology slides, apply these models to new slides, and assess performance using analytical tools including predictive heatmaps, mosaic maps, ROCs, and more.

The ``slideflow`` package includes a ``Project`` class to help coordinate project organization and supervise execution of the pipeline.  This documentation starts with a high-level overview of the pipeline, and will include examples of how to execute pipeline functions using the ``Project`` class. We will end by describing some of the source code components in more detail and provide examples for how you may further customize this pipeline according to your needs.

The current implementation has been developed in Python 3.7 and 3.8, using Tensorflow 2.5, in both Ubuntu 20.04 and CentOS 7.