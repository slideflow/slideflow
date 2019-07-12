Introduction
============

``slideflow`` is a Python package which aims to provide an easy and intuitive way of building and testing convolutional neural networks (CNNs) for use in histology image analysis. It is built using ``Keras`` (with ``Tensorflow`` backend, see `here <https://www.tensorflow.org/>`_) and supports many standard network architectures - including InceptionV3, Resnet, and VGG16 - as well as custom architectures. The overarching goal of the package is to provide tools to train and test models on histology slides, apply these models to new slides, and analyze performance by generating predictive heatmaps, ROCs, and mosaic maps. 

The ``slideflow`` package includes a ``SlideFlowProject`` class to help coordinate project organization and supervise execution of the pipeline.  In this documentation, we will review a high-level overview of the pipeline and examine how to execute each component of the pipeline using the ``SlideFlowProject`` class. We will end by describing some of the source code components in more detail and provide examples for how you may further customize this pipeline according to your needs.

The current implementation has been developed in Python 3.7 using Tensorflow 2.0 (beta), and tested in Ubuntu 18.04, Windows 10, and CentOS 7.