.. currentmodule:: slideflow.activations

slideflow.activations
=====================

This module provides several classes for calculating and visualizing both logits and intermediate
layer activations from tiles, slides, and entire datasets.

:class:`slideflow.model.Features` creates an interface to efficiently generate features/layer activations and logits
from either a batch of images (returning a batch of activations/logits) or a whole-slide image (returning a grid of
activations/logits).

:class:`slideflow.model.DatasetFeatures` calculates features and logits for an entire dataset, storing
result arrays into a dictionary mapping slide names to the generated activations. This buffer of whole-dataset
activations can then be used for functions requiring analysis of whole-dataset activations, including
:class:`slideflow.statistics.SlideMap` and :class:`slideflow.mosiac.Mosaic`.

:class:`slideflow.heatmap.Heatmap` uses a model to generate predictions across a whole-slide image through
progressive convolution. These prediction heatmaps can be interactively displayed or saved for later use.

.. automodule: slideflow.activations

Features
--------

.. autoclass:: Features
    :inherited-members:

DatasetFeatures
---------------
.. autoclass:: DatasetFeatures
    :inherited-members:

Heatmap
-------
.. autoclass:: Heatmap
    :inherited-members: