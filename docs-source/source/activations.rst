.. currentmodule:: slideflow.activations

slideflow.activations
=====================

This module provides several classes for calculating and visualizing both logits and intermediate
layer activations from tiles, slides, and entire datasets.

:class:`slideflow.activations.ActivationsInterface` creates an interface to efficiently generate activations/logits from
either a batch of images (returning a batch of activations/logits) or a whole-slide image (returning a grid of
activations/logits). The use of a ``@tf.function`` backend allows for performant results even when looping through
datasets in eager execution.

:class:`slideflow.activations.ActivationsVisualizer` calculates activations/logits for an entire dataset, storing
result arrays into a dictionary mapping slide names to the generated activations. This buffer of whole-dataset
activations can then be used for functions requiring analysis of whole-dataset activations, including
:class:`slideflow.statistics.SlideMap` and :class:`slideflow.mosiac.Mosaic`.

:class:`slideflow.activations.Heatmap` uses a model to generate predictions across a whole-slide image through
progressive convolution. These prediction heatmaps can be interactively displayed or saved for later use.

.. automodule: slideflow.activations

ActivationsInterface
--------------------

.. autoclass:: ActivationsInterface
    :inherited-members:

ActivationsVisualizer
---------------------
.. autoclass:: ActivationsVisualizer
    :inherited-members:

Heatmap
-------
.. autoclass:: Heatmap
    :inherited-members: