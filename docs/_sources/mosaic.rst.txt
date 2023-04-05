.. currentmodule:: slideflow

.. _mosaic:

slideflow.Mosaic
================

:class:`slideflow.Mosaic` plots tile images onto a map of slides, generating a mosaic map.

The idea of a mosaic map is to visualize image feature variation across slides and among categories, in an attempt
to better understand the kinds of image features discriminative models might be using to generate class predictions.
They are created by first generating whole-dataset layer features (using
:class:`slideflow.DatasetFeatures`), which are then mapped into two-dimensional space using UMAP
dimensionality reduction (:class:`slideflow.SlideMap`). This resulting SlideMap is then passed to
:class:`slideflow.Mosaic`, which overlays tile images onto the dimensionality-reduced slide map.

An example of a mosaic map can be found in Figure 4 of `this paper <https://doi.org/10.1038/s41379-020-00724-3>`_.
It bears some resemblence to the Activation Atlases created by
`Google and OpenAI <https://distill.pub/2019/activation-atlas/>`_, without the use of feature inversion.

See :ref:`mosaic_map` for an example of how a mosaic map can be used in the context of a project.

.. autoclass:: Mosaic

Methods
-------

.. autofunction:: slideflow.Mosaic.generate_grid
.. autofunction:: slideflow.Mosaic.plot
.. autofunction:: slideflow.Mosaic.points_at_grid_index
.. autofunction:: slideflow.Mosaic.save
.. autofunction:: slideflow.Mosaic.save_report
.. autofunction:: slideflow.Mosaic.view