.. currentmodule:: slideflow.cellseg

slideflow.cellseg
=================

This module contains utility functions for performing whole-slide image cell segmentation with Cellpose.

See :ref:`cellseg` for more information.

.. autofunction:: segment_slide

Segmentation
************
.. autoclass:: Segmentation
.. autofunction:: slideflow.cellseg.Segmentation.apply_rois
.. autofunction:: slideflow.cellseg.Segmentation.calculate_centroids
.. autofunction:: slideflow.cellseg.Segmentation.calculate_outlines
.. autofunction:: slideflow.cellseg.Segmentation.centroids
.. autofunction:: slideflow.cellseg.Segmentation.centroid_to_image
.. autofunction:: slideflow.cellseg.Segmentation.extract_centroids
.. autofunction:: slideflow.cellseg.Segmentation.mask_to_image
.. autofunction:: slideflow.cellseg.Segmentation.outline_to_image
.. autofunction:: slideflow.cellseg.Segmentation.save