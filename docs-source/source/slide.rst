.. currentmodule:: slideflow.slide

slideflow.slide
===============

This module contains classes to load slides and extract tiles. For optimal performance, tile extraction should
generally not be performed by instancing these classes directly, but by calling either
:func:`slideflow.Project.extract_tiles` or :func:`slideflow.Dataset.extract_tiles`, which include performance
optimizations and additional functionality.

slideflow.WSI
*************

.. autoclass:: WSI

Attributes
----------

.. autosummary::

    WSI.dimensions
    WSI.qc_mask
    WSI.levels
    WSI.level_dimensions
    WSI.level_downsamples
    WSI.level_mpp
    WSI.properties
    WSI.slide
    WSI.vendor

Methods
-------

.. autofunction:: slideflow.WSI.apply_qc_mask
.. autofunction:: slideflow.WSI.apply_segmentation
.. autofunction:: slideflow.WSI.build_generator
.. autofunction:: slideflow.WSI.dim_to_mpp
.. autofunction:: slideflow.WSI.get_tile_mask
.. autofunction:: slideflow.WSI.extract_cells
.. autofunction:: slideflow.WSI.extract_tiles
.. autofunction:: slideflow.WSI.load_csv_roi
.. autofunction:: slideflow.WSI.load_json_roi
.. autofunction:: slideflow.WSI.mpp_to_dim
.. autofunction:: slideflow.WSI.predict
.. autofunction:: slideflow.WSI.preview
.. autofunction:: slideflow.WSI.qc
.. autofunction:: slideflow.WSI.remove_qc
.. autofunction:: slideflow.WSI.tensorflow
.. autofunction:: slideflow.WSI.torch
.. autofunction:: slideflow.WSI.thumb
.. autofunction:: slideflow.WSI.view


slideflow.TMA
*************

.. autoclass:: TMA

Attributes
----------

.. autosummary::

    TMA.dimensions
    TMA.qc_mask
    TMA.levels
    TMA.level_dimensions
    TMA.level_downsamples
    TMA.level_mpp
    TMA.properties
    TMA.slide
    TMA.vendor

Methods
-------

.. autofunction:: slideflow.TMA.apply_qc_mask
.. autofunction:: slideflow.TMA.build_generator
.. autofunction:: slideflow.TMA.dim_to_mpp
.. autofunction:: slideflow.TMA.extract_tiles
.. autofunction:: slideflow.TMA.mpp_to_dim
.. autofunction:: slideflow.TMA.preview
.. autofunction:: slideflow.TMA.qc
.. autofunction:: slideflow.TMA.remove_qc
.. autofunction:: slideflow.TMA.thumb

Other functions
***************

.. autofunction:: slideflow.slide.predict