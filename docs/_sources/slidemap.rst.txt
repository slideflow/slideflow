.. currentmodule:: slideflow

slideflow.SlideMap
==================

:class:`slideflow.SlideMap` assists with visualizing tiles and slides in two-dimensional space.

Once a model has been trained, tile-level predictions and intermediate layer activations can be calculated
across an entire dataset with :class:`slideflow.DatasetFeatures`.
The :class:`slideflow.SlideMap` class can then perform dimensionality reduction on these dataset-wide
activations, plotting tiles and slides in two-dimensional space. Visualizing the distribution and clustering
of tile-level and slide-level layer activations can help reveal underlying structures in the dataset and shared
visual features among classes.

The primary method of use is first generating an :class:`slideflow.DatasetFeatures` from a trained
model, then using :meth:`slideflow.DatasetFeatures.map_activations`, which returns an instance of
:class:`slideflow.SlideMap`.

.. code-block:: python

    ftrs = sf.DatasetFeatures(model='/path/', ...)
    slide_map = ftrs.map_activations()

Alternatively, if you would like to map slides from a dataset in two-dimensional space using pre-calculated *x* and *y*
coordinates, you can use the :meth:`sldieflow.SlideMap.from_xy` class method. In addition to X and Y, this method
requires supplying tile-level metadata in the form of a list of dicts. Each dict must contain the name of the origin
slide and the tile index in the slide TFRecord.

.. code-block:: python

    x = np.array(...)
    y = np.array(...)
    slides = ['slide1', 'slide1', 'slide5', ...]
    slide_map = sf.SlideMap.from_xy(x=x, y=y, slides=slides)

.. autoclass:: SlideMap

Methods
-------

.. autofunction:: slideflow.SlideMap.activations
.. autofunction:: slideflow.SlideMap.build_mosaic
.. autofunction:: slideflow.SlideMap.cluster
.. autofunction:: slideflow.SlideMap.neighbors
.. autofunction:: slideflow.SlideMap.filter
.. autofunction:: slideflow.SlideMap.umap_transform
.. autofunction:: slideflow.SlideMap.label
.. autofunction:: slideflow.SlideMap.label_by_preds
.. autofunction:: slideflow.SlideMap.label_by_slide
.. autofunction:: slideflow.SlideMap.label_by_uncertainty
.. autofunction:: slideflow.SlideMap.load
.. autofunction:: slideflow.SlideMap.load_coordinates
.. autofunction:: slideflow.SlideMap.load_umap
.. autofunction:: slideflow.SlideMap.plot
.. autofunction:: slideflow.SlideMap.plot_3d
.. autofunction:: slideflow.SlideMap.save
.. autofunction:: slideflow.SlideMap.save_3d
.. autofunction:: slideflow.SlideMap.save_plot
.. autofunction:: slideflow.SlideMap.save_coordinates
.. autofunction:: slideflow.SlideMap.save_umap
.. autofunction:: slideflow.SlideMap.save_encoder
