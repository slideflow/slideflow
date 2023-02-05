.. currentmodule:: slideflow.stats

slideflow.stats
===============

In addition to containing functions used during model training and evaluation, this module provides
the :class:`slideflow.SlideMap` class designed to assist with visualizing tiles and slides
in two-dimensional space.

Once a model has been trained, tile-level predictions and intermediate layer activations can be calculated
across an entire dataset with :class:`slideflow.DatasetFeatures`.
The :class:`slideflow.SlideMap` class can then perform dimensionality reduction on these dataset-wide
activations, plotting tiles and slides in two-dimensional space. Visualizing the distribution and clustering
of tile-level and slide-level layer activations can help reveal underlying structures in the dataset and shared
visual features among classes.

The primary method of use is first generating an :class:`slideflow.DatasetFeatures` from a trained
model, then creating an instance of a :class:`slideflow.SlideMap` by using the ``from_features`` class
method:

.. code-block:: python

    df = sf.DatasetFeatures(model='/path/', ...)
    slide_map = sf.SlideMap.from_features(df)

Alternatively, if you would like to map slides from a dataset in two-dimensional space using pre-calculated *x* and *y*
coordinates, you can use the ``from_xy`` class method. In addition to X and Y, this method requires supplying
tile-level metadata in the form of a list of dicts. Each dict must contain the name of the origin slide and the tile
index in the slide TFRecord.

.. code-block:: python

    x = np.array(...)
    y = np.array(...)
    slides = ['slide1', 'slide1', 'slide5', ...]
    slide_map = sf.SlideMap.from_xy(x=x, y=y, slides=slides)

.. automodule: slideflow.stats
    :imported-members:

SlideMap
--------

.. autoclass:: slideflow.SlideMap
    :inherited-members:

Other functions
---------------
.. autofunction:: df_from_pred

.. autofunction:: eval_dataset

.. autofunction:: group_reduce

.. autofunction:: metrics_from_dataset

.. autofunction:: name_columns

.. autofunction:: predict_dataset

.. autofunction:: calculate_centroid

.. autofunction:: get_centroid_index