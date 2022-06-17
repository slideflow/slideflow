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
coordinates, you can use the ``from_precalculated`` class method. In addition to X and Y, this method requires supplying
tile-level metadata in the form of a list of dicts. Each dict must contain the name of the origin slide and the tile
index in the slide TFRecord.

.. code-block:: python

    dataset = project.dataset(tile_px=299, tile_um=302)
    slides = dataset.slides()
    x = np.array(...)
    y = np.array(...)
    meta = [{'slide': ..., 'index': ...} for i in range(len(x))]
    slide_map = sf.SlideMap.from_precalculated(slides, x, y, meta)

.. automodule: slideflow.stats
    :imported-members:

SlideMap
--------

.. autoclass:: slideflow.SlideMap
    :inherited-members:

basic_metrics
----------------------
.. autofunction:: basic_metrics

calculate_centroid
------------------
.. autofunction:: calculate_centroid

concordance_index
----------------------
.. autofunction:: concordance_index

filtered_prediction
-------------------
.. autofunction:: filtered_prediction

generate_combined_roc
----------------------
.. autofunction:: generate_combined_roc

generate_roc
----------------------
.. autofunction:: generate_roc

generate_scatter
----------------------
.. autofunction:: generate_scatter

gen_umap
--------
.. autofunction:: gen_umap

get_centroid_index
------------------
.. autofunction:: get_centroid_index

metrics_from_dataset
---------------------
.. autofunction:: metrics_from_dataset

metrics_from_pred
-----------------
.. autofunction:: metrics_from_pred

normalize_layout
----------------
.. autofunction:: normalize_layout

read_predictions
----------------
.. autofunction:: read_predictions

predict_from_tensorflow
------------------------
.. autofunction:: predict_from_tensorflow

predict_from_torch
----------------------
.. autofunction:: predict_from_torch

save_histogram
----------------------
.. autofunction:: save_histogram

pred_to_df
-------------------------
.. autofunction:: pred_to_df

to_onehot
----------------------
.. autofunction:: to_onehot














