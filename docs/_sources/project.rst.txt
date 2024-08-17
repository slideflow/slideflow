.. currentmodule:: slideflow

.. _project:

slideflow.Project
=================

.. autoclass:: Project

Attributes
----------

.. autosummary::

    Project.annotations
    Project.dataset_config
    Project.eval_dir
    Project.models_dir
    Project.name
    Project.neptune_api
    Project.neptune_workspace
    Project.sources

Methods
-------

.. autofunction:: slideflow.Project.add_source

.. autofunction:: slideflow.Project.associate_slide_names

.. autofunction:: slideflow.Project.cell_segmentation

.. autofunction:: slideflow.Project.create_blank_annotations

.. autofunction:: slideflow.Project.create_hp_sweep

.. autofunction:: slideflow.Project.evaluate

.. autofunction:: slideflow.Project.evaluate_mil

.. autofunction:: slideflow.Project.extract_cells

.. autofunction:: slideflow.Project.extract_tiles

.. autofunction:: slideflow.Project.gan_train

.. autofunction:: slideflow.Project.gan_generate

.. autofunction:: slideflow.Project.generate_features

.. autofunction:: slideflow.Project.generate_feature_bags

.. autofunction:: slideflow.Project.generate_heatmaps

.. autofunction:: slideflow.Project.generate_mosaic

.. autofunction:: slideflow.Project.generate_mosaic_from_annotations

.. autofunction:: slideflow.Project.generate_tfrecord_heatmap

.. autofunction:: slideflow.Project.dataset

.. autofunction:: slideflow.Project.predict

.. autofunction:: slideflow.Project.predict_ensemble

.. autofunction:: slideflow.Project.predict_wsi

.. autofunction:: slideflow.Project.save

.. autofunction:: slideflow.Project.smac_search

.. autofunction:: slideflow.Project.train

.. autofunction:: slideflow.Project.train_ensemble

.. autofunction:: slideflow.Project.train_mil

.. autofunction:: slideflow.Project.train_simclr
