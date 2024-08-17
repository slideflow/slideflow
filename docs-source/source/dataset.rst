.. currentmodule:: slideflow

.. _dataset:

slideflow.Dataset
=================

.. autoclass:: Dataset

Attributes
----------

.. autosummary::

    Dataset.annotations

    Dataset.filters
    Dataset.filter_blank
    Dataset.filtered_annotations
    Dataset.img_format
    Dataset.min_tiles
    Dataset.num_tiles

Methods
-------

.. autofunction:: slideflow.Dataset.balance
.. autofunction:: slideflow.Dataset.build_index
.. autofunction:: slideflow.Dataset.cell_segmentation
.. autofunction:: slideflow.Dataset.check_duplicates
.. autofunction:: slideflow.Dataset.clear_filters
.. autofunction:: slideflow.Dataset.clip
.. autofunction:: slideflow.Dataset.convert_xml_rois
.. autofunction:: slideflow.Dataset.extract_cells
.. autofunction:: slideflow.Dataset.extract_tiles
.. autofunction:: slideflow.Dataset.extract_tiles_from_tfrecords
.. autofunction:: slideflow.Dataset.filter
.. autofunction:: slideflow.Dataset.find_slide
.. autofunction:: slideflow.Dataset.find_tfrecord
.. autofunction:: slideflow.Dataset.generate_feature_bags
.. autofunction:: slideflow.Dataset.get_tfrecord_locations
.. autofunction:: slideflow.Dataset.get_tile_dataframe
.. autofunction:: slideflow.Dataset.harmonize_labels
.. autofunction:: slideflow.Dataset.is_float
.. autofunction:: slideflow.Dataset.kfold_split
.. autofunction:: slideflow.Dataset.labels
.. autofunction:: slideflow.Dataset.load_annotations
.. autofunction:: slideflow.Dataset.load_indices
.. autofunction:: slideflow.Dataset.manifest
.. autofunction:: slideflow.Dataset.manifest_histogram
.. autofunction:: slideflow.Dataset.patients
.. autofunction:: slideflow.Dataset.get_bags
.. autofunction:: slideflow.Dataset.read_tfrecord_by_location
.. autofunction:: slideflow.Dataset.remove_filter
.. autofunction:: slideflow.Dataset.rebuild_index
.. autofunction:: slideflow.Dataset.resize_tfrecords
.. autofunction:: slideflow.Dataset.rois
.. autofunction:: slideflow.Dataset.slide_manifest
.. autofunction:: slideflow.Dataset.slide_paths
.. autofunction:: slideflow.Dataset.slides
.. autofunction:: slideflow.Dataset.split
.. autofunction:: slideflow.Dataset.split_tfrecords_by_roi
.. autofunction:: slideflow.Dataset.summary
.. autofunction:: slideflow.Dataset.tensorflow
.. autofunction:: slideflow.Dataset.tfrecord_report
.. autofunction:: slideflow.Dataset.tfrecord_heatmap
.. autofunction:: slideflow.Dataset.tfrecords
.. autofunction:: slideflow.Dataset.tfrecords_by_subfolder
.. autofunction:: slideflow.Dataset.tfrecords_folders
.. autofunction:: slideflow.Dataset.tfrecords_from_tiles
.. autofunction:: slideflow.Dataset.tfrecords_have_locations
.. autofunction:: slideflow.Dataset.transform_tfrecords
.. autofunction:: slideflow.Dataset.thumbnails
.. autofunction:: slideflow.Dataset.torch
.. autofunction:: slideflow.Dataset.unclip
.. autofunction:: slideflow.Dataset.update_manifest
.. autofunction:: slideflow.Dataset.update_annotations_with_slidenames
.. autofunction:: slideflow.Dataset.verify_annotations_slides
.. autofunction:: slideflow.Dataset.verify_img_format
.. autofunction:: slideflow.Dataset.verify_slide_names
