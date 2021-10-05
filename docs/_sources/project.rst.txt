.. currentmodule:: slideflow.project

slideflow.Project
=================

This class provides a high-level interface that simplifies execution of pipeline functions. Nearly all pipeline tasks
can be accomplished with the methods in this class, although directly interacting with the various objects in this
package will enable more granular control.

.. autoclass:: Project
    :members: __init__, from_prompt, select_gpu, add_source, associate_slide_names, create_blank_annotations, create_blank_train_config, create_hyperparameter_sweep, evaluate, evaluate_clam, extract_tiles, extract_tiles_from_tfrecords, generate_activations, generate_features_for_clam, generate_heatmaps, generate_mosaic, generate_mosaic_from_annotations, generate_thumbnails, generate_tfrecord_heatmap, get_dataset, predict_wsi, resize_tfrecords, save, slide_report, tfrecord_report, train, train_clam
