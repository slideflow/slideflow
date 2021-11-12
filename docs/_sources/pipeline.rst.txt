Pipeline Overview
=================

The overall pipeline is separated into two phases and 6 steps.

The first phase - **Model Creation** - involves three steps: 1) labeling slides with regions of interest (ROIs), 2) segmenting and preparing image tiles from the slides, and 3) training a model.

The second phase - **Model Assessment** - also involves three steps: 1) analytics describing model performance, including basic measures like percent accuracy, ROCs, and scatter plots, 2) creating heatmap overlays for whole-slide images to visualize predictions, and 3) generating mosaic maps to visualize learned image features.

A high-level overview of each of the six steps is provided below. We will examine execution of these steps in more detail in the following sections.

Step 1: Create ROIs
*******************

1) **Label ROIs** (optional). Using `QuPath <https://qupath.github.io/>`_, annotate whole-slide images with the Polygon tool. Then, click **Automate** -> **Show script editor**. In the box that comes up, click **File** -> **Open** and load the ``qupath_roi.groovy`` script. Press CTRL + R and wait for the script to finish. Alternatively, you can load multiple SVS files into a QuPath project and run the script on the entire project using "Run for project".

.. note::
    This step may be skipped if you are performing analysis on whole-slide images, rather than annotated tumor regions.

Step 2: Data Preparation
************************

2) **Extract tiles**. Once ROIs have been created, tiles will need to be extracted from the ROIs across all of your slides. Tiles will be extracted at a given magnification size in microns, and saved at a given resolution in pixels. The optimal extraction size in both microns and pixels will depend on your dataset and model architecture. Poor quality tiles - including background tiles or tiles with high whitespace content - will be automatically discarded. Tiles will be stored as TFRecords, a binary file format used to improve dataset reading performance during training. Each slide will have its own TFRecord file containing its extracted tiles.

3) **Set aside final evaluation set**. Using the project annotations CSV file, designate which slides should be saved for final evaluation.

4) **Establish training and validation dataset**. By default, three-fold cross-validation will be performed during training. Many other validation strategies are also supported (:ref:`validation_planning`).

Step 3: Model Training
**********************

5) **Choose hyperparameters**. Before training can begin, you must choose both a model architecture (e.g. InceptionV3, VGG16, ResNet, etc.) and a set of hyperparameters (e.g. batch size, learning rate, etc.). This can be done explicitly one at a time, or an automatic hyperparameter sweep can be configured.

6) **Initiate training**. Train your model across all desired hyperparameters and select the best-performing hyperparameter combination for final evaluation testing.

Step 4: Analytics
*****************
Validation testing is performed both during training - at specified epochs - and after training has completed. Various metrics are recorded in the project directory at these intervals to assist with model performance assessment, including:

- **Training and validation loss**
- **Training and validation accuracy** (for categorical outcomes)
- **Tile-level, slide-level, and patient-level AUROC and AP** (for categorical outcomes)
- **Tile-level, slide-level, and patient-level scatter plots with R-squared** (for continuous outcomes)
- **Tile-level, slide-level, and patient-level C-index** (for Cox Proportional Hazards models)
- **Histograms of predictions** (for continuous outcomes)

Step 5: Visualizing Results with Heatmaps
*****************************************
In addition to the above metrics, performance of a trained model can be assessed by visualizing predictions for a set slides as heatmaps.

.. image:: heatmap_example.png

Step 6: Mosaic maps
*******************
Finally, learned image features can be visualized using dimensionality reduction on model layer activations. A set of image tiles is first provided to your trained model, which calculates activations at a specified intermediate layer. Tile-level activations are then plotted with dimensionality reduction (UMAP), and points on the plot are replaced with image tiles, generating a mosaic map.

.. image:: mosaic_example.png
