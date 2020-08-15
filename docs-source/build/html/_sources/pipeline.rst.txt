Pipeline Overview
=================

.. image:: pipeline_overview.png

The overall pipeline is separated into two phases and 6 steps. 

The first phase - **Model Creation** - involves three steps: 1) labeling slides with regions of interest (ROIs), 2) tessellating and preparing image tiles from the slides, and 3) training a neural network model. 

The second phase - **Model Assessment** - also involves three steps: 1) analytics describing model performance, including basic measures like percent accuracy as well as ROCs and scatter plots, 2) creating heatmap overlays for whole-slide images to visualize predictions, and 3) generating mosaic maps to visualize learned image features.

A high-level overview of each of the six steps is provided below. We will examine execution of these steps in more detail in the following sections.

Step 1: Create ROIs
*******************

1) **Label ROIs**. Using `QuPath <https://qupath.github.io/>`_, annotate whole-slide images with the Polygon tool. Then, click **Automate** -> **Show script editor**. In the box that comes up, click **File** -> **Open** and load the ``qupath_roi.groovy`` script. Press CTRL + R and wait for the script to finish.

*You may choose to speed-up workflow by loading multiple SVS files into a QuPath project, and then running the script on the entire project using "Run for project."*	

Step 2: Data Preparation
************************

.. image:: tile_extraction.png

2) **Extract tiles**. Once ROIs have been created, tiles will need to be extracted from the ROIs across all of your slides. Tiles will be extracted at a given magnification size in microns, and saved at a given resolution in pixels. The optimal extraction size in both microns and pixels will depend on your dataset and model architecture.

.. image:: saving_tfrecords.png

3) **Create TFRecords**. Tiles should then be collected and stored as TFRecords, a binary file format used to improve dataset reading performance during training. Each slide should have its own TFRecord file containing its extracted tiles. 

.. image:: dataset_assembly.png

4) **Create validation set**. After TFrecords have been saved, a certain number of slides should be set aside for validation testing during training; slideflow will default to setting aside 20% of your slides for validation.

Step 3: Model Training
**********************

5) **Choose hyperparameters**. Before training can begin, you must choose both a model architecture (e.g. InceptionV3, VGG16, ResNet, etc.) and a set of hyperparameters (e.g. batch size, learning rate, etc.). One often does not know the best model architecture and hyperparameters to use for a given dataset; training will often need to occur across a variety of models and different combinations of hyperaparameters in order to find the combination with the best performance. You may either choose to train a single model and hyperparameter set one at a time, or you can setup an automatic hyperparameter sweep to test many combinations at once. 

6) **Initiate training**. After the hyperparameters have been set up, training can commence. Traing your model across all desired hyperparameters and select a hyperparameter combination for final model training and external evaluation testing.

Step 4: Analytics
*****************
Validation testing is performed both during training - at specified intervals, or epochs - and after training has completed. Validation testing is used to assess model performance and anticipated generalizability. Various metrics are recorded in the project directory at these intervals to assist with model performance assessment, including:

- **Training loss**
- **Validation loss**
- **Training accuracy** (for categorical outcomes)
- **Validation accuracy** (for categorical outcomes)
- **Tile-level, slide-level, and patient-level ROC and AUC** (for categorical outcomes)
- **Tile-level, slide-level, and patient-level scatter plots with R-squared** (for continuous outcomes)
- **Histograms of predictions** (for continuous outcomes)

Step 5: Visualizing Results with Heatmaps
*****************************************
In addition to the above metrics, performance of a trained model can be assessed by visualizing predictions for a set slides as heatmaps.

.. image:: heatmap_example.png
	
Step 6: Mosaic maps
*******************
Finally, learned image features can be visualized using dimensionality reduction on penultimate layer activations. 

A set of image tiles is first provided to your trained model, which calculate *penultimate* layer activations (as opposed to final layer activations, or predictions). These penultimate layer activations represent image features and are plotted with dimensionality reduction (UMAP/t-SNE). Points on the plot are replaced with image tiles, generating a mosaic map.

.. image:: mosaic_example.png
