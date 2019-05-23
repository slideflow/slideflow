Workflow
========

.. image:: pipeline.png

Workflow is separated into two phases and 6 steps. 

The first phase - **Model Creation** - involves three steps: 1) labeling slides with regions of interest (ROIs), 2) tessellating and preparing image tiles from the slides, and 3) training a neural network model. 

The second phase - **Model Assessment** - also involves three steps: 1) analytics describing model performance, including basic measures like percent accuracy as well as ROCs and dendrograms, 2) creating heatmap overlays for whole-slide images to visualize predictions, and 3) generating mosaic maps to visualize learned image features.

All six steps are described in detail below

Step 1: Create ROIs
*******************

1) **Label ROIs**. Using `QuPath <https://qupath.github.io/>`_, annotate whole-slide images with the Polygon tool. Then, load the Groovy script that will export the tile annotations into CSV format by clicking **Automate** -> **Show script editor**. In the box that comes up, click **File** -> **Open** and load the ``qupath_roi.groovy`` script. Then, press CTRL + R and wait for the script to finish.

*You may choose to speed-up workflow by loading multiple SVS files into a QuPath project, and then running the script on the entire project using "Run for project."*	

Step 2: Data Preparation
************************

2) **Extract tiles**. Use the script ``convoluter.py`` to tessellate image tiles from the ROIs. CSV files generated in the previous step should be included in the same directory as your SVS files. The syntax is as follows:

.. code-block:: bash

	$ python3 convoluter.py -s SVS_DIR -o PROJECT_DIR -c NUM_CLASSES --px TILE_PX
				--um TILE_UM --fp16 --export [--augment]
				--num_threads 8

...where :class:`SVS_DIR` is the directory to your SVS slides (and CSV ROI files), :class:`PROJECT_DIR` is the place to save the extracted tiles, and :class:`NUM_CLASSES` is the number of output classes. :class:`TILE_PX` equals the width of extract tiles in pixels, and :class:`TILE_UM` is the width of your extracted tiles in microns. Use the :class:`--augment` flag to augment your dataset with flipped/rotated images.

3) **Organize training directory**. Assemble image tiles into a directory tree as below, with the parent directory named :class:`train_data`, child directories 0 - *n* (where *n* indicates the number of unique categories), and grandchild directories named according to case number.

.. code-block:: none

	./train_data
	├── 0
	|   └──	234874-1
	|    	├── image.jpg
	|    	└── image.jpg
	|   └── 234877-2
	|    	├── image.jpg
	|    	└── image.jpg
	├── 1
	|   └── 234817-1
	|    	├── image.jpg
	|    	└── image.jpg
	|   └──	234892-1
	|    	└── image.jpg
	├── 2
	|   └── 234912-1
	|    	└── image.jpg
	|   └── 234991-1
	|    	└── image.jpg

4) **Create validation set**. Use the module ``data_utils`` to separate your data into training and validation sets. To create a validation set using 10% of your data, for example, use the following command:

.. code-block:: bash

	$ python3 data_utils.py --dir path/to/parent/directory --build
	
To re-merge a validation and training dataset, use the "--merge" flag:

.. code-block:: bash

	$ python3 data_utils.py --dir path/to/parent/directory --merge

Step 3: Model Training
**********************

After the training and validation set are prepared, training is as simple as running the main package:

.. code-block:: bash

	$ python3 histcon.py
	
To retrain an existing model, supply a "--retrain" flag with the directory of your pretrained model:

.. code-block:: bash

	$ python3 histcon.py --retrain path/to/pretrained/model
	
Hyperparameters, including image size, number of classes, batch size, learning rate, logging frequency, and so on, can be changed by editing the global constants in the :class:`HistconModel` class.

.. code-block:: python

	IMAGE_SIZE = 512
	NUM_CLASSES = 5

	NUM_EXAMPLES_PER_EPOCH = 1024

	# Constants for the training process.
	MOVING_AVERAGE_DECAY = 0.9999 		# Decay to use for the moving average.
	NUM_EPOCHS_PER_DECAY = 240.0		# Epochs after which learning rate decays.
	LEARNING_RATE_DECAY_FACTOR = 0.05	# Learning rate decay factor.
	INITIAL_LEARNING_RATE = 0.001			# Initial learning rate.

	# Variables previous created with parser & FLAGS
	BATCH_SIZE = 32
	WHOLE_IMAGE = '' # Filename of whole image (JPG) to evaluate with saved model
	MAX_EPOCH = 30
	LOG_FREQUENCY = 20 # How often to log results to console, in steps
	SUMMARY_STEPS = 20 # How often to save summaries for Tensorboard display, in steps
	TEST_FREQUENCY = 800 # How often to run validation testing, in steps
	USE_FP16 = True

During training, progress can be monitored using Tensorflow's bundled ``Tensorboard`` package:

.. code-block:: bash

	$ tensorboard --logdir=/path/to/histcon/models/active

... and then opening http://localhost:6006 in your web browser.

The "Custom Scalars" page displays both the training and validation loss (cross-entropy). Stop training by exiting the ``histcon`` program once loss has stabilized ("convergence") and before the validation loss starts becoming worse ("divergence"). Time to convergence varies based on the data and hyperparameters, but generally occurs within 2-30 epochs. One epoch equals one pass through the training data, and is equal to the :class:`batch size` * :class:`number of steps`.

Step 4: Analytics
*****************
Including model performance statistics, ROC, and dendrogram creation. Currently being implemented, documentation will be updated once complete.

Step 5: Visualizing Results with Heatmaps
*****************************************

After a model has been trained, accuracy can be assessed by visualizing predictions for a whole slide image.  The module ``convoluter`` loads a trained model, applies it convolutionally across a whole-slide image, and visualizes the results with a heatmap overlay.

.. code-block:: bash

	$ python3 convoluter.py --model path/to/model/dir --slide path/to/whole/case.svs
				--size 512 --classes 5 --batch 64 --px 512 --um 255 --fp16

To perform functions on a batch of slides, supply a folder of images instead of a single image: 

.. code-block:: bash

	$ python3 convoluter.py --model path/to/model/dir --slide path/to/whole/images
				--size 512 --classes 5 --batch 64 --px 512 --um 255 --fp16
	
To calculate predictions and save heatmap overlays as images, use the :class:`--save` flag:

.. code-block:: bash

	$ python3 convoluter.py --model path/to/model/dir --slide path/to/whole/images
				--size 512 --classes 5 --batch 64 --px 512 --um 255 --fp16 --save
	
Finally, to extract final layer weights for later use (e.g. visualization with dimensionality reduction), add the :class:`--final` flag. The weights will be saved to a csv file:

.. code-block:: bash

	$ python3 convoluter.py --model path/to/model/dir --slide path/to/whole/images
				--size 512 --classes 5 --batch 64 --px 512 --um 255 --fp16 --final
	
Step 6: Mosaic maps
*******************
Visualize learned image features with UMAP/t-SNE plots and mosaic maps. Code completed (``mosaic.py``), documentation pending.