Introduction
============

``histcon`` is a Python package which aims to provide an easy and intuitive way of building and testing convolutional neural networks (CNNs) for use in histology image analysis. It is built using ``Tensorflow`` (see `here <https://www.tensorflow.org/>`_) and currently utilizes Google's `Inception-v4 <https://github.com/tensorflow/models/tree/master/research/slim>`_ network architecture.

The ``histcon`` object will initialize data input streams, build an Inception-V4 model, and initiate training upon calling ``histcon.train``. Re-training (transfer learning) is available through ``histcon.retrain``. After a model has been trained, whole-slide-image predictions can be generated and visualized with heatmap overlays using ``convoluter``. Other bundled objects, including ``data_utils``, ``nconvert_util``, and ``packing``, provide easy-to-use tools for generating datasets from annotated whole-slide images.

The current implementation has been developed in Python 3 using Tensorflow 1.12 and tested in Ubuntu 18.04, but should work on most Debian-based OSs.

Workflow: Data Preparation
**************************

*If the training images are to be obtained from whole-slide images in SVS format, the following applies:*

1) **Whole-slide image tile extraction**. Using `QuPath <https://qupath.github.io/>`_, annotate whole-slide images with the Polygon tool. You may need to delete old tiles if annotations have already been performed. To do so, click the **"Hierarchy"** tab in the upper-left corner, click the arrow next to one of the annotation Polygons, and select all of the "Tiles" beneath. Then, click the **"Annotations"** tab and press **"Delete"**. Do this for each of the Polygon annotations.

If no tiles or annotations have been generated, first create new annotations using the Polygon tool to select the relevant areas of interest. Then, select all of your Polygons in the **Hierarchy** tab. Then, in the main menu, click **Analyze -> Region identification -> Tiles & superpixels -> Create tiles**. Change the tile size to whatever size you're trying to extract, in μm. **Uncheck** the "trim to ROI" option, and **check** the "make annotation tiles" option. Then, click run.

Finally, load the script that will export the tile annotations into saved JPG files by clicking **Automate** -> **Show script editor**. In the box that comes up, click **File** -> **Open** and load the "qupath_tile_export.groovy" script. Change the directory in Line 17 to whatever directory you want your exported tiles to sit in. Then, press CTRL + R and wait for the script to finish.

*If the training data has already been sectioned into tiles, begin the workflow at this point:*	

2) **Resize tiles**. Use the module ``nconvert_util`` (requires the separate linux package :class:`nconvert`) to resize extracted image tiles. If, for example, you needed to extract tiles of size 2048 px for a magnification size 560 µm, but want to train on a tile size of 512 px, you would reduce your extracted 2048x2048 tiles to 25% resolution (512x512) using the command below. Supply the "--augment" flag to also generate flipped/rotated images.

.. code-block:: bash

    $ python3 nconvert_util.py --dir /path/to/tile/directory --size 512 --augment

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

Workflow: Model Training
************************

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

Workflow: Visualizing Results
*****************************

After a model has been trained, accuracy can be assessed by visualizing predictions for a whole slide image.  The module ``convoluter`` loads a trained model, applies it convolutionally across a whole-slide image, and visualizes the results with a heatmap overlay.

.. code-block:: bash

	$ python3 convoluter.py --dir path/to/model/dir --image path/to/whole/image.jpg --size 512 --classes 5 --batch 128 --fp16
	
In order to reduce computational waste, calculated logits are saved in a "pkl" file. To load a pre-calculated pkl file instead of re-calculating heatmap logits, use the following syntax:

.. code-block:: bash

	$ python3 convoluter.py --load path/to/weights.pkl --image path/to/whole/image.jpg --size 512 --classes 5
	
To batch calculate weights and save .pkl files, supply a folder of images instead of a single image: 

.. code-block:: bash

	$ python3 convoluter.py --dir path/to/model/dir --folder path/to/whole/images --size 512 --classes 5 --batch 128 --fp16
	
Finally, to load a folder of .pkl files and batch save heatmap overlays as JPEG images, use the "--save" flag:

.. code-block:: bash

	$ python3 convoluter.py --folder path/to/whole/images_and_pkls --size 512 --classes 5 --save