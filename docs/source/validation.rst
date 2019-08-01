.. _validation_planning:

Validation Planning
===================

The easiest way to get the ``slideflow`` pipeline up and running is to use the bundled project management class, ``SlideFlowProject``. In this section, we will examine how to set up a new project and then use the project to execute each of the pipeline steps. 

Before we start, make sure you have each of the following:

1.	A collection of slides in SVS format.
2.	A collection of ROIs in CSV format, generated using QuPath.
3.	A plan for which slides will be used for training and which will be used for final testing.

We will use the ``run_project.py`` script in the slideflow directory to both create and manage our project. To create a new project (or run commands on an existing project), use the following syntax:

.. code-block:: console

	james@example:~/slideflow/source$ python3 run_project.py -p /path/to/project/directory

...where the -p flag is used to designate the path to your project directory.

Project Configuration
*********************

Upon first executing the script, you will be asked a series of questions regarding your project. Default answers are given in brackets (if the question is a yes/no question, the default answer is the letter which is capitalized); if you press enter without typing anything, the default will be chosen. You can always change your answers later by editing ``settings.json`` in your project folder. Below is an overview of what you’ll be asked for.

+-------------------------------+-------------------------------------------------------+
| **Slides directory** 		| Path to folder containing SVS slides. 		|
+-------------------------------+-------------------------------------------------------+
| **ROI directory**		| Path to folder with ROI CSV files. 			|
+-------------------------------+-------------------------------------------------------+
| **Annotations file**		| Path to CSV containing annotations.   		|
|				| Each line represents a unique patient and slide.	|
+-------------------------------+-------------------------------------------------------+
| **Tiles directory**		| Folder in which to store extracted tiles.		|
+-------------------------------+-------------------------------------------------------+
| **TFRecord directory**	| Folder in which to store TFRecords.			|
+-------------------------------+-------------------------------------------------------+
| **Models directory**		| Folder in which to store trained models.		|
+-------------------------------+-------------------------------------------------------+
| **Pretraining directory**	| Whether to use pretraining. 'imagenet' is default, 	|
|				| but able to use any saved .h5 model.			|
+-------------------------------+-------------------------------------------------------+
| **Tile microns**		| Size of extracted tiles, in microns.			|
+-------------------------------+-------------------------------------------------------+
| **Tile pixels**		| Size of extracted tiles, in pixels.			|
+-------------------------------+-------------------------------------------------------+
| **Validation fraction**	| Fraction of data to save for validation testing.	|
|				| Default is 20%.					|
+-------------------------------+-------------------------------------------------------+
| **Validation target**		| How to select validation data; by tile or by slide.	|
|				| Default is 'per-slide'				|
+-------------------------------+-------------------------------------------------------+
| **Validation strategy**	| Type of validation testing (K-fold, fixed plan, none)	|
|				| Default is 'k-fold'					|
+-------------------------------+-------------------------------------------------------+

For more information about setting up a validation plan, see :ref:`validation_planning`.

Set up annotations
******************

Your annotations CSV file is used to label patients and slides with clinical data and/or other outcome variables that will be used for training.
Each line in the annotations file should correspond to a unique patient/slide (*Note: v0.9.9 of slideflow currently supports only one slide per patient*).

The annotations file may contain as many columns as you would like, but it must contain the following headers at minimum:

- **submitter_id**: patient identifier
- **slide**: slide name (without the .jpg/.svs extension)
- **category**: some outcome variable

Extract tiles
*************

After the project has been setup, open the ``actions.py`` file located in the project directory. It should look something like this:

.. code-block:: python

	def main(SFP):
		#SFP.create_blank_annotations_file(scan_for_cases=True)
		#SFP.associate_slide_names()
		
		#SFP.extract_tiles(filter_header=['dataset'], filter_values=['train'])
			   
		#SFP.create_hyperparameter_sweep(finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
		#				learning_rate=[0.00001, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
		#				balanced_validation='NO_BALANCE', augment=True, filename=None)
		#SFP.train(subfolder='train1-per-slide',
		#	  category_header="submitter_id",
		#	  filter_header=["dataset"],
		#	  filter_values=["train1"],
		#	  model_type='linear')
		#	  batch_file='batch_train.tsv')

		#SFP.evaluate(model='HPSweep0', subfolder='train_data', category_header="description", filter_header='description', filter_values=['FTC'])
		#SFP.generate_heatmaps('HPSweep0', filter_header='slide', filter_values=['234801'])
		#SFP.generate_mosaic(model="/home/shawarma/data/slideflow_projects/thyroid_5_cat/models/HPSweep0/trained_model.h5", subfolder="train")
		pass

The ``main()`` function contains several example commands, commented out with "#". These serve as examples to help remind you of arguments you can use when executing project functions.

Our first step is tile extraction. To configure our project to extract tiles, uncomment the line with the ``extract_tiles()`` function. If you want to extract tiles for all of your slides, you may delete all of the suggested arguments and simply use the following syntax:

.. code-block:: python

	SFP.extract_tiles()

The optional ``filter_header`` and ``filter_values`` arguments are used to filter a subset of slides to act on, according to your annotations file. If not supplied, all valid slides will be used by default.

To filter according to a column in your annotations file, set ``filter_header`` to the column name. If you are filtering by multiple columns, supply names in a list. Next, set ``filter_values`` to a list containing all possible values you want to include. If you are filtering by multiple columns, this will be a nested list.

For example, to extract tiles only for slides that are labeled as "train" in the "dataset" column header in your annotations file, do:

.. code-block:: python

	SFP.extract_tiles(filter_header="dataset", filter_values=["train"])

Note: this same syntax with ``filter_header`` and ``filter_values`` can be used with nearly all functions in slideflow, including train(), evaluate(), generate_heatmaps(), and generate_mosaic().

To execute the command we've prepared, save the ``actions.py`` file and go to your slideflow directory. Use ``run_project.py`` to begin the tile extraction:

.. code-block:: console

	james@example:~/slideflow/source$ python3 run_project.py -p /path/to/project/directory

Tiles will be automatically stored in TFRecord format and separated into training and validation steps if necessary.

Prepare hyperparameters
***********************

The next step is to prepare your hyperparameters and save them to a batch_train CSV file. Use the ``create_hyperparameter_sweep`` function to automatically set up a sweep of hyperparameter combinations. For example, the following code would set up a batch_train file with two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

.. code-block:: python

	SFP.create_hyperparameter_sweep(finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
					learning_rate=[0.01, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
					balanced_validation='NO_BALANCE', augment=True, filename=None)

As above, to execute this command, save your ``actions.py`` file and execute the ``run_project.py`` script in the slideflow directory.

Train models
************

To start training, use the ``train()`` function, supplying an outcome variable with the ``category_header`` argument. You may filter slides for training using the ``filter_header`` and ``filter_values`` arguments, as above. 

The ``train()`` function will automatically train across all hyperparameters listed in the batch file. It will also automatically train using k-fold validation if supplied in project settings.

If you are using a continuous variable as an outcome measure, set the argument ``model_type`` equal to 'linear'.

For example, to train using only slides labeled as "train" in the "dataset" column, with the outcome variable defined by the column "category", use the following syntax:

.. code-block:: python

	SFP.train(category_header="category",
		  filter_header=["dataset"],
		  filter_values=["train"])

Once training has finished, performance metrics - including accuracy, loss, etc. - can be found in the ``results.log`` file in the project directory. Additional analytic data, including ROCs and scatter plots, are saved in the model directories.

Evaluate models
***************

Once you have finished your hyperparameter selection and would like to test your model on a saved external evaluation dataset, you can perform a model evaluation using the ``evaluate`` function. Specify the model you want to test with the ``model`` argument.

For example, to evaluate performance of model "HPSweep0" on slides labeled as "evaluation" in the "dataset" column of our annotations file, use the following:

.. code-block:: python

	SFP.evaluate(model="HPSweep0",
		  category_header="category",
		  filter_header=["dataset"],
		  filter_values=["evaluation"])

Generate heatmaps
*****************

If you would like to generate a predictive heatmap for a set of slides, use the ``generate_heatmaps()`` function as below:

.. code-block:: python

	SFP.generate_heatmaps(model="HPSweep0",
		  filter_header=["dataset"],
		  filter_values=["evaluation"])

Generate mosaic maps
********************

You can also generate mosaic maps using similar syntax to the above. In addition to simply supplyling a model name, you can also provide a saved \*.h5 model directly:

.. code-block:: python

	SFP.generate_mosaic(model="/path/to/saved/model.h5",
		  filter_header=["dataset"],
		  filter_values=["evaluation"])
