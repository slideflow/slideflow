
Extract tiles
*************

Our first step is tile extraction. To configure our project to extract tiles, uncomment the line with the ``extract_tiles()`` function. If you want to extract tiles for all of your slides, you may delete all of the suggested arguments and simply use the following syntax:

.. code-block:: python

	SFP.extract_tiles()

The optional ``filter_header`` and ``filter_values`` arguments are used to filter a subset of slides to act on, according to your annotations file. If not supplied, all valid slides will be used by default.

To filter according to a column in your annotations file, set ``filter_header`` to the column name. If you are filtering by multiple columns, supply names in a list. Next, set ``filter_values`` to a list containing all possible values you want to include. If you are filtering by multiple columns, this will be a nested list.

For example, to extract tiles only for slides that are labeled as "train" in the "dataset" column header in your annotations file, do:

.. code-block:: python

	SFP.extract_tiles(filter_header="dataset", filter_values=["train"])

Note: this same syntax with ``filter_header`` and ``filter_values`` can be used with nearly all functions in slideflow, including train(), evaluate(), generate_heatmaps(), and generate_mosaic().


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
