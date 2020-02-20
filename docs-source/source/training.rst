Training
========

Prepare hyperparameters
***********************

There are two methods for configuring model hyperparameters. If you intend to train a model using a single combination of hyperparameters, use the ``HyperParameters`` class:

.. code-block:: python

	from slideflow.model import HyperParameters
	hp = HyperParameters(finetune_epochs=[1], toplayer_epochs=0, model='Xception', pooling='avg', loss='sparse_categorical_crossentropy',
				learning_rate=0.00001, batch_size=8, hidden_layers=1, optimizer='Adam', early_stop=False, 
				early_stop_patience=0, balanced_training='BALANCE_BY_PATIENT', balanced_validation='NO_BALANCE', 
				augment=True)

Alternatively, if you intend to perform a sweep across multiple hyperparameter combinations, use the ``create_hyperparameter_sweep`` function to automatically configure a sweep to a batch_train CSV file. For example, the following code would set up a batch_train file with two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

.. code-block:: python

	SFP.create_hyperparameter_sweep(finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
					learning_rate=[0.01, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
					balanced_validation='NO_BALANCE', augment=True, filename=None)

Available hyperparameters include:

- **finetune_epochs** - number of epochs to spend training the full model
- **toplayer_epochs** - number of epochs to spend training just the final layer, with all convolutional layers "locked" (sometimes used for transfer learning)
- **model** - model architecture; please see `Keras application documentation <https://keras.io/applications/>`_ for all options
- **pooling** - pooling strategy to use before final fully-connected layers; either 'max', 'avg', or 'none'
- **loss** - loss function; please see `Keras loss documentation <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_ for all options
- **learning_rate** - learning rate for training
- **batch_size** - batch size for training
- **hidden_layers** - number of fully-connected final hidden layers before softmax prediction
- **optimizer** - training optimizer; please see `Keras opt documentation <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ for all options
- **early_stop** - whether to use early stopping if validation loss is not decreasing
- **early_stop_patience** - number of epochs to wait before allowing early stopping
- **balanced_training** - training input balancing strategy; please see :ref:`balancing` for more details
- **balanced_validation** - validation input balancing strategy; please see :ref:`balancing` for more details
- **augment** - whether to augment data with random flipping/rotating during training

Begin training
**************

Once your hyperparameter settings have been chosen you may begin training using the ``train`` function. Documentation of the function is given below:

.. autofunction:: slideflow.SlideflowProject.train
   :noindex:

If you used the ``HyperParameters`` class to configure a single combination of parameters, pass this object via the ``hyperparameters`` argument. If you configured a hyperparameter sweep, set the ``batch_file`` argument to the name of your hyperparameter sweep file (saved by default to 'batch_train.tsv').

Your outcome variable is specified with the ``outcome_header`` argument. You may filter slides for training using the ``filter`` argument, as previously described. 

The validation settings configured in the project settings file (``settings.json``) will be used by default. If you would like to use a different validation plan than the default configuration, you many manually pass the relevant variables as arguments (e.g. ``validation_strategy``, ``validation_fraction``, and ``validation_k_fold``).

If you are using a continuous variable as an outcome measure, set the argument ``model_type`` equal to 'linear'.

For example, to train using only slides labeled as "train" in the "dataset" column, with the outcome variable defined by the column "category", use the following syntax:

.. code-block:: python

	SFP.train(outcome_header="category",
		  filters={"dataset": ["train"]},
		  batch_file='batch_train.tsv')


To begin training, save your ``actions.py`` file and execute the ``run_project.py`` script in the slideflow directory.

Once training has finished, performance metrics - including accuracy, loss, etc. - can be found in the ``results.log`` file in the project directory. Additional analytic data, including ROCs and scatter plots, are saved in the model directories.

Monitoring performance
**********************

During training, progress can be monitored using Tensorflow's bundled ``Tensorboard`` package:

.. code-block:: bash

	$ tensorboard --logdir=/path/to/model/directory

... and then opening http://localhost:6006 in your web browser.
