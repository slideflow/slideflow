Training
========

Prepare hyperparameters
***********************

The next step is to prepare your hyperparameters and save them to a batch_train CSV file. Use the ``create_hyperparameter_sweep`` function to automatically set up a sweep of hyperparameter combinations. For example, the following code would set up a batch_train file with two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

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

As above, to execute this command, save your ``actions.py`` file and execute the ``run_project.py`` script in the slideflow directory.

Begin training
**************

Once your hyperparameter settings have been chosen and saved to your batch_train CSV file, you may begin training using the ``train`` function. Documentation of the function is given below:

.. autofunction:: slideflow.SlideFlowProject.train
   :noindex:

Your outcome variable is specified with the ``category_header`` argument. You may filter slides for training using the ``filter_header`` and ``filter_values`` arguments, as previously described. 

The ``train()`` function will automatically train across all hyperparameters listed in the batch file, and will use the validation plan supplied in project settings. If you would like to use a different plan than described in ``settings.json``, you many manually choose your validation strategy by passing the relevant variables as arguments (e.g. ``validation_strategy``, ``validation_fraction``, and ``validation_k_fold``).

If you are using a continuous variable as an outcome measure, set the argument ``model_type`` equal to 'linear'.

For example, to train using only slides labeled as "train" in the "dataset" column, with the outcome variable defined by the column "category", use the following syntax:

.. code-block:: python

	SFP.train(category_header="category",
		  filter_header=["dataset"],
		  filter_values=["train"])

Once training has finished, performance metrics - including accuracy, loss, etc. - can be found in the ``results.log`` file in the project directory. Additional analytic data, including ROCs and scatter plots, are saved in the model directories.

Monitoring performance
**********************

During training, progress can be monitored using Tensorflow's bundled ``Tensorboard`` package:

.. code-block:: bash

	$ tensorboard --logdir=/path/to/histcon/models/active

... and then opening http://localhost:6006 in your web browser.
