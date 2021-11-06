Training
========

Prepare hyperparameters
***********************

There are two methods for configuring model hyperparameters. If you intend to train a model using a single combination of hyperparameters, use the ``ModelParams`` class:

.. code-block:: python

    hp = sf.model.ModelParams(
        epochs=[1, 5],
        model='xception',
        loss='sparse_categorical_crossentropy',
        learning_rate=0.00001,
        batch_size=8)

Alternatively, if you intend to perform a sweep across multiple hyperparameter combinations, use the ``create_hyperparameter_sweep`` function to automatically save a sweep to a TSV file. For example, the following would set up a batch_train file with two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

.. code-block:: python

    P.create_hyperparameter_sweep(
        epochs=[5],
        toplayer_epochs=0,
        model=['xception'],
        pooling=['avg'],
        loss='sparse_categorical_crossentropy',
        learning_rate=[0.01, 0.001],
        batch_size=64,
        hidden_layers=[1],
        optimizer='Adam',
        early_stop=True,
        early_stop_patience=15,
        training_balance=['category'],
        validation_balance='none',
        augment=True)

Available hyperparameters include:

- **tile_px** - size of extracted tiles in pixels
- **tile_um** - size of extracted tiles in microns
- **epochs** - number of epochs to spend training the full model
- **toplayer_epochs** - number of epochs to spend training just the final layer, with all convolutional layers "locked" (sometimes used for transfer learning)
- **model** - model architecture; please see `Keras application documentation <https://keras.io/applications/>`_ for all options
- **pooling** - pooling strategy to use before final fully-connected layers; either 'max', 'avg', or 'none'
- **loss** - loss function; please see `Keras loss documentation <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_ for all options
- **learning_rate** - learning rate for training
- **learning_rate_decay** - learning rate decay during training
- **learning_rate_decay_steps** - number of steps after which to decay learning rate
- **batch_size** - batch size for training
- **hidden_layers** - number of fully-connected final hidden layers before softmax prediction
- **hidden_layer_width** - width of hidden layers
- **optimizer** - training optimizer; please see `Keras opt documentation <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ for all options
- **early_stop** - whether to use early stopping if validation loss is not decreasing
- **early_stop_patience** - number of epochs to wait before allowing early stopping
- **early_stop_method** - metric to use for early stopping, e.g. 'loss' or 'accuracy'
- **training_balance** - training input balancing strategy; please see :ref:`balancing` for more details
- **validation_balance** - validation input balancing strategy; please see :ref:`balancing` for more details
- **trainable_layers** - number of layers available for training, other layers will be frozen. If 0, all layers are trained
- **L2_weight** - if provided, adds L2 regularization to all layers with this weight
- **dropout** - dropout, used for post-convolutional layer.
- **augment** - Image augmentations to perform, including flipping/rotating and random JPEG compression. Please see :class:`slideflow.model.ModelParams` for more details.

If you are using a continuous variable as an outcome measure, be sure to use a linear loss function. Linear loss functions can be viewed in ``slideflow.model.ModelParams.LinearLossDict``, and all available loss functions are in ``slideflow.model.ModelParams.AllLossDict``.

Begin training
**************

Once your hyperparameter settings have been chosen you may begin training using the ``train`` function. Documentation of the function is given below:

.. autofunction:: slideflow.project.Project.train
   :noindex:

If you used the ``ModelParams`` class to configure a single combination of parameters, pass this object via the ``params`` argument. If you configured a hyperparameter sweep, set the ``batch_file`` argument to the name of your hyperparameter sweep file (saved by default to 'batch_train.tsv').

Your outcome variable(s) are specified with the ``outcome_label_headers`` argument. You may filter slides for training using the ``filter`` argument, as previously described.

For example, to train using only slides labeled as "train" in the "dataset" column, with the outcome variable defined by the column "category", use the following syntax:

.. code-block:: python

    P.train(outcome_label_headers="category",
          filters={"dataset": ["train"]},
          batch_file='batch_train.tsv')

If you would like to use a different validation plan than the default, pass the relevant keyword arguments to the training function.

Once training has finished, performance metrics - including accuracy, loss, etc. - can be found in the ``results_log.csv`` file in the project directory. Additional data, including ROCs and scatter plots, are saved in the model directories.

At each designated epoch, models are saved in their own folders. Each model directory will include a copy of its hyperparameters in a ``params.json`` file, and a copy of its training/validation slide manifest in ``slide.log``.

Multiple outcomes
*****************

Slideflow supports both categorical and linear outcomes, as well as training to single or multiple outcomes at once. To use multiple outcomes simultaneously, simply pass multiple annotation headers to the ``outcome_label_headers`` argument.

Multiple input variables
************************

In addition to training using image data, clinical data can also be provided as model input by passing annotation column headers to the variable ''input_header''. This input is merged at the post-convolutional layer, prior to any configured hidden layers.

If desired, models can also be trained with clinical input data alone, without images, by using the hyperparameter argument ``drop_images=True``.

Cox Proportional Hazards (CPH) models
*************************************

Models can also be trained to a time series outcome using CPH and negative log likelihood loss. For CPH models, use 'negative_log_likelihood' loss and set ``outcome_label_header`` equal to the annotation column indicating event *time*. Specify the event *type* (0 or 1) by passing the event type annotation column to the argument ``input_header``. If you are using multiple clinical inputs, the first header passed to ``input_header`` must be event type. CPH models are not compatible with multiple outcomes.

Distributed training across GPUs
********************************

If multiple GPUs are available, training can be distributed by passing the argument ``multi_gpu=True``. If provided, slideflow will use all available (and visible) GPUs for training.

.. note::
    There is currently a bug in Tensorflow 2.5+ with Python 3.8+ which prevents multi gpu training using MirroredStrategy. You can follow the issue on `Tensorflow Github <https://github.com/tensorflow/tensorflow/issues/50487>`_.

Monitoring performance
**********************

During training, progress can be monitored using Tensorflow's bundled ``Tensorboard`` package by passing the argument ``use_tensorboard=True``. This functionality was disabled by default due to a recent bug in Tensorflow. To use tensorboard to monitor training, execute:

.. code-block:: bash

    $ tensorboard --logdir=/path/to/model/directory

... and open http://localhost:6006 in your web browser.
