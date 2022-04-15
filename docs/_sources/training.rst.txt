Training
========

Prepare hyperparameters
***********************

The first step of model training is configuring a set of model parameters / training hyperparameters. There are two methods for configuring model parameters. If you intend to train a model using a single combination of hyperparameters, use the ``ModelParams`` class:

.. code-block:: python

    import slideflow as sf

    hp = sf.ModelParams(
      epochs=[1, 5],
      model='xception',
      learning_rate=0.0001,
      batch_size=8,
      ...
    )

Alternatively, if you intend to perform a sweep across multiple hyperparameter combinations, use the ``Project.create_hp_sweep()`` function to automatically save a sweep to a JSON file. For example, the following would set up a batch_train file with two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

.. code-block:: python

    P.create_hp_sweep(
      epochs=[5],
      toplayer_epochs=0,
      model=['xception'],
      pooling=['avg'],
      loss='sparse_categorical_crossentropy',
      learning_rate=[0.001, 0.0001],
      batch_size=64,
      hidden_layers=[1],
      optimizer='Adam',
      augment='xyrj'
    )

Available hyperparameters include:

- **augment** - Image augmentations to perform, including flipping/rotating and random JPEG compression. Please see :class:`slideflow.model.ModelParams` for more details.
- **batch_size** - Batch size for training.
- **dropout** - Adds dropout layers after each fully-connected layer.
- **early_stop** - Stop training early if validation loss/accuracy is not decreasing.
- **early_stop_patience** - Number of epochs to wait before allowing early stopping.
- **early_stop_method** - mMtric to use for early stopping. Includes 'loss', 'accuracy', or 'manual'.
- **epochs** - Number of epochs to spend training the full model.
- **include_top** - Include the default, preconfigured, fully connected top layers of the specified model.
- **hidden_layers** - Number of fully-connected final hidden layers before softmax prediction.
- **hidden_layer_width** - Width of hidden layers.
- **l1** - Adds L1 regularization to all convolutional layers with this weight.
- **l1_dense** - Adds L1 regularization to all fully-conected Dense layers with this weight.
- **l2** - Adds L2 regularization to all convolutional layers with this weight.
- **l2_dense** - Adds L2 regularization to all fully-conected Dense layers with this weight.
- **learning_rate** - Learning rate for training.
- **learning_rate_decay** - lLarning rate decay during training.
- **learning_rate_decay_steps** - Number of steps after which to decay learning rate
- **loss** - loss function; please see `Keras loss documentation <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_ for all options.
- **manual_early_stop_epoch** - Manually trigger early stopping at this epoch/batch.
- **manual_early_stop_batch** - Manually trigger early stopping at this epoch/batch.
- **model** - Model architecture; please see `Keras application documentation <https://keras.io/applications/>`_ for all options.
- **normalizer** - Normalization method to use on images.
- **normalizer_source** - Optional path to normalization image to use as the source.
- **optimizer** - Training optimizer; please see `Keras opt documentation <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ for all options.
- **pooling** - Pooling strategy to use before final fully-connected layers; either 'max', 'avg', or 'none'.
- **tile_px** - Size of extracted tiles in pixels.
- **tile_um** - Size of extracted tiles in microns.
- **toplayer_epochs** - Number of epochs to spend training just the final layer, with all convolutional layers "locked" (sometimes used for transfer learning).
- **trainable_layers** - Number of layers available for training, other layers will be frozen. If 0, all layers are trained.
- **training_balance** - Training input balancing strategy; please see :ref:`balancing` for more details.
- **uq** - Enable uncertainty quantification (UQ) during inference. Requires dropout to be non-zero.
- **validation_balance** - Validation input balancing strategy; please see :ref:`balancing` for more details.

If you are using a continuous variable as an outcome measure, be sure to use a linear loss function. Linear loss functions can be viewed in ``slideflow.model.ModelParams.LinearLossDict``, and all available loss functions are in ``slideflow.model.ModelParams.AllLossDict``.

Begin training
**************

Once your hyperparameter settings have been chosen you may begin training using the ``train`` function. Documentation of the function is given below:

.. autofunction:: slideflow.Project.train
   :noindex:

If you used the ``ModelParams`` class to configure a single combination of parameters, pass this object via the ``params`` argument. If you configured a hyperparameter sweep, set this argument to the name of your hyperparameter sweep file (saved by default to 'sweep.json').

Your outcome variable(s) are specified with the ``outcomes`` argument. You may filter slides for training using the ``filter`` argument, as previously described.

For example, to train using only slides labeled as "train" in the "dataset" column, with the outcome variable defined by the column "category", use the following syntax:

.. code-block:: python

    P.train(
      outcomes="category",
      filters={"dataset": ["train"]},
      params='sweep.json'
    )

If you would like to use a different validation plan than the default, pass the relevant keyword arguments to the training function.

Once training has finished, performance metrics - including accuracy, loss, etc. - can be found in the ``results_log.csv`` file in the project directory. Additional data, including ROCs and scatter plots, are saved in the model directories.

At each designated epoch, models are saved in their own folders. Each model directory will include a copy of its hyperparameters in a ``params.json`` file, and a copy of its training/validation slide manifest in ``slide.log``.

Multiple outcomes
*****************

Slideflow supports both categorical and continuous outcomes, as well as training to single or multiple outcomes at once. To use multiple outcomes simultaneously, simply pass multiple annotation headers to the ``outcomes`` argument.

Multiple input variables
************************

In addition to training using image data, clinical data can also be provided as model input by passing annotation column headers to the variable ''input_header''. This input is merged at the post-convolutional layer, prior to any configured hidden layers.

If desired, models can also be trained with clinical input data alone, without images, by using the hyperparameter argument ``drop_images=True``.

Cox Proportional Hazards (CPH) models
*************************************

Models can also be trained to a time series outcome using CPH and negative log likelihood loss. For CPH models, use `'negative_log_likelihood'` loss and set ``outcomes`` equal to the annotation column indicating event *time*. Specify the event *type* (0 or 1) by passing the event type annotation column to the argument ``input_header``. If you are using multiple clinical inputs, the first header passed to ``input_header`` must be event type. CPH models are not compatible with multiple outcomes.

.. note::
    CPH models are currently unavailable with the PyTorch backend. PyTorch support for CPH outcomes is in development.

Distributed training across GPUs
********************************

If multiple GPUs are available, training can be distributed by passing the argument ``multi_gpu=True``. If provided, slideflow will use all available (and visible) GPUs for training.

Monitoring performance
**********************

During training, progress can be monitored using Tensorflow's bundled ``Tensorboard`` package by passing the argument ``use_tensorboard=True``. This functionality was disabled by default due to a recent bug in Tensorflow. To use tensorboard to monitor training, execute:

.. code-block:: bash

    $ tensorboard --logdir=/path/to/model/directory

... and open http://localhost:6006 in your web browser.
