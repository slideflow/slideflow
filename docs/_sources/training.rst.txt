Training
========

Prepare hyperparameters
***********************

The first step of model training is configuring a set of model parameters and hyperparameters. There are two methods for configuring model parameters. If you intend to train a model with a single set of parameters, use the ``ModelParams`` class:

.. code-block:: python

    import slideflow as sf

    hp = sf.ModelParams(
      epochs=[1, 5],
      model='xception',
      learning_rate=0.0001,
      batch_size=8,
      ...
    )

Use :meth:`slideflow.Project.create_hp_sweep()` to prepare a grid-search sweep, saving the configuration to a JSON file. For example, the following would configure a sweep with only two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

.. code-block:: python

    P.create_hp_sweep(
      filename='sweep.json',
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

The sweep is then executed by passing the JSON path to the ``params`` argument of ``Project.train()``:

.. code-block:: python

    P.train(params='sweep.json', ...)

In addition to grid-search sweeps, you can also perform Bayesian hyperparameter optimization using `SMAC3 <https://automl.github.io/SMAC3/master/>`_. Start by setting the `configuration space <https://automl.github.io/ConfigSpace/master/>`_:

.. code-block:: python

    import ConfigSpace.hyperparameters as cs_hp
    from ConfigSpace import ConfigurationSpace

    cs = ConfigurationSpace()
    cs.add_hyperparameter(cs_hp.UniformIntegerHyperparameter("epochs", 1, 2))
    cs.add_hyperparameter(cs_hp.UniformFloatHyperparameter("dropout", 0, 0.5))

Then, simply replace ``Project.train()`` with :meth:`slideflow.Project.smac_search()`, providing the configuration space to the argument ``smac_configspace``:

.. code-block:: python

    P.train(..., smac_configspace=cs)

Available hyperparameters are listed in the :class:`slideflow.model.ModelParams` documentation.

.. note::

    If you are using a continuous variable as an outcome measure, be sure to use a linear loss function. Linear loss functions can be viewed in ``slideflow.model.ModelParams.LinearLossDict``, and all available loss functions are in ``slideflow.model.ModelParams.AllLossDict``.

Begin training
**************

Once your hyperparameter settings have been chosen, you may begin training using the ``Project.train()`` function:

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

Finding results
***************

Performance metrics - including accuracy, loss, etc. - are returned from the ``Project.train()`` function as a dictionary and saved in ``results_log.csv`` files in both the project directory and model directory. Additional data, including ROCs and scatter plots, are saved in the model directories. Pandas DataFrames containing tile-, slide-, and patient-level predictions are also saved in the model directory.

At each designated epoch, models are saved in their own folders. Each model directory will include a copy of its hyperparameters in a ``params.json`` file, and a copy of its training/validation slide manifest in ``slide.log``.

Multiple outcomes
*****************

Slideflow supports both categorical and continuous outcomes, as well as training to single or multiple outcomes at once. To train with multiple outcomes simultaneously, simply pass multiple annotation headers to the ``outcomes`` argument.

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
