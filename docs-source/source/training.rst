.. _training:

Training
========

Slideflow offers tools for training many types of neural networks, including:

- **Weakly supervised, tile-based models**: Models trained on image tiles, with labels inherited from the parent slide.
- **Weakly supervised, multi-instance learning**: Models trained on feature vectors, with labels inherited from the parent slide.
- **Strongly supervised models**: Models trained on image tiles, with labels assigned by ROI.
- **Self-supervised pretraining**: Contrastive pretraining with or without labels (e.g. `SimCLR <https://arxiv.org/abs/2002.05709>`_).
- **Generative adversarial networks**: Models trained to generate synthetic images (e.g. `StyleGAN2/3 <https://arxiv.org/abs/1912.04958>`_).
- **Segmentation models**: Models trained to identify and classify tissue regions (e.g. `U-Net <https://arxiv.org/abs/1505.04597>`_).

In this section, we will walk through the process of training a weakly supervised tile-based model. :ref:`Strong supervision <tile_labels>`, :ref:`Multi-instance learning (MIL) <mil>`, :ref:`self-supervised pretraining (SSL) <simclr_ssl>`, :ref:`generative adversarial networks (GAN) <stylegan>`, and :ref:`segmentation` are described in other sections.

Prepare hyperparameters
***********************

The first step of training a weakly-supervised model is configuring model parameters and hyperparameters with :class:`slideflow.ModelParams`. ``ModelParams`` determines the model architecture, loss, preprocessing augmentations, and training hyperparameters.

.. code-block:: python

    import slideflow as sf

    hp = sf.ModelParams(
      epochs=[1, 5],
      model='xception',
      learning_rate=0.0001,
      batch_size=8,
      ...
    )

See the :class:`slideflow.ModelParams` API documentation for a list of available hyperparameters.

.. note::

    If you are using a continuous variable as an outcome measure, be sure to use a regression loss function. Regression loss functions can be viewed in ``slideflow.ModelParams.RegressionLossDict``, and all available loss functions are in ``slideflow.ModelParams.AllLossDict``.

Training a model
****************

Slideflow provides two methods for training models: with the high-level :meth:`slideflow.Project.train` function or with the lower-level :class:`slideflow.model.Trainer`. The former provides an easier interface for executing complex training tasks with a single function call, while the latter provides lower-level access for greater customizability.

.. _training_with_project:

Training with a Project
-----------------------

:meth:`slideflow.Project.train` provides an easy API for executing complex training plans and organizing results in the project directory. This is the recommended way to train models in Slideflow. There are two required arguments for this function:

- ``outcomes``: Name (or list of names) of annotation header columns, from which to determine slide labels.
- ``params``: Model parameters.

The default validation plan is three-fold cross-validation, but the validation strategy can be customized via keyword arguments (``val_strategy``, ``val_k_fold``, etc) as described in the API documentation. If crossfold validation is used, each model in the crossfold will be trained sequentially. Read more about :ref:`validation strategies <validation_strategies>`.

By default, all slides in the project will be used for training. You can restrict your training/validation data to only a subset of slides in the project with one of two methods: either by providing ``filters`` or a filtered :class:`slideflow.Dataset`.

For example, you can use the ``filters`` argument to train/validate only using slides labeled as "train_and_val" in the "dataset" column with the following syntax:

.. code-block:: python

    results = P.train(
      outcomes="tumor_type",
      params=sf.ModelParams(...),
      filters={"dataset": ["train_and_val"]}
    )

Alternatively, you can restrict the training/validation dataset by providing a :class:`slideflow.Dataset` to the ``dataset`` argument:

.. code-block:: python

    dataset = P.dataset(tile_px=299, tile_um=302)
    dataset = dataset.filter({"dataset": ["train_and_val"]})

    results = P.train(
      outcomes="tumor_type",
      params=sf.ModelParams(...),
      dataset=dataset
    )

In both cases, slides will be further split into training and validation sets using the specified validation settings (defaulting to three-fold cross-validation).

For more granular control over the validation dataset used, you can supply a :class:`slideflow.Dataset` to the ``val_dataset`` argument. Doing so will cause the rest of the validation keyword arguments to be ignored.

.. code-block:: python

    dataset = P.dataset(tile_px=299, tile_um=302)
    train_dataset = dataset.filter({"dataset": ["train"]})
    val_dataset = dataset.filter({"dataset": ["val"]})

    results = P.train(
      outcomes="tumor_type",
      params=sf.ModelParams(...),
      dataset=train_dataset
      val_dataset=val_dataset
    )

Performance metrics - including accuracy, loss, etc. - are returned as a dictionary and saved in ``results_log.csv`` in both the project directory and model directory. Additional data, including ROCs and scatter plots, are saved in the model directories. Pandas DataFrames containing tile-, slide-, and patient-level predictions are also saved in the model directory.

At each designated epoch, models are saved in their own folders. Each model directory will include a copy of its hyperparameters in a ``params.json`` file, and a copy of its training/validation slide manifest in ``slide.log``.

.. _training_with_trainer:

Using a Trainer
---------------

You can also train models outside the context of a project by using :class:`slideflow.model.Trainer`. This lower-level interface provides greater flexibility for customization and allows models to be trained without requiring a Project to be set up. It lacks several convenience features afforded by using :meth:`slideflow.Project.train`, however, such as cross-validation, logging, and label preparation for easy multi-outcome support.

For this training approach, start by building a trainer with :func:`slideflow.model.build_trainer`, which requires:

- ``hp``: :class:`slideflow.ModelParams` object.
- ``outdir``: Directory in which to save models and checkpoints.
- ``labels``: Dictionary mapping slide names to outcome labels.

:class:`slideflow.Dataset` provides a ``.labels()`` function that can generate this required labels dictionary.

.. code-block:: python

    # Prepare dataset and labels
    dataset = P.dataset(tile_px=299, tile_um=302)
    labels, unique_labels = dataset.labels('tumor_type')

    # Split into training/validation
    train_dataset = dataset.filter({"dataset": ["train"]})
    val_dataset = dataset.filter({"dataset": ["val"]})

    # Determine model parameters
    hp = sf.ModelParams(
        tile_px=299,
        tile_um=302,
        batch_size=32,
        ...
    )

    # Prepare a Trainer
    trainer = sf.model.build_trainer(
        hp=hp,
        outdir='path',
        labels=labels
    )

Use :meth:`slideflow.model.Trainer.train` to train a model using your specified training and validation datasets.

.. code-block:: python

    # Train a model
    trainer.train(train_dataset, val_dataset)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

      {
        "epochs": {
          "epoch3": {
            "train_metrics": {
              "loss": 0.497
              "accuracy": 0.806
              "val_loss": 0.719
              "val_accuracy": 0.778
            },
            "val_metrics": {
              "loss": 0.727
              "accuracy": 0.770
            },
            "tile": {
              "Outcome 0": [
                0.580
                0.580
              ]
            },
            "slide": {
              "Outcome 0": [
                0.658
                0.658
              ]
            },
            "patient": {
              "Outcome 0": [
                0.657
                0.657
              ]
            }
          }
        }
      }

Read more about the ``Trainer`` class and available keyword arguments in the :class:`API documentation <slideflow.model.Trainer>`.

Multiple outcomes
*****************

Slideflow supports both classification and regression, as well as training to single or multiple outcomes at once. To train with multiple outcomes simultaneously, simply pass multiple annotation headers to the ``outcomes`` argument of :meth:`slideflow.Project.train`.

Time-to-event / survival outcomes
*********************************

Models can also be trained to a time series outcome using Cox Proportional Hazards (CPH) and negative log likelihood loss. For time-to-event / survival models, use ``'negative_log_likelihood'`` loss and set ``outcomes`` equal to the annotation column indicating event *time*. Specify the event *type* (0 or 1) by passing the event type annotation column to the argument ``input_header``. If you are using multiple clinical inputs, the first header passed to ``input_header`` must be event type. Survival models are not compatible with multiple outcomes.

.. note::
    Survival models are currently only available with the Tensorflow backend. PyTorch support for survival outcomes is in development.

Multimodal models
*****************

In addition to training using image data, clinical data can also be provided as model input by passing annotation column headers to the variable ``input_header``. This input is concatenated at the post-convolutional layer, prior to any configured hidden layers.

If desired, models can also be trained with clinical input data alone, without images, by using the hyperparameter argument ``drop_images=True``.

.. _hyperparameter_optimization:

Hyperparameter optimization
***************************

Slideflow includes several tools for assisting with hyperparameter optimization, as described in the next sections.

Testing multiple combinations
-----------------------------

You can easily test a series of hyperparameter combinations by passing a list of ``ModelParams`` object to the ``params`` argument of :meth:`slideflow.Project.train`.

.. code-block:: python

    hp1 = sf.ModelParams(..., batch_size=32)
    hp2 = sf.ModelParams(..., batch_size=64)

    P.create_hp_sweep(
      ...,
      params=[hp1, hp2]
    )

Grid-search sweep
-----------------

You can also prepare a grid-search sweep, testing every permutation across a series of hyperparameter ranges. Use :meth:`slideflow.Project.create_hp_sweep`, which will calculate and save the sweep configuration to a JSON file. For example, the following would configure a sweep with only two combinations; the first with a learning rate of 0.01, and the second with a learning rate of 0.001:

.. code-block:: python

    P.create_hp_sweep(
      filename='sweep.json',
      model=['xception'],
      loss='sparse_categorical_crossentropy',
      learning_rate=[0.001, 0.0001],
      batch_size=64,
    )

The sweep is then executed by passing the JSON path to the ``params`` argument of :meth:`slideflow.Project.train()`:

.. code-block:: python

    P.train(params='sweep.json', ...)

.. _bayesian_optimization:

Bayesian optimization
---------------------

You can also perform Bayesian hyperparameter optimization using `SMAC3 <https://automl.github.io/SMAC3/master/>`_, which uses a `configuration space <https://automl.github.io/ConfigSpace/master/>`_ to determine the types and ranges of hyperparameters to search.

Slideflow provides several functions to assist with building these configuration spaces. :func:`slideflow.util.create_search_space` allows you to define a range to search for each hyperparameter via keyword arguments:

.. code-block:: python

    import slideflow as sf

    config_space = sf.util.create_search_space(
        normalizer=['macenko', 'reinhard', 'none'],
        dropout=(0.1, 0.5),
        learning_rate=(1e-4, 1e-5)
    )

:func:`slideflow.util.broad_search_space` and :func:`slideflow.util.shallow_search_space` provide preconfigured search spaces that will search a broad and narrow range of hyperparameters, respectively. You can also customize a preconfigured search space using keyword arguments. For example, to do a broad search but disable L1 searching:

.. code-block:: python

    import slideflow as sf

    config_space = sf.util.broad_search_space(l1=None)

See the linked API documentation for each function for more details about the respective search spaces.

Once the search space is determined, you can perform the hyperparameter optimization by simply replacing :meth:`slideflow.Project.train` with :meth:`slideflow.Project.smac_search`, providing the configuration space to the argument ``smac_configspace``. By default, SMAC3 will optimize the tile-level AUROC, but the optimization metric can be customized with the keyword argument ``smac_metric``.

.. code-block:: python

    # Base hyperparameters
    hp = sf.ModelParams(tile_px=299, ...)

    # Configuration space to optimize
    config_space = sf.util.shallow_search_space()

    # Run the Bayesian optimization
    best_config, history = P.smac_search(
        outcomes='tumor_type',
        params=hp,
        smac_configspace=cs,
        smac_metric='tile_auc',
        ...
    )
    print(history)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

            dropout        l1        l2    metric
        0  0.126269  0.306857  0.183902  0.271778
        1  0.315987  0.014661  0.413443  0.283289
        2  0.123149  0.311893  0.184439  0.250339
        3  0.250000  0.250000  0.250000  0.247641
        4  0.208070  0.018481  0.121243  0.257633

:meth:`slideflow.Project.smac_search` returns the best configuration and a history of models trained during the search. This history is a Pandas DataFrame with hyperparameters for columns, and a "metric" column with the optimization metric result for each trained model. The run history is also saved in CSV format in the associated model folder.

See the API documentation for available customization via keyword arguments.

.. _custom_loss:

Customizing model or loss
*************************

Slideflow supports dozens of model architectures, but you can also train with a custom architecture, as demonstrated in :ref:`tutorial3`.

Similarly, you can also train with a custom loss function by supplying a dictionary to the ``loss`` argument in ``ModelParams``, with the keys ``type`` (which must be either ``'classification'``, ``'regression'``, or ``'survival'``) and ``fn`` (a callable loss function).

For Tensorflow/Keras, the loss function must accept arguments ``y_true, y_pred``. For regression losses, ``y_true`` may need to be cast to ``tf.float32``. An example custom regression loss is given below:

.. code-block:: python

  # Custom Tensorflow loss
  def custom_regression_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


For PyTorch, the loss function must return a nested loss function with arguments ``output, target``. An example regression loss is given below:

.. code-block:: python

  # Custom PyTorch loss
  def custom_regression_loss():
    def loss_fn(output, target):
      return torch.mean((target - output) ** 2)
    return loss_fn


In both cases, the loss function is applied as follows:

.. code-block:: python

  hp = sf.ModelParams(..., loss={'type': 'regression', 'fn': custom_regression_loss})


Using multiple GPUs
*******************

Slideflow can perform distributed training if multiple GPUs are available. Enable distributed training by passing the argument ``multi_gpu=True``, which will allow Slideflow to use all available (and visible) GPUs.

.. _from_wsi:

Training without TFRecords
**************************

It is also possible to train deep learning models directly from slides, without first generating TFRecords. This may be advantageous for rapidly prototyping models on a large dataset, or when tuning the tile size for a dataset.

Use the argument ``from_wsi=True`` in either the :meth:`slideflow.Project.train` or :meth:`slideflow.model.Trainer.train` functions. Image tiles will be dynamically extracted from slides during training, and background will be automatically removed via Otsu's thresholding.

.. note::

    Using the :ref:`cuCIM backend <slide_backend>` will greatly improve performance when training without TFRecords.

Monitoring performance
**********************

Tensorboard
-----------

During training, progress can be monitored using Tensorflow's bundled ``Tensorboard`` package by passing the argument ``use_tensorboard=True``. This functionality was disabled by default due to a recent bug in Tensorflow. To use tensorboard to monitor training, execute:

.. code-block:: bash

    $ tensorboard --logdir=/path/to/model/directory

... and open http://localhost:6006 in your web browser.

Neptune.ai
----------

Experiments can be automatically logged with `Neptune.ai <https://app.neptune.ai>`_. To enable logging, first locate your Neptune API token and workspace ID, and configure the environmental variables ``NEPTUNE_API_TOKEN`` and ``NEPTUNE_WORKSPACE``.

With the environmental variables set, Neptune logs are enabled by passing ``use_neptune=True`` to ``sf.load_project``.

.. code-block:: python

    P = sf.load_project('/project/path', use_neptune=True)