Quickstart
==========

This section provides an example of using Slideflow to build a deep learning classifier from digital pathology slides. Follow the links in each section for more information.

Preparing a project
*******************

Slideflow experiments are organized using :class:`slideflow.Project`, which supervises storage of data, saved models, and results. The ``slideflow.project`` module has three preconfigured projects with associated slides and clinical annotations: ``LungAdenoSquam``, ``ThyroidBRS``, and ``BreastER``.

For this example, we will the ``LungAdenoSquam`` project to train a classifier to predict lung adenocarcinoma (Adeno) vs. squamous cell carcinoma (Squam).

.. code-block:: python

    import slideflow as sf

    # Download preconfigured project, with slides and annotations.
    project = sf.create_project(
        root='data',
        cfg=sf.project.LungAdenoSquam(),
        download=True
    )

Read more about :ref:`setting up a project on your own data <project_setup>`.

Data preparation
****************

The core imaging data used in Slideflow are image tiles :ref:`extracted from slides <filtering>` at a specific magnification and pixel resolution. Tile extraction and downstream image processing is handled through the primitive :ref:`slideflow.Dataset <datasets_and_validation>`. We can request a ``Dataset`` at a given tile size from our project using :meth:`slideflow.Project.dataset`. Tile magnification can be specified in microns (as an ``int``) or as optical magnification (e.g. ``'40x'``).

.. code-block:: python

    # Prepare a dataset of image tiles.
    dataset = project.dataset(
        tile_px=299,   # Tile size, in pixels.
        tile_um='10x'  # Tile size, in microns or magnification.
    )
    dataset.summary()

.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Overview:
    ╒===============================================╕
    │ Configuration file: │ /mnt/data/datasets.json │
    │ Tile size (px):     │ 299                     │
    │ Tile size (um):     │ 10x                     │
    │ Slides:             │ 941                     │
    │ Patients:           │ 941                     │
    │ Slides with ROIs:   │ 941                     │
    │ Patients with ROIs: │ 941                     │
    ╘===============================================╛

    Filters:
    ╒====================╕
    │ Filters:      │ {} │
    ├--------------------┤
    │ Filter Blank: │ [] │
    ├--------------------┤
    │ Min Tiles:    │ 0  │
    ╘====================╛

    Sources:

    TCGA_LUNG
    ╒==============================================╕
    │ slides    │ /mnt/raid/SLIDES/TCGA_LUNG       │
    │ roi       │ /mnt/raid/SLIDES/TCGA_LUNG       │
    │ tiles     │ /mnt/rocket/tiles/TCGA_LUNG      │
    │ tfrecords │ /mnt/rocket/tfrecords/TCGA_LUNG/ │
    │ label     │ 299px_10x                        │
    ╘==============================================╛

    Number of tiles in TFRecords: 0
    Annotation columns:
    Index(['patient', 'subtype', 'site', 'slide'],
        dtype='object')

Tile extraction
---------------

We prepare imaging data for training by extracting tiles from slides. Background areas of slides will be filtered out with Otsu's thresholding.

.. code-block:: python

    # Extract tiles from all slides in the dataset.
    dataset.extract_tiles(qc='otsu')

Read more about tile extraction and :ref:`slide processing in Slideflow <filtering>`.

Held-out test sets
------------------

Now that we have our dataset and we've completed the initial tile image processing, we'll split the dataset into a training cohort and a held-out test cohort with :meth:`slideflow.Dataset.split`. We'll split while balancing the outcome ``'subtype'`` equally in the training and test dataset, with 30% of the data retained in the held-out set.

.. code-block:: python

    # Split our dataset into a training and held-out test set.
    train_dataset, test_dataset = dataset.split(
        model_type='classification',
        labels='subtype',
        val_fraction=0.3
    )

Read more about :ref:`Dataset management <datasets_and_validation>`.

Configuring models
******************

Neural network models are prepared for training with :class:`slideflow.ModelParams`, through which we define the model architecture, loss, and hyperparameters. Dozens of architectures are available in both the Tensorflow and PyTorch backends, and both neural network :ref:`architectures <tutorial3>` and :ref:`loss <custom_loss>` functions can be customized. In this example, we will use the included Xception network.

.. code-block:: python

    # Prepare a model and hyperparameters.
    params = sf.ModelParams(
        tile_px=299,
        tile_um='10x',
        model='xception',
        batch_size=64,
        learning_rate=0.0001
    )

Read more about :ref:`hyperparameter optimization in Slideflow <training>`.

Training a model
****************

Models can be trained from these hyperparameter configurations using :meth:`Project.train`. Models can be trained to categorical, multi-categorical, continuous, or time-series outcomes, and the training process is :ref:`highly configurable <training>`. In this case, we are training a binary categorization model to predict the outcome ``'subtype'``, and we will distribute training across multiple GPUs.

By default, Slideflow will train/validate on the full dataset using k-fold cross-validation, but validation settings :ref:`can be customized <validation_planning>`. If you would like to restrict training to only a subset of your data - for example, to leave a held-out test set untouched - you can manually specify a dataset for training. In this case, we will train on ``train_dataset``, and allow Slideflow to further split this into training and validation using three-fold cross-validation.

.. code-block:: python

    # Train a model from a set of hyperparameters.
    results = P.train(
        'subtype',
        dataset=train_dataset,
        params=params,
        val_strategy='k-fold',
        val_k_fold=3,
        multi_gpu=True,
    )

Models and training results will be saved in the project ``models/`` folder.

Read more about :ref:`training a model <training>`.

Evaluating a trained model
**************************

After training, you can test model performance on a held-out test dataset with :meth:`Project.evaluate`, or generate predictions without evaluation (when ground-truth labels are not available) with :meth:`Project.predict`. As with :meth:`Project.train`, we can specify a :class:`slideflow.Dataset` to evaluate.

.. code-block:: python

    # Train a model from a set of hyperparameters.
    test_results = P.evaluate(
        model='/path/to/trained_model_epoch1'
        outcomes='subtype',
        dataset=test_dataset
    )

Read more about :ref:`model evaluation <evaluation>`.

Post-hoc analysis
*****************

Slideflow includes a number of analytical tools for working with trained models. Read more about :ref:`heatmaps <evaluation>`, :ref:`model explainability <stylegan>`, :ref:`analysis of layer activations <activations>`, and real-time inference in an interactive :ref:`whole-slide image reader <studio>`.