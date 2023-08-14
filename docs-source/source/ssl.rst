.. currentmodule:: slideflow.simclr

.. _simclr_ssl:

Self-Supervised Learning (SSL)
==============================

Slideflow provides easy access to training the self-supervised, contrastive learning framework `SimCLR <https://arxiv.org/abs/2002.05709>`_. Self-supervised learning provides an avenue for learning useful visual representations in your dataset without requiring ground-truth labels. These visual representations can be exported as feature vectors and used for downstream analyses such as :ref:`dimensionality reduction <slidemap>` or :ref:`multi-instance learning <clam_mil>`.

The ``slideflow.simclr`` module contains a `forked Tensorflow implementation <https://github.com/jamesdolezal/simclr/>`_ minimally modified to interface with Slideflow. SimCLR models can be trained with :meth:`slideflow.Project.train_simclr`, and SimCLR features can be calculated as with other models using :meth:`slideflow.Project.generate_features`.

Training SimCLR
***************

First, determine the SimCLR training parameters with :func:`slideflow.simclr.get_args`. This function accepts parameters via keyword arguments, such as ``learning_rate`` and ``temperature``, and returns a configured :class:`slideflow.simclr.SimCLR_Args`.

.. code-block:: python

    from slideflow import simclr

    args = simclr.get_args(
        temperature=0.1,
        learning_rate=0.3,
        train_epochs=100,
        image_size=299
    )

Next, assemble a training and (optionally) a validation dataset. The validation dataset is used to assess contrastive loss during training, but is not required.

.. code-block:: python

    import slideflow as sf

    # Load a project and dataset
    P = sf.load_project('path')
    dataset = P.dataset(tile_px=299, tile_um=302)

    # Split dataset into training/validation
    train_dts, val_dts = dataset.split(
        val_fraction=0.3,
        model_type='categorical',
        labels='subtype')

Finally, SimCLR can be trained with :meth:`slideflow.Project.train_simclr`. You can train with a single dataset:

.. code-block:: python

    P.train_simclr(args, dataset)

You can train with an optional validation dataset:

.. code-block:: python

    P.train_simclr(
        args,
        train_dataset=train_dts,
        val_dataset=val_dts
    )

And you can also optionally provide labels for training the supervised head. To train a supervised head, you'll also need to set the SimCLR argument ``lineareval_while_pretraining=True``.

.. code-block:: python

    # SimCLR args
    args = simclr.get_args(
        ...,
        lineareval_while_pretraining=True
    )

    # Train with validation & supervised head
    P.train_simclr(
        args,
        train_dataset=train_dts,
        val_dataset=val_dts,
        outcomes='subtype'
    )

The SimCLR model checkpoints and final saved model will be saved in the ``simclr/`` folder within the project root directory.

Generating features
*******************

Generating SimCLR features is straightforward - use the same :meth:`slideflow.Project.generate_features` and :class:`slideflow.DatasetFeatures` interfaces as :ref:`previously described <dataset_features>`, providing a path to a saved SimCLR model or checkpoint.

.. code-block:: python

    from slideflow.model import build_feature_extractor

    # Create the SimCLR feature extractor
    simclr = build_feature_extractor(
        'simclr',
        ckpt='/path/to/simclr.ckpt'
    )

    # Calculate SimCLR features for a dataset
    features = P.generate_features(simclr, ...)
