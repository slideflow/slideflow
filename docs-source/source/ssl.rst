.. currentmodule:: slideflow.simclr

.. _simclr_ssl:

Self-Supervised Learning (SSL)
==============================

Slideflow provides easy access to training the self-supervised, contrastive learning framework `SimCLR <https://arxiv.org/abs/2002.05709>`_. Self-supervised learning provides an avenue for learning useful visual representations in your dataset without requiring ground-truth labels. These visual representations can be exported as feature vectors and used for downstream analyses such as :ref:`dimensionality reduction <slidemap>` or :ref:`multi-instance learning <mil>`.

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
        model_type='classification',
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

.. _dinov2:

Training DINOv2
***************

A lightly modified version of `DINOv2 <https://arxiv.org/abs/2304.07193>`__ with Slideflow integration is available on `GitHub <https://github.com/jamesdolezal/dinov2>`_. This version facilitates training DINOv2 with Slideflow datasets and adds stain augmentation to the training pipeline.

To train DINOv2, first install the package:

.. code-block:: bash

    pip install git+https://github.com/jamesdolezal/dinov2.git

Next, configure the training parameters and datsets by providing a configuration YAML file. This configuration file should contain a ``slideflow`` section, which specifies the Slideflow project and dataset to use for training. An example YAML file is shown below:

.. code-block:: yaml

    train:
      dataset_path: slideflow
      batch_size_per_gpu: 32
      slideflow:
        project: "/mnt/data/projects/TCGA_THCA_BRAF"
        dataset:
          tile_px: 299
          tile_um: 302
          filters:
            brs_class:
            - "Braf-like"
            - "Ras-like"
        seed: 42
        outcome_labels: "brs_class"
        normalizer: "reinhard_mask"
        interleave_kwargs: null

See the `DINOv2 README <https://github.com/jamesdolezal/dinov2>`_ for more details on the configuration file format.

Finally, train DINOv2 using the same command-line interface as the original DINOv2 implementation. For example, to train DINOv2 on 4 GPUs on a single node:

.. code-block:: bash

    torchrun --nproc_per_node=4 -m "dinov2.train.train" \
        --config-file /path/to/config.yaml \
        --output-dir /path/to/output_dir

The teacher weights will be saved in ``outdir/eval/.../teacher_checkpoint.pth``, and the final configuration YAML will be saved in ``outdir/config.yaml``.

Generating features
*******************

Generating features from a trained SSL is straightforward - use the same :meth:`slideflow.Project.generate_features` and :class:`slideflow.DatasetFeatures` interfaces as :ref:`previously described <dataset_features>`, providing a path to a saved SimCLR model or checkpoint.

.. code-block:: python

    import slideflow as sf

    # Create the SimCLR feature extractor
    simclr = sf.build_feature_extractor(
        'simclr',
        ckpt='/path/to/simclr.ckpt'
    )

    # Calculate SimCLR features for a dataset
    features = P.generate_features(simclr, ...)

For DINOv2 models, use ``'dinov2'`` as the first argument, and pass the model configuration YAML file to ``cfg`` and the teacher checkpoint weights to ``weights``.

.. code-block:: python

    dinov2 = build_feature_extractor(
        'dinov2',
        weights='/path/to/teacher_checkpoint.pth',
        cfg='/path/to/config.yaml'
    )