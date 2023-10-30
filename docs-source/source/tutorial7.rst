.. _tutorial7:

Tutorial 7: Training with custom augmentations
==============================================

In this tutorial, we'll take a look at how you can use custom image augmentations when training a model with Slideflow. This tutorial builds off of :ref:`tutorial2`, so if you haven't already, you should read that tutorial first.

Our goal will be to train a model on a sparse outcome, such as ER status (roughly 4:1 positive:negative), with a custom augmentation that will oversample the minority class.  This tutorial will use PyTorch, but the same principles apply when using Tensorflow.

.. code-block:: python

    >>> import os
    >>> os.environ['SF_BACKEND'] = 'torch'

First, we'll start by loading a project and preparing our datasets, just like in :ref:`tutorial2`:

.. code-block:: python

    >>> import slideflow as sf
    >>> P = sf.load_project('/home/er_project')
    >>> full_dataset = P.dataset(
    ...   tile_px=256,
    ...   tile_um=128,
    ...   filters={
    ...     'er_status_by_ihc': ['Positive', 'Negative']
    ... })
    >>> labels, _ = full_dataset.labels('er_status_by_ihc')
    >>> train, val = full_dataset.split(
    ...   labels='er_status_by_ihc',
    ...   val_strategy='k-fold',
    ...   val_k_fold=3,
    ...   k_fold_iter=1
    ... )

If tiles have not yet been extracted from slides, do that now.

.. code-block:: python

    >>> dataset.extract_tiles(qc='otsu')

By default, Slideflow will equally sample from all slides / TFRecords during training, resulting in oversampling of slides with fewer tiles. In this case, we want to oversample the minority class (ER negative), so we'll use category-level balancing. Sampling strategies are discussed in detail in the :ref:`Developer Notes <balancing>`.

.. code-block:: python

    >>> train = train.balance('er_status_by_ihc', strategy='category')

Next, we'll set up our model hyperparameters, using the same parameters as in :ref:`tutorial2`. We still want to use Slideflow's default augmentation (random flip/rotation and JPEG compression), so we'll use the hyperparameter ``augment=True``. Our custom augmentation will be applied after the default augmentation.

.. code-block:: python

    >>> hp = sf.ModelParams(
    ...   tile_px=256,
    ...   tile_um=128,
    ...   model='xception',
    ...   batch_size=32,
    ...   epochs=[3],
    ...   augment=True
    ... )

Now, we'll define our custom augmentation. Augmentations are functions that take a single Tensor (:class:`tf.Tensor` or :class:`torch.Tensor`) as input and return a single Tensor as output. Our training augmentation will include a random color jitter, random gaussian blur, and random auto-contrast.

.. code-block:: python

    >>> import torch
    >>> from torchvision import transforms
    >>> augment = transforms.Compose([
    ...     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ...     transforms.RandomAutocontrast(),
    ...     transforms.GaussianBlur(3)
    ... ])

Transformations can be applied to training or validation data by passing a dictionary - with the keys 'train' and/or 'val' - to the ``transform`` argument of :class:`slideflow.Trainer`. If a transformation should be applied to both training and validation, it can be passed directly to the ``transform`` argument. In this case, we'll apply our custom augmentation to the training dataset only.

.. code-block:: python

    >>> trainer = sf.model.build_trainer(
    ...   hp=hp,
    ...   outdir='/some/directory',
    ...   labels=labels,
    ...   transform={'train': augment},
    ... )

Now we can start training. Pass the training and validation datasets to the :meth:`slideflow.model.Trainer.train` method of our trainer, assigning the output to a new variable ``results``.

.. code-block:: python

    >>> results = trainer.train(train, val)

And that's it! You've trained a model with a custom augmentation. You can now use the model to make predictions on new data, or use the model to make predictions on the validation dataset.