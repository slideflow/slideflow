.. _dataloaders:

Dataloaders: Sampling and Augmentation
======================================

With support for both Tensorflow and PyTorch, Slideflow provides several options for dataset sampling, processing, and augmentation. Here, we'll review the options for creating dataloaders - objects that read and process TFRecord data and return images and labels - in each framework. In all cases, data are read from TFRecords generated through :ref:`filtering`. The TFRecord data format is discussed in more detail in the :ref:`tfrecords` note.

Tensorflow
**********

.. |TFRecordDataset| replace:: ``tf.data.TFRecordDataset``
.. _TFRecordDataset: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset

The :meth:`slideflow.Dataset.tensorflow()` method provides an easy interface for creating a ``tf.data.Dataset`` that reads and interleaves from tfrecords in a Slideflow dataset. Behind the scenes, this method uses the |TFRecordDataset|_ class for reading and parsing each TFRecord.

The returned ``tf.data.Dataset`` object is an iterable-only dataset whose returned values depend on the arguments provided to the ``.tensorflow()`` function.

If no arguments are provided, the returned dataset will yield a tuple of ``(image, None)``, where the image is a ``tf.Tensor`` of shape ``[tile_height, tile_width, num_channels]`` and type ``tf.uint8``.

If the ``labels`` argument is provided (dictionary mapping slide names to a numeric label), the returned dataset will yield a tuple of ``(image, label)``, where the label is a ``tf.Tensor`` with a shape and type that matches the provided labels.

.. code-block:: python

    import slideflow as sf

    # Create a dataset object
    project = sf.load_project(...)
    dataset = project.dataset(...)

    # Get the labels
    labels, unique_labels = dataset.labels('HPV_status')

    # Create a tensorflow dataset
    # that yields (image, label) tuples
    tf_dataset = dataset.tensorflow(labels=labels)

    for image, label in tf_dataset:
        # Do something with the image and label...
        ...

Slide names and tile locations
------------------------------

Dataloaders can be configured to return slide names and tile locations in addition to the image and label. This is done by providing the ``incl_slidenames`` and ``incl_loc`` arguments to the ``.tensorflow()`` method. Both arguments are boolean values and default to ``False``.

Setting ``incl_slidenames=True`` will return the slidename as a Tensor (dtype=string) after the label. Setting ``incl_loc=True`` will return the x and y locations, both as Tensors (dtype=int64), as the last two values of the tuple.

.. code-block:: python

    tf_dataset = dataset.tensorflow(incl_slidenames=True, incl_loc=True)

    for image, label, slide, loc_x, loc_y in tf_dataset:
        ...

Image preprocessing
-------------------

.. |per_image_standardization| replace:: ``tf.image.per_image_standardization()``
.. _per_image_standardization: https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

Dataloaders created with ``.tensorflow()`` include several image preprocessing options. These options are provided as keyword arguments to the ``.tensorflow()`` method and are executed in the order listed below:

- **crop_left** (int): Crop images to this top-left x/y coordinate. Default is ``None``.
- **crop_width** (int): Crop images to this width. Default is ``None``.
- **resize_target** (int): Resize images to this width/height. Default is ``None``.
- **resize_method** (str): Resize method. Default is ``"lanczos3"``.
- **resize_aa** (bool): Enable antialiasing if resizing. Defaults to ``True``.
- **normalizer** (``StainNormalizer``): Perform stain normalization.
- **augment** (str): Perform augmentations based on the provided string. Combine characters to perform multiple augmentations (e.g. ``'xyrj'``). Options include:
    - ``'n'``: Perform :ref:`stain_augmentation` (done concurrently with stain normalization)
    - ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
    - ``'r'``: Random 90-degree rotation
    - ``'x'``: Random horizontal flip
    - ``'y'``: Random vertical flip
    - ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
- **transform** (Any): Arbitrary function to apply to each image. The function must accept a single argument (the image) and return a single value (the transformed image).
- **standardize** (bool): Standardize images with |per_image_standardization|_, returning a ``tf.float32`` image. Default is ``False``, returning a ``tf.uint8`` image.

Dataset sharding
----------------

Tensorflow dataloaders can be sharded into multiple partitions, ensuring that data is not duplicated when performing distributed training across multiple processes or nodes. This is done by providing the ``shard_idx`` and ``num_shards`` arguments to the ``.tensorflow()`` method. The ``shard_idx`` argument is an integer specifying the shard number, and ``num_shards`` is an integer specifying the total number of shards.

.. code-block:: python

    # Shard the dataset for GPU 1 of 4
    tf_dataset = dataset.tensorflow(
        ...,
        shard_idx=0,
        num_shards=4
    )

PyTorch
*******

.. |DataLoader| replace:: ``torch.utils.data.DataLoader``
.. _DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

As with Tensorflow, the :meth:`slideflow.Dataset.torch()` method creates a |DataLoader|_ that reads images from TFRecords. In the backend, TFRecords are read using :func:`slideflow.tfrecord.torch.MultiTFRecordDataset` and processed as described in :ref:`tfrecords`.

The returned |DataLoader|_ is an iterable-only dataloader whose returned values depend on the arguments provided to the ``.torch()`` function. An indexable, map-style dataset is also available when using PyTorch, as described in :ref:`indexable_dataloader`.

If no arguments are provided, the returned dataloader will yield a tuple of ``(image, None)``, where the image is a ``torch.Tensor`` of shape ``[num_channels, tile_height, tile_width]`` and type ``torch.uint8``. Labels are assigned as described above. Slide names and tile location can also be returned, using the same arguments as `described above <https://slideflow.dev/dataloaders/#slide-names-and-tile-locations>`_.


.. code-block:: python

    import slideflow as sf

    # Create a dataset object
    project = sf.load_project(...)
    dataset = project.dataset(...)

    # Create a tensorflow dataset
    torch_dl = dataset.torch()

    for image, label in torch_dl:
        # Do something with the image...
        ...

Image preprocessing
-------------------

Dataloaders created with ``.torch()`` include several image preprocessing options, provided as keyword arguments to the ``.torch()`` method. These preprocessing steps are executed in the order listed below:

- **normalizer** (``StainNormalizer``): Perform stain normalization.
- **augment** (str): Perform augmentations based on the provided string. Combine characters to perform multiple augmentations (e.g. ``'xyrj'``). Augmentations are executed in the order characters appear in the string. Options include:
    - ``'n'``: Perform :ref:`stain_augmentation` (done concurrently with stain normalization)
    - ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
    - ``'r'``: Random 90-degree rotation
    - ``'x'``: Random horizontal flip
    - ``'y'``: Random vertical flip
    - ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
- **transform** (Any): Arbitrary function to apply to each image, including `torchvision transforms <https://pytorch.org/vision/main/transforms.html>`_. The function must accept a single argument (the image, in ``(num_channels, height, width)`` format) and return a single value (the transformed image).
- **standardize** (bool): Standardize images with ``image / 127.5 - 1``, returning a ``torch.float32`` image. Default is ``False``, returning a ``torch.uint8`` image.

Below is an example of using the ``transform`` argument to apply a torchvision transform to each image:

.. code-block:: python

    import torchvision.transforms as T

    # Create a torch dataloader
    torch_dataloader = dataset.torch(
        transform=T.Compose([
            RandomResizedCrop(size=(224, 224), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
    )

    for image, label in torch_dataloader:
        # Do something with the image and label...
        ...

Dataset sharding
----------------

PyTorch Dataloaders can similarly be sharded into multiple partitions, ensuring that data is not duplicated when performing distributed training across multiple process or nodes.

Sharding is done in two stages. First, dataloaders can be split into partitions using the ``rank`` and ``num_replicas`` arguments to the ``.torch()`` method. The ``rank`` argument is an integer specifying the rank of the current process, and ``num_replicas`` is an integer specifying the total number of processes.

.. code-block:: python

    # Shard the dataset for GPU 1 of 4
    torch_dataloader = dataset.torch(
        ...,
        rank=0,
        num_replicas=4
    )

The second stage of sharding happens in the background: if a dataloader is built with multiple worker processes (``Dataset.torch(num_workers=...)``), partitions will be automatically further subdivided into smaller chunks, ensuring that each worker process reads a unique subset of the data.

Labeling
********

The ``label`` argument to the ``.tensorflow()`` and ``.torch()`` methods accept a dictionary mapping slide names to a numeric label. During TFRecord reading, the slide name is used to lookup the label from the provided dictionary.

.. warning::

    Labels are assigned to image tiles based on the slide names inside a :ref:`tfrecord <tfrecords>` file, not by the filename of the tfrecord. This means that renaming a TFRecord file will not change the label of the tiles inside the file. If you need to change the slide names associated with tiles inside a TFRecord, the TFRecord file must be regenerated.

The most common way to generate labels is to use the :meth:`slideflow.Dataset.labels()` method, which returns a dictionary mapping slide names to numeric labels. For categorical labels, the numeric labels correspond to the index of the label in the ``unique_labels`` list. For example, if the ``unique_labels`` list is ``['HPV-', 'HPV+']``, then the mapping of numeric labels would be ``{ 'HPV-': 0, 'HPV+': 1 }``.

.. code-block:: python

    >>> labels, unique_labels = dataset.labels('HPV_status')
    >>> unique_labels
    ['HPV-', 'HPV+']
    >>> labels
    {'slide1': 0,
     'slide2': 1,
     ...
    }
    >>> tf_dataset = dataset.tensorflow(labels=labels)

.. _sampling:

Sampling
********

Dataloaders created with ``.tensorflow()`` and ``.torch()`` are iterable-only dataloaders, meaning that they cannot be indexed directly. This is because the underlying TFRecords are sampled in a streaming fashion, and the dataloader does not know what the next record will be until it has been read. This is in contrast to the :ref:`indexable_dataloader` method described below, which creates an indexable, map-style dataset.

Dataloaders created with ``.tensorflow()`` and ``.torch()`` can be configured to sample from TFRecords in several ways, with options for infinite vs. finite sampling, oversampling, and undersampling. These sampling methods are described below.

Infinite dataloaders
--------------------

By default, dataloaders created with ``.tensorflow()`` and ``.torch()`` will sample from TFRecords in an infinite loop. This is useful for training, where the dataloader should continue to yield images until the training process is complete. By default, images are sampled from TFRecords with uniform sampling, meaning that each TFRecord has an equal chance of yielding an image. This sampling strategy can be configured, as described below.

.. note::

    When training :ref:`tile-based models <training>`, a dataloader is considered to have yielded one "epoch" of data when it has yielded the number of images equal to the number of tiles in the dataset. Due to the random sampling from TFRecords, this means that some images will be overrepresented (images from TFRecords with fewer tiles) and some will be underrepresented (images from TFRecords with many tiles).

Finite dataloaders
------------------

Dataloaders can also be configured with finite sampling, yielding tiles from TFRecords exactly once. This is accomplished by passing the argument ``infinite=False`` to the ``.tensorflow()`` or ``.torch()`` methods.

.. _balancing:

Oversampling with balancing
---------------------------

Oversampling methods control the probability that tiles are read from each TFRecord, affecting the balance of data across slides, patients, and outcome categories. Oversampling is configured at the Dataset level, using the :meth:`slideflow.Dataset.balance` method. This method returns a copy of the dataset with the specified oversampling strategy.

**Slide-level balancing**: By default, images are sampled from TFRecords with uniform probability, meaning that each TFRecord has an equal chance of yielding an image. This is equivalent to both ``.balance(strategy='slide')`` and ``.balance(strategy=None)``. This strategy will oversample images from slides with fewer tiles, and undersample images from slides with more tiles.

.. code-block:: python

    # Sample from TFRecords with equal probability
    dataset = dataset.balance(strategy='slide')

**Patient-level balancing**: To sample from TFRecords with probability proportional to the number of tiles in each patient, use ``.balance(strategy='patient')``. This strategy will oversample images from patients with fewer tiles, and undersample images from patients with more tiles.

.. code-block:: python

    # Sample from TFRecords with probability proportional
    # to the number of tiles in each patient.
    dataset = dataset.balance(strategy='patient')

**Tile-level balancing**: To sample from TFRecords with uniform probability across image tiles, use ``.balance(strategy='tile')``. This strategy will sample from TFRecords with probability proportional to the number of tiles in the TFRecord, resulting in higher representation of slides with more tiles.

.. code-block:: python

    # Sample from TFRecords with probability proportional
    # to the number of tiles in each TFRecord.
    dataset = dataset.balance(strategy='tile')

**Category-level balancing**: To sample from TFRecords with probability proportional to the number of tiles in each outcome category, use ``.balance(strategy='category')``. This strategy will oversample images from outcome categories with fewer tiles, and undersample images from outcome categories with more tiles. This strategy will also perform slide-level balancing within each category. Category-level balancing is only available when using categorical labels.

.. code-block:: python

    # Sample from TFRecords with probability proportional
    # to the number of tiles in each category
    # "HPV-" and "HPV+".
    dataset = dataset.balance("HPV_status", strategy='category')

**Custom balancing**: The ``.balance()`` method saves sampling probability weights to ``Dataset.prob_weights``, a dictionary mapping TFRecord paths to sampling weights. Custom balancing can be performed by overriding this dictionary with custom weights.

.. code-block:: python

    >>> dataset = dataset.balance(strategy='slide')
    >>> dataset.prob_weights
    {'/path/to/tfrecord1': 0.002,
     '/path/to/tfrecord2': 0.003,
     ...
    }
    >>> dataset.prob_weights = {...}

Balancing is automatically applied to dataloaders created with the ``.tensorflow()`` and ``.torch()`` methods.

Undersampling with clipping
---------------------------

Datasets can also be configured to undersample TFRecords using :meth:`slideflow.Dataset.clip`. Several undersampling strategies are available.

**Slide-level clipping**: TFRecords can be clipped to a maximum number of tiles per slide using ``.clip(max_tiles)``. This strategy will clip TFRecords with more tiles than the specified ``max_tiles`` value, resulting in a maximum of ``max_tiles`` tiles per slide.

**Patient-level clipping**: TFRecords can be clipped to a maximum number of tiles per patient using ``.clip(max_tiles, strategy='patient')``. For patients with more than one slide/TFRecord, TFRecords will be clipped proportionally.

**Outcome-level clipping**: TFRecords can also be clipped to a maximum number of tiles per outcome category using ``.clip(max_tiles, strategy='category', headers=...)``. The outcome category is specified by the ``headers`` argument, which can be a single header name or a list of header names. Within each category, TFRecords will be clipped proportionally.

**Custom clipping**: The ``.clip()`` method saves clipping values to ``Dataset._clip``, a dictionary mapping TFRecord paths to counts of how many tiles should be sampled from the TFRecord. Custom clipping can be performed by overriding this dictionary with custom weights.

.. code-block:: python

    >>> dataset = dataset.clip(100)
    >>> dataset._clip
    {'/path/to/tfrecord1': 76,
     '/path/to/tfrecord2': 100,
     ...
    }
    >>> dataset._clip = {...}

Undersampling via dataset clipping is automatically applied to dataloaders created with ``.tensorflow()`` and ``.torch()``.

During training
---------------

If you are training a Slideflow model by directly providing a training and validation dataset to the :meth:`slideflow.Project.train` method, you can configure the datasets to perform oversampling and undersampling as described above. For example:

.. code-block:: python

    import slideflow as sf

    # Load a project
    project = sf.load_project(...)

    # Configure a training dataset with tile-level balancing
    # and clipping to max 100 tiles per TFRecord
    train = project.dataset(...).balance(strategy='tile').clip(100)

    # Get a validation dataset
    val = project.dataset(...)

    # Train a model
    project.train(
        ...,
        dataset=train,
        val_dataset=val,
    )

Alternatively, you can configure oversampling during training through the ``training_balance`` and ``validation_balance`` hyperparameters, as described in the :ref:`ModelParams <model_params>` documentation. Undersampling with dataset clipping can be performed with the ``max_tiles`` argument. Configuring oversampling/undersampling with this method propagates the configuration to all datasets generated during cross-validation.

.. code-block:: python

    import slideflow as sf

    # Load a project
    project = sf.load_project(...)

    # Configure hyperparameters with tile-level
    # balancing/oversampling for the training data
    hp = sf.ModelParams(
        ...,
        training_balance='tile',
        validation_balance=None,
    )

    # Train a model.
    # Undersample/clip data to max 100 tiles per TFRecord.
    project.train(
        ...,
        params=hp,
        max_tiles=100
    )


.. _indexable_dataloader:

Direct indexing
***************

An indexable, map-style dataloader can be created for PyTorch using :class:`slideflow.io.torch.IndexedInterleaver`, which returns a ``torch.utils.data.Dataset``. Indexable datasets are only available for the PyTorch backend.

This indexable dataset is created from a list of TFRecords and accepts many arguments for controlling labels, augmentation and image transformations.

.. code-block:: python

    from slideflow.io.torch import IndexedInterleaver

    # Create a dataset object
    project = sf.load_project(...)
    dataset = project.dataset(...)

    # Get the TFRecords
    tfrecords = dataset.tfrecords()

    # Assemble labels
    labels, _ = dataset.labels("HPV_status")

    # Create an indexable dataset
    dts = IndexedInterleaver(
        tfrecords,
        labels=labels,
        augment="xyrj",
        transform=T.Compose([
            T.RandomResizedCrop(size=(224, 224),
                                antialias=True),
        ]),
        normalizer=None,
        standardize=True,
        shuffle=True,
        seed=42,
    )

The returned dataset is indexable, meaning that it can be indexed directly to retrieve a single image and label.

.. code-block:: python

    >>> len(dts)
    284114
    >>> image, label = dts[0]
    >>> image.shape
    torch.Size([3, 224, 224])
    >>> image.dtype
    torch.float32

The dataset can be configured to return slide names and tile locations by setting the ``incl_slidenames`` and ``incl_loc`` arguments to ``True``, as described above.

Dataset sharding is supported with the same ``rank`` and ``num_replicas`` arguments as described above.

.. code-block:: python

    # Shard for GPU 1 of 4
    dts = IndexedInterleaver(
        ...,
        rank=0,
        num_replicas=4
    )

:class:`slideflow.io.IndexedInterleaver` supports undersampling via the `clip` argument (array of clipping values for each TFRecord), but does not support oversampling or balancing.

.. code-block:: python

    # Specify TFRecord clipping values
    dts = IndexedInterleaver(
        tfrecords=...,
        clip=[100, 75, ...], # Same length as tfrecords
        ...
    )

A |DataLoader|_ can then be created from the indexable dataset using the ``torch.utils.data.DataLoader`` class, as described in the PyTorch documentation.

.. code-block:: python

    from torch.utils.data import DataLoader

    # Create a dataloader
    dl = DataLoader(
        dts,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for image, label in dl:
        # Do something with the image and label...
        ...
