Switching backends
==================

The default backend for this package is Tensorflow/Keras, but a full PyTorch backend is also included, with a dedicated TFRecord reader/writer that ensures saved image tiles can be served to both Tensorflow and PyTorch models in cross-compatible fashion.

If using the Tensorflow backend, PyTorch does not need to be installed; the reverse is true as well.

To switch backends, simply set the environmental variable ``SF_BACKEND`` equal to either ``torch`` or ``tensorflow``:

.. code-block:: console

    export SF_BACKEND=torch


TFRecord DataLoader
*******************

In addition to using the built-in training tools, you can use tiles that have been extracted with Slideflow with completely external projects. The :class:`slideflow.Dataset` class includes both :func:`torch` and :func:`tensorflow` functions to prepare a DataLoader or Tensorflow tf.data.Dataset instance that interleaves and processs images from stored TFRecords.

.. code-block:: python

    from slideflow import Project

    P = Project('/project/path', ...)
    dts = P.dataset(tile_px=299, tile_um=302, filters=None)

If you want to perform any balancing, use the :meth:`slideflow.Datset.balance` method:

.. code-block:: python

    dts = dts.balance('HPV_status', strategy='category')

Finally, use the :meth:`slideflow.Dataset.torch` method to create a DataLoader object:

.. code-block:: python

    dataloader = dts.torch(
        labels       = ...       # Your outcome label
        batch_size   = 64,       # Batch size
        num_workers  = 6,        # Number of workers reading tfrecords
        infinite     = True,     # True for training, False for validation
        augment      = True,     # Flip/rotate/compression augmentation
        standardize  = True,     # Standardize images: mean 0, variance of 1
        pin_memory   = False,    # Pin memory to GPUs
    )

The returned dataloader can then be used directly with your external PyTorch applications.