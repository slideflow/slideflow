Custom Training Loops
=====================

To use ``*.tfrecords`` from extracted tiles in a custom training loop or entirely separate architecture (such as `StyleGAN2 <https://github.com/jamesdolezal/stylegan2-slideflow>`_ or `YoloV5 <https://github.com/ultralytics/yolov5>`_), Tensorflow ``tf.data.Dataset`` or PyTorch ``torch.utils.data.DataLoader`` objects can be created for easily serving processed images to your custom trainer.

TFRecord DataLoader
*******************

The :class:`slideflow.Dataset` class includes functions to prepare a Tensorflow ``tf.data.Dataset`` or PyTorch ``torch.utils.data.DataLoader`` object to interleave and process images from stored TFRecords. First, create a ``Dataset`` object at a given tile size:

.. code-block:: python

    from slideflow import Project

    P = Project('/project/path', ...)
    dts = P.dataset(tile_px=299, tile_um=302)

If you want to perform any mini-batch balancing, use the ``.balance()`` method:

.. code-block:: python

    dts = dts.balance('HPV_status', strategy='category')

Other dataset options can also be applied at this step. For example, to clip the maximum number of tiles to take from a slide, use the ``.clip()`` method:

.. code-block:: python

    dts = dts.clip(500)

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

or the :meth:`slideflow.Dataset.tensorflow` method to create a ``tf.data.Dataset``:

.. code-block:: python

    dataloader = dts.tensorflow(
        labels       = ...       # Your outcome label
        batch_size   = 64,       # Batch size
        infinite     = True,     # True for training, False for validation
        augment      = True,     # Flip/rotate/compression augmentation
        standardize  = True,     # Standardize images
    )

The returned dataloaders can then be used directly with your external applications. Read more about :ref:`creating and using dataloaders <dataloaders>`.