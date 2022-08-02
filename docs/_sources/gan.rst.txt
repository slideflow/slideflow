.. currentmodule:: slideflow.gan

slideflow.gan
=============

.. automodule:: slideflow.gan
   :members:

Training StyleGAN2
------------------

The easiest way to train GANs is with :meth:`slideflow.Project.gan_train`, which
supervises training of StyleGAN2. Both standard and class-conditional GANs are
supported. To train a GAN, pass a :class:`slideflow.Dataset`, experiment label,
and StyleGAN2 keyword arguments to this function:

.. code-block:: python

    import slideflow as sf

    P = sf.Project('/project/path')
    dataset = P.dataset(tile_px=512, tile_um=400)

    P.gan_train(
      dataset=dataset,
      exp_label="ExperimentLabel",
      gpus=4,
      batch_size=32,
      ...
    )

To train this as a class-conditional GAN, simply provide a list of categorical
outcome labels to the ``outcomes`` argument:

.. code-block:: python

    P.gan_train(
      ...,
      outcomes='er_status'
    )

The trained networks will be saved in the ``gan/`` subfolder in the project directory.

See the :meth:`slideflow.Project.gan_train` documentation for additional
keyword arguments to customize training.

Generating Images
-----------------

Images can be generated from a trained GAN and exported either as loose images
in PNG or JPG format, or alternatively stored in TFRecords. Images are generated from a list
of seeds (list of int). Use the :meth:`slideflow.Project.gan_generate` function
to generate images, with ``out`` set to a directory path if exporting loose images,
or ``out`` set to a filename ending in ``.tfrecords`` if saving images in
TFRecord format:

.. code-block:: python

    network_pkl = '/path/to/trained/gan.pkl'
    P.gan_generate(
      network_pkl,
      out='target.tfrecords',
      seeds=range(100),
      ...
    )

The image format is set with the ``format`` argument:

.. code-block:: python

    P.gan_generate(
      ...,
      format='jpg',
    )

Class index (for class-conditional GANs) is set with ``class_idx``:

.. code-block:: python

    P.gan_generate(
      ...,
      class_idx=1,
    )

Finally, images can be resized after generation to match a target tile size:

.. code-block:: python

    P.gan_generate(
      ...,
      gan_px=512,
      gan_um=400,
      target_px=299,
      target_um=302,
    )

StyleGAN2 Interpolator
----------------------

.. autoclass:: StyleGAN2Interpolator
    :inherited-members:

Utility functions
-----------------

.. automodule:: slideflow.gan.utils
   :members: