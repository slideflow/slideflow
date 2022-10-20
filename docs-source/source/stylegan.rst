.. currentmodule:: slideflow.gan

.. _stylegan:

StyleGAN
========

.. video:: stylemix.webm
    :width: 100%
    :autoplay:

|

Slideflow includes tools to easily interface with the PyTorch implementations of `StyleGAN2 <https://github.com/jamesdolezal/stylegan2-slideflow>`_ and `StyleGAN3 <https://github.com/jamesdolezal/stylegan3-slideflow>`_, allowing you to train networks, generate images, interpolate between class labels, and interactively visualize GAN-generated images and their predictions.

Training StyleGAN
------------------

The easiest way to train StyleGAN is by using :meth:`slideflow.Project.gan_train`, which
supervises training of StyleGAN2/3. Both standard and class-conditional GANs are
supported. To train a GAN, pass a :class:`slideflow.Dataset`, experiment label,
and StyleGAN keyword arguments to this function:

.. code-block:: python

    import slideflow as sf

    P = sf.Project('/project/path')
    dataset = P.dataset(tile_px=512, tile_um=400)

    P.gan_train(
      dataset=dataset,
      model='stylegan3',
      cfg='stylegan3-r',
      exp_label="ExperimentLabel",
      gpus=4,
      batch=32,
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

Generating images
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

Interactive visualization
-------------------------

Workbench can be used to interactively visualize GAN-generated images (see :ref:`workbench`). Images can be directly exported from this interface. This tool also enables you to visualize real-time predictions for GAN generated images when as inputs to a trained classifier.
