.. currentmodule:: slideflow.gan

.. _stylegan:

Generative Networks (GANs)
==========================

.. video:: https://media.githubusercontent.com/media/slideflow/slideflow/master/docs/stylegan.webm
    :autoplay:

|

Slideflow includes tools to easily interface with the PyTorch implementations of `StyleGAN2 <https://github.com/jamesdolezal/stylegan2-slideflow>`_ and `StyleGAN3 <https://github.com/jamesdolezal/stylegan3-slideflow>`_, allowing you to train these Generative Adversarial Networks (GANs). Slideflow additionally includes tools to assist with image generation, interpolation between class labels, and interactively visualize GAN-generated images and their predictions. See our manuscript on the use of GANs to `generate synthetic histology <https://arxiv.org/abs/2211.06522>`_ for an example of how these networks might be used.


.. note::

    StyleGAN requires PyTorch <0.13 and Slideflow-NonCommercial, which can be installed with:

    .. code-block:: bash

        pip install slideflow-noncommercial


Training StyleGAN
*****************

The easiest way to train StyleGAN2/StyleGAN3 is with :meth:`slideflow.Project.gan_train`. Both standard and class-conditional GANs are
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

The trained networks will be saved in the ``gan/`` subfolder in the project directory.

StyleGAN2/3 can only be trained on images with sizes that are powers of 2. You can crop and/or resize images from a Dataset to match this requirement by using the ``crop`` and/or ``resize`` arguments:

.. code-block:: python

    dataset = P.dataset(tile_px=299, ...)

    # Train a GAN on images resized to 256x256
    P.gan_train(
      ...,
      resize=256,
    )

See the :meth:`slideflow.Project.gan_train` documentation for additional
keyword arguments to customize training.

Class conditioning
------------------

GANs can also be trained with class conditioning. To train a class-conditional GAN, simply provide a list of categorical
outcome labels to the ``outcomes`` argument of :meth:`slideflow.Project.gan_train`. For example, to train a GAN with class conditioning on ER status:

.. code-block:: python

    P.gan_train(
      ...,
      outcomes='er_status'
    )

Tile-level labels
-----------------

In addition to class conditioning with slide-level labels, StyleGAN2/StyleGAN3 can be trained with tile-level class conditioning. Tile-level labels can be generated through ROI annotations, as described in :ref:`tile_labels`.

Prepare a pandas dataframe, indexed with the format ``{slide}-{x}-{y}``, where ``slide`` is the name of the slide (without extension), ``x`` is the corresponding tile x-coordinate, and ``y`` is the tile y-coordinate. The dataframe should have a single column, ``label``, containing onehot-encoded category labels. For example:

.. code-block:: python

    import pandas as pd

    df = pd.DataFrame(
      index=[
        'slide1-251-425',
        'slide1-560-241',
        'slide1-321-502',
        ...
      ],
      data={
        'label': [
          [1, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          ...
        ]
      }
    )

This dataframe can be generated, as described in :ref:`tile_labels`, through the :meth:`slideflow.Dataset.get_tile_dataframe` function. For GAN conditioning, the ``label`` column should be onehot-encoded.

Once the dataframe is complete, save it in parquet format:

.. code-block:: python

    df.to_parquet('tile_labels.parquet')

And supply this file to the ``tile_labels`` argument of :meth:`slideflow.Project.gan_train`:

.. code-block:: python

    P.gan_train(
      ...,
      tile_labels='tile_labels.parquet'
    )

Generating images
*****************

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

Slideflow Studio can be used to interactively visualize GAN-generated images (see :ref:`studio`). Images can be directly exported from this interface. This tool also enables you to visualize real-time predictions for GAN generated images when as inputs to a trained classifier.

For more examples of using Slideflow to work with GAN-generated images, see `our GitHub repository <https://github.com/jamesdolezal/synthetic-histology>`_ for code accompanying the previously referenced manuscript.