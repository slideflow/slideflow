.. currentmodule:: slideflow.norm

slideflow.norm
===============

The ``slideflow.norm`` submodule includes tools for H&E stain normalization and augmentation.

Available stain normalization algorithms include:

- **macenko**: `Original Macenko paper <https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf>`_.
- **macenko_fast**: Modified Macenko algorithm with the brightness standardization step removed.
- **reinhard**: `Original Reinhard paper <https://ieeexplore.ieee.org/document/946629>`_.
- **reinhard_fast**: Modified Reinhard algorithm with the brightness standardization step removed.
- **reinhard_mask**: Modified Reinhard algorithm, with background/whitespace removed.
- **reinhard_fast_mask**: Modified Reinhard-Fast algorithm, with background/whitespace removed.
- **vahadane**: `Original Vahadane paper <https://ieeexplore.ieee.org/document/7460968>`_.
- **augment**: HSV colorspace augmentation.
- **cyclegan**: CycleGAN-based stain normalization, as implemented by `Zingman et al <https://github.com/Boehringer-Ingelheim/stain-transfer>`_ (PyTorch only)

Overview
********

The main normalizer interface, :class:`slideflow.norm.StainNormalizer`, offers
efficient numpy implementations for the Macenko, Reinhard, and Vahadane H&E stain normalization algorithms, as well
as an HSV colorspace stain augmentation method. This normalizer can convert
images to and from Tensors, numpy arrays, and raw JPEG/PNG images.

In addition to these numpy implementations, PyTorch-native and Tensorflow-native
implementations are also provided, which offer performance improvements, GPU acceleration,
and/or vectorized application. The native normalizers are found in
``slideflow.norm.tensorflow`` and ``slideflow.norm.torch``, respectively.

The Vahadane normalizer has two numpy implementations available: SPAMS
(``vahadane_spams``) and sklearn (``vahadane_sklearn``). By default,
the SPAMS implementation will be used if unspecified (``method='vahadane'``).

Use :func:`slideflow.norm.autoselect` to get the fastest available normalizer
for a given method and active backend (Tensorflow/PyTorch).

How to use
**********

There are four ways you can use stain normalizers: 1) on individual images, 2) during dataset iteration, 3) during tile extraction, or 4) on-the-fly during training.

Individual images
-----------------

Stain normalizers can be used directly on individual images or batches of images. The Tensorflow and PyTorch-native stain normalizers perform operations on Tensors, allowing you to incoporate stain normalization into an external preprocessing pipeline.

Load a backend-native stain normalizer with ``autoselect``, then transform an image with ``StainNormalizer.transform()``. This function will auto-detect the source image type, perform the most efficient transformation possible, and return normalized images of the same type.

.. code-block:: python

    import slideflow as sf

    macenko = sf.norm.autoselect('macenko')
    image = macenko.transform(image)

You can use :meth:`slideflow.norm.StainNormalizer.fit` to fit the normalizer to a custom reference image, or use one of our preset fits.

Dataloader pre-processing
-------------------------

You can apply stain normalization during dataloader preprocessing by passing the ``StainNormalizer`` object to the ``normalizer`` argument of either ``Dataset.tensorflow()`` or ``Dataset.torch()``.

.. code-block:: python

    import slideflow as sf

    # Get a PyTorch-native Macenko normalizer
    macenko = sf.norm.autoselect('macenko')

    # Create a PyTorch dataloader that applies stain normalization
    dataset = sf.Dataset(...)
    dataloader = dataset.torch(..., normalizer=macenko)

.. note::

    GPU acceleration cannot be performed within a PyTorch dataloader. Stain normalizers have a ``.preprocess()`` function that stain-normalizes and standardizes a batch of images, so the workflow to normalize on GPU in a custom PyTorch training loop would be:

    - Get a Dataloader with ``dataset.torch(standardize=False, normalize=False)``
    - On an image batch, preprocess with ``normalizer.preprocess()``:

    .. code-block:: python

        # Slideflow dataset
        dataset = Project.dataset(tile_px=..., tile_um=...)

        # Create PyTorch dataloader
        dataloader = dataset.torch(..., standardize=False)

        # Get a stain normalizer
        normalizer = sf.norm.autoselect('reinhard')

        # Iterate through the dataloader
        for img_batch, labels in dataloader:

            # Stain normalize using GPU
            img_batch = img_batch.to('cuda')
            with torch.no_grad():
                proc_batch = normalizer.preprocess(img_batch)

            ...


During tile extraction
----------------------

Image tiles can be normalized during tile extraction by using the ``normalizer`` and ``normalizer_source`` arguments. ``normalizer`` is the name of the algorithm. The normalizer source - either a path to a reference image, or a ``str`` indicating one of our presets (e.g. ``'v1'``, ``'v2'``, ``'v3'``) - can also be set with ``normalizer_source``.

.. code-block:: python

    P.extract_tiles(
      tile_px=299,
      tile_um=302,
      normalizer='reinhard'
    )

On-the-fly
----------

Performing stain normalization on-the-fly provides greater flexibility, as it allows you to change normalization strategies without re-extracting all of your image tiles.

Real-time normalization can be performed for most pipeline functions - such as model training or feature generation - by setting the ``normalizer`` and/or ``normalizer_source`` hyperparameters.

.. code-block:: python

    from slideflow.model import ModelParams
    hp = ModelParams(..., normalizer='reinhard')

If a model was trained using a normalizer, the normalizer algorithm and fit information will be stored in the model metadata file, ``params.json``, in the saved model folder. Any Slideflow function that uses this model will automatically process images using the same normalization strategy.

.. _normalizer_performance:

Performance
***********

Slideflow has Tensorflow, PyTorch, and Numpy/OpenCV implementations of stain normalization algorithms. Performance benchmarks for these implementations
are given below:

.. list-table:: **Performance Benchmarks** (299 x 299 images, Slideflow 2.0.0, benchmarked on 3960X and A100 40GB)
    :header-rows: 1

    * -
      - Tensorflow backend
      - PyTorch backend
    * - macenko
      - 929 img/s (**native**)
      - 881 img/s (**native**)
    * - macenko_fast
      - 1,404 img/s (**native**)
      - 1,088 img/s (**native**)
    * - reinhard
      - 1,136 img/s (**native**)
      - 3,329 img/s (**native**)
    * - reinhard_fast
      - 4,226 img/s (**native**)
      - 4,187 img/s (**native**)
    * - reinhard_mask
      - 1,136 img/s (**native**)
      - 3,941 img/s (**native**)
    * - reinhard_fast_mask
      - 4,496 img/s (**native**)
      - 4,058 img/s (**native**)
    * - vahadane_spams
      - 0.7 img/s
      - 2.2 img/s
    * - vahadane_sklearn
      - 0.9 img/s
      - 1.0 img/s

.. _contextual_normalization:

Contextual Normalization
************************

Contextual stain normalization allows you to stain normalize an image using the staining context of a separate image. When the context image is a thumbnail of the whole slide, this may provide slight improvements in normalization quality for areas of a slide that are predominantly eosin (e.g. necrosis or low cellularity). For the Macenko normalizer, this works by determining the maximum H&E concentrations from the context image rather than the image being transformed. For the Reinhard normalizer, channel means and standard deviations are calculated from the context image instead of the image being transformed. This normalization approach can result in poor quality images if the context image has pen marks or other artifacts, so we do not recommend using this approach without ROIs or effective slide-level filtering.

Contextual normalization can be enabled during tile extraction by passing the argument ``context_normalize=True`` to :meth:`slideflow.Dataset.extract_tiles()`.

You can use contextual normalization when manually using a ``StainNormalizer`` object by using the ``.context()`` function. The context can either be a slide (path or ``sf.WSI``) or an image (Tensor or np.ndarray).

.. code-block:: python

    import slideflow as sf

    # Get a Macenko normalizer
    macenko = sf.norm.autoselect('macenko')

    # Use a given slide as context
    slide = sf.WSI('slide.svs', ...)

    # Context normalize an image
    with macenko.context(slide):
        img = macenko.transform(img)

You can also manually set or clear the normalizer context with ``.set_context()`` and ``.clear_context()``:

.. code-block:: python

    # Set the normalizer context
    macenko.set_context(slide)

    # Context normalize an image
    img = macenko.transform(img)

    # Remove the normalizer context
    macenko.clear_context()

Contextual normalization is not supported with on-the-fly normalization during training or dataset iteration.

.. _stain_augmentation:

Stain Augmentation
******************

One of the benefits of on-the-fly stain normalization is the ability to perform dynamic stain augmentation with normalization. For Reinhard normalizers, this is performed by randomizing the channel means and channel standard deviations. For Macenko normalizers, stain augmentation is performed by randomizing the stain matrix target and the target concentrations. In all cases, randomization is performed by sampling from a normal distribution whose mean is the reference fit and whose standard deviation is a predefined value (in ``sf.norm.utils.augment_presets``). Of note, this strategy differs from the more commonly used strategy `described by Tellez <https://doi.org/10.1109/tmi.2018.2820199>`_, where augmentation is performed by randomly perturbing images in the stain matrix space without normalization.

To enable stain augmentation, add the letter 'n' to the ``augment`` parameter when training a model.

.. code-block:: python

    import slideflow as sf

    # Open a project
    project = sf.Project(...)

    # Add stain augmentation to augmentation pipeline
    params = sf.ModelParams(..., augment='xryjn')

    # Train a model
    project.train(..., params=params)

When using a StainNormalizer object, you can perform a combination of normalization and augmention for an image by using the argument ``augment=True`` when calling :meth:`StainNormalizer.transform`:

.. code-block:: python

    import slideflow as sf

    # Get a Macenko normalizer
    macenko = sf.norm.autoselect('macenko')

    # Perform combination of stain normalization and augmentation
    img = macenko.transform(img, augment=True)

To stain augment an image without normalization, use the method :meth:`StainNormalizer.augment`:

.. code-block:: python

    import slideflow as sf

    # Get a Macenko normalizer
    macenko = sf.norm.autoselect('macenko')

    # Perform stain augmentation
    img = macenko.augment(img)


StainNormalizer
***************

.. autoclass:: StainNormalizer
.. autofunction:: slideflow.norm.StainNormalizer.fit
.. autofunction:: slideflow.norm.StainNormalizer.get_fit
.. autofunction:: slideflow.norm.StainNormalizer.set_fit
.. autofunction:: slideflow.norm.StainNormalizer.augment
.. autofunction:: slideflow.norm.StainNormalizer.transform
.. autofunction:: slideflow.norm.StainNormalizer.jpeg_to_jpeg
.. autofunction:: slideflow.norm.StainNormalizer.jpeg_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.png_to_png
.. autofunction:: slideflow.norm.StainNormalizer.png_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.rgb_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.tf_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.tf_to_tf
.. autofunction:: slideflow.norm.StainNormalizer.torch_to_torch

Example images
**************

.. figure:: norm_compare/wsi_norm_compare.jpg

    Comparison of normalizers applied to a whole-slide image.

.. figure:: norm_compare/tile_norm_compare.jpg

    Comparison of normalizers applied to an image tile.

.. figure:: norm_compare/wsi_unnormalized.jpg

    Unnormalized whole-slide images.

.. figure:: norm_compare/wsi_reinhard_v1.jpg

    Whole-slide images normalized with **Reinhard**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_reinhard_v2.jpg

    Whole-slide images normalized with **Reinhard**, fit to preset "v2"

.. figure:: norm_compare/wsi_macenko_v1.jpg

    Whole-slide images normalized with **Macenko**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_macenko_v2.jpg

    Whole-slide images normalized with **Macenko**, fit to preset "v2"

.. figure:: norm_compare/wsi_vahadane_v1.jpg

    Whole-slide images normalized with **Vahadane**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_vahadane_v2.jpg

    Whole-slide images normalized with **Vahadane**, fit to preset "v2"

.. figure:: norm_compare/wsi_vahadane_spams_v1.jpg

    Whole-slide images normalized with **Vahadane (SPAMS)**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_vahadane_spams_v2.jpg

    Whole-slide images normalized with **Vahadane (SPAMS)**, fit to preset "v2"

.. figure:: norm_compare/tile_unnormalized.jpg

    Unnormalized image tiles.

.. figure:: norm_compare/tile_reinhard_v1.jpg

    Image tiles normalized with **Reinhard Mask**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_reinhard_v2.jpg

    Image tiles normalized with **Reinhard Mask**, fit to preset "v2"

.. figure:: norm_compare/tile_macenko_v1.jpg

    Image tiles normalized with **Macenko**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_macenko_v2.jpg

    Image tiles normalized with **Macenko**, fit to preset "v2"

.. figure:: norm_compare/tile_vahadane_v1.jpg

    Image tiles normalized with **Vahadane**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_vahadane_v2.jpg

    Image tiles normalized with **Vahadane**, fit to preset "v2"

.. figure:: norm_compare/tile_vahadane_spams_v1.jpg

    Image tiles normalized with **Vahadane (SPAMS)**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_vahadane_spams_v2.jpg

    Image tiles normalized with **Vahadane (SPAMS)**, fit to preset "v2"