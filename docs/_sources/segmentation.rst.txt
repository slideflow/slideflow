.. currentmodule:: slideflow.segmentation

.. _segmentation:

Tissue Segmentation
===================

In addition to classification tasks, Slideflow also supports training and deploying whole-slide tissue segmentation models. Segmentation models identify and label regions of interest in a slide, and can be used for tasks such as tumor identification, tissue labeling, or quality control. Once trained, these models can be used for :ref:`slide QC <filtering>`, generating :ref:`regions of interest <regions_of_interest>`, or live deployment in :ref:`Slideflow Studio <studio>`.

.. note::

    Tissue segmentation requires PyTorch. Dependencies can be installed with ``pip install slideflow[torch]``.

Segmentation Modes
------------------

Tissue segmentation is performed at the whole-slide level, trained on randomly cropped sections of the slide thumbnail at a specified resolution. Slideflow supports three segmentation modes:

- ``'binary'``: For binary segmentation, the goal is to differentiate a single tissue type from background.
- ``'multiclass'``: For multiclass segmentation, the goal is twofold: differentiate tissue from background, and assign a class label to each identified region. This is useful in instances where regions have non-overlapping labels.
- ``'multilabel'``: For multilabel segmentation, the goal is to assign each tissue type to a class, but regions may have overlapping labels.

Generating Data
---------------

.. note::
    Segmentation thumbnails and masks do not need to be explicitly exported prior to training. They will be generated automatically during training if they do not exist. However, exporting them beforehand can be useful for data visualization, troubleshooting, and computational efficiency.


Segmentation models in Slideflow are trained on regions of interest, which can be generated as discussed in :ref:`regions_of_interest` and :ref:`studio_roi`. Once ROIs have been generated and (optionally) labeled, whole-slide thumbnails and ROI masks can be exported using ``segment.export_thumbs_and_masks()``. The ``mpp`` argument specifies the resolution of the exported images in microns-per-pixel. We recommend ``mpp=20`` for a good balance between image size and memory requirements, or ``mpp=10`` for tasks needing higher resolution.

.. code-block:: python

    from slideflow import segment

    # Load a project and dataset
    project = slideflow.load_project('path/to/project')
    dataset = project.dataset()

    # Export thumbnails and masks
    segment.export_thumbs_and_masks(
        dataset,
        mpp=20,   # Microns-per-pixel resolution
        dest='path/to/output'
    )

By default, ROIs are exported as binary masks. To export multidimensional masks for multiclass or multilabel applications, use the ``mode`` and ``labels`` arguments. When ``mode`` is ``'multiclass'`` or ``'multilabel'``, masks will be exported in (N, W, H) format, where N is the number of unique ROI labels. The ``labels`` argument should be a list of strings corresponding to the ROI labels in the dataset that should be included.

.. code-block:: python

    ...

    # Export thumbnails and masks
    segment.export_thumbs_and_masks(
        dataset,
        mpp=20,   # Microns-per-pixel resolution
        dest='path/to/output',
        mode='multiclass',
        labels=['tumor', 'stroma', 'necrosis']
    )


Training a Model
----------------

Segmentation models are configured using a :class:`segment.SegmentConfig` object. This object specifies the model architecture, image resolution (MPP), training parameters, and other settings. For example, to configure a model for multiclass segmentation with a resolution of 20 MPP, use:

.. code-block:: python

    from slideflow import segment

    # Create a config object
    config = segment.SegmentConfig(
        mpp=20,     # Microns-per-pixel resolution
        size=1024,  # Size of cropped/rotated images during training
        mode='multiclass',
        labels=['tumor', 'stroma', 'necrosis'],
        arch='Unet',
        encoder_name='resnet34',
        train_batch_size=16,
        epochs=10,
        lr=1e-4,
    )

Slideflow uses the `segmentation_models_pytorch <https://github.com/qubvel/segmentation_models.pytorch>`_ library to implement segmentation models. The ``arch`` argument specifies the model architecture, and the ``encoder_name`` argument specifies the encoder backbone. See available models and encoders in the `segmentation_models_pytorch documentation <https://smp.readthedocs.io/en/latest/models.html>`_.

The segmentation model can then be trained using the :func:`segment.train` function. This function takes a :class:`segment.SegmentConfig` object and a :class:`slideflow.Dataset` object as arguments. During training, segmentation thumbnails and masks are randomly cropped to the specified ``size``, and images/masks then undergo augmentation with random flipping/rotating.

For example, to train a model for binary segmentation with a resolution of 20 MPP, use:

.. code-block:: python

    from slideflow import segment

    # Create a config object
    config = segment.SegmentConfig(mpp=20, mode='binary', arch='FPN')

    # Train the model
    segment.train(config, dataset, dest='path/to/output')

To use thumbnails and masks previously exported with :func:`segment.export_thumbs_and_masks`, specify the path to the exported data using the ``data_source`` argument. This is more computationally efficient than generating data on-the-fly during training. For example:

.. code-block:: python

    from slideflow import segment

    # Export thumbnails and masks
    segment.export_thumbs_and_masks(dataset, mpp=20, dest='masks/')

    # Create a config object
    config = segment.SegmentConfig(mpp=20, mode='binary', arch='FPN')

    # Train the model
    segment.train(config, dataset, data_source='masks/', dest='path/to/output')

After training, the model will be saved as a ``model.pth`` file in the destination directory specified by ``dest``, and the model configuration will be saved as a ``segment_config.json`` file.

Model Inference
---------------

After training, models can be loaded using :func:`segment.load_model_and_config`. This function takes a path to a model file as an argument, and returns a tuple containing the model and configuration object. For example:

.. code-block:: python

    from slideflow import segment

    # Load the model and config
    model, config = segment.load_model_and_config('path/to/model.pth')

To run inference on a slide, use the :meth:`segment.SegmentModel.run_slide_inference` method. This method takes a :class:`slideflow.WSI` object or str (path to slide) as an argument, and returns an array of pixel-level predictions. For binary models, the output shape will be ``(H, W)``. For multiclass models, the output shape will be ``(N+1, H, W)`` (the first channel is predicted background), and for multilabel models, the output shape will be ``(N, H, W)``, where ``N`` is the number of labels.

.. code-block:: python

    from slideflow import segment

    # Load the model and config
    model, config = segment.load_model_and_config('path/to/model.pth')

    # Run inference, returning an np.ndarray
    pred = model.run_slide_inference('/path/to/slide')

You can also run inference directly on an arbitrary image using the :meth:`segment.SegmentModel.run_tiled_inference` method. This method takes an image array (np.ndarray, in W, H, C format) as an argument, and returns an array of pixel-level predictions. Predictions are generated in tiles and merged. The output shape will be ``(H, W)`` for binary models, ``(N+1, H, W)`` for multiclass models, and ``(N, H, W)`` for multilabel models.

Generating QC Masks
-------------------

The :class:`slideflow.slide.qc.Segment` class provides an easy interface for generating QC masks from a segmentation model. This class takes a path to a trained segmentation model as an argument, and can be used for QC :ref:`as previously described <filtering>`. For example:

.. code-block:: python

    import slideflow as sf
    from slideflow.slide import qc

    # Load a project and dataset
    project = sf.load_project('path/to/project')
    dataset = project.dataset(299, 302)

    # Create a QC mask
    segmenter = qc.Segment('/path/to/model.pth')

    # Extract tiles with this QC
    dataset.extract_tiles(..., qc=segmenter)

You can also use this interface for applying QC to a single slide:

.. code-block:: python

    import slideflow as sf
    from slideflow.slide import qc

    # Load the slide
    wsi = sf.WSI('/path/to/slide', ...)

    # Create the QC algorithm
    segmenter = qc.Segment('/path/to/model.pth')

    # Apply QC
    applied_mask = wsi.qc(segmenter)

For binary models, the QC mask will filter out tiles that are predicted to be background.

For multiclass models, the QC mask will filter out tiles predicted to be background (class index 0). This can be customized by setting ``class_idx`` to another value. For example, to create a QC algorithm that filters out tiles predicted to be tumor (class index 1), use:

.. code-block:: python

    segmenter = qc.Segment('/path/to/model.pth', class_idx=1)

For multilabel models, the QC mask will filter out tiles predicted to be background for all class labels. This can be customized to filter out tiles based only on a specific class label by setting ``class_idx``. For example, to create a QC algorithm that filters out tiles that are not predicted to be tumor (class index 1) while ignoring predictions for necrosis (class index 2), use:

.. code-block:: python

    segmenter = qc.Segment('/path/to/model.pth', class_idx=1)

In all cases, the thresholding direction can be reversed with by setting ``threshold_direction='greater'``. This might be useful, for example, if the segmentation model was trained to identify pen marks or artifacts, and you want to filter out areas predicted to be artifacts.

.. code-block:: python

    segmenter = qc.Segment('/path/to/model.pth', threshold_direction='greater')

Generating ROIs
---------------

The :class:`slideflow.slide.qc.Segment` also provides an easy interface for generating regions of interest (ROIs). Use :meth:`slideflow.slide.qc.Segment.generate_rois` method to generate and apply ROIs to a slide. If the segmentation model is multiclass or multilabel, generated ROIs will be labeled. For example:

.. code-block:: python

    import slideflow as sf
    from slideflow.slide import qc

    # Load a project and dataset
    wsi = sf.WSI('/path/to/slide', ...)

    # Create a QC mask
    segmenter = qc.Segment('/path/to/model.pth')

    # Generate and apply ROIs to a slide
    roi_outlines = segmenter.generate_rois(wsi)

By default, this will apply generated ROIs directly to the :class:`slideflow.WSI` object. If you wish to calculate ROI outlines without applying them to the slide, use the argument ``apply=False``.

In addition to generating ROIs for a single slide, you can also generate ROIs for an entire dataset using :meth:`slideflow.Dataset.generate_rois`. For example:

.. code-block:: python

    import slideflow as sf

    # Load a project and dataset.
    project = sf.load_project('path/to/project')
    dataset = project.dataset()

    # Generate ROIs for all slides in the dataset.
    dataset.generate_rois('path/to/model.pth')

ROIs will be saved in the ROIs directory as configured in the dataset settings. Alternatively, ROIs can be exported to a user-defined directory using the ``dest`` argument.

By default, ROIs will be generated for all slides in the dataset, skipping slides with existing ROIs. To overwrite any existing ROIs, use the ``overwrite=True`` argument.


Deployment in Studio
--------------------

.. video:: tissue_seg.mp4
    :autoplay:

|

Segmentation models can be deployed in :ref:`Slideflow Studio <studio>` for live segmentation and QC. To do this, start by training a segmentation model as described above. Then, see the :ref:`studio_segmentation` documentation for instructions on how to deploy the model for live QC and/or ROI generation.


Complete Example
----------------

1. Label ROIs
*************

Create labeled ROIs as described in :ref:`studio_roi`.

2. Train a model
****************

.. code-block:: python

    import slideflow as sf
    from slideflow import segment

    # Load a project and dataset
    project = sf.load_project('path/to/project')
    dataset = project.dataset()

    # Train a binary segmentation model
    config = segment.SegmentConfig(mpp=20, mode='binary', arch='FPN')
    segment.train(config, dataset, dest='path/to/output')

3. Generate ROIs (optional)
***************************

.. code-block:: python

    import slideflow as sf

    # Load a project and dataset.
    project = sf.load_project('path/to/project')
    dataset = project.dataset()

    # Generate ROIs for all slides in the dataset.
    dataset.generate_rois('path/to/model.pth')

4. Deploy in Studio
*******************

Use the model for either QC or ROI generation in Slideflow Studio, as described in :ref:`studio_segmentation`.

