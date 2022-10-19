.. _filtering:

Tile Extraction
===============

.. image:: tile_extraction_overview.png

|

The next step is tile extraction using the ``extract_tiles()`` function. The only arguments required are ``tile_px`` and ``tile_um``, which determine the size of the extracted image tiles in pixels and microns, respectively:

.. code-block:: python

    P.extract_tiles(tile_px=299, tile_um=302)

Slide scanners may have differing microns-per-pixel (MPP) resolutions, so "10X" magnification from one scanner may be slightly different than "10X" on another scanner. Specifying a fixed ``tile_um`` ensures all image tiles have both the same pixel size and micron size. This MPP-harmonization step uses the `Libvips resize <https://www.libvips.org/API/current/libvips-resample.html#vips-resize>`_ function on extracted images. To disable this step and instead extract tiles at a given `downsample layer <https://dicom.nema.org/dicom/dicomwsi/>`_, set ``tile_um`` equal to a magnification level rather than micron size:

.. code-block:: python

    P.extract_tiles(tile_px=299, tile_um="10x")

You can also choose to only extract tiles from a subset or your data. Use a column in the project annotations file to designate which slides should have tiles extracted by passing a dictionary to the ``filters`` argument. This dictionary should have keys equal to column names and values equal to a list of all acceptable values you want to include. If this argument is not supplied, all valid slides will be extracted.

For example, to extract tiles only for slides that are labeled as "train" in the "dataset" column header in your annotations file, do:

.. code-block:: python

    P.extract_tiles(
      tile_px=299,
      tile_um=302,
      filters={"dataset": ["train"]}
    )

To further filter by the annotation header "mutation_status", including only slides with the category "braf" or "ras", do:

.. code-block:: python

    P.extract_tiles(
      tile_px=299,
      tile_um=302,
      filters={
        "dataset": ["train"],
        "mutation_status": ["braf", "ras"]
      }
    )

.. note::
    The ``filters`` argument can be also used for filtering input slides in many slideflow functions, including ``train()``, ``evaluate()``, ``generate_heatmaps()``, and ``generate_mosaic()``.

Tiles will be extracted at the specified pixel and micron size. Tiles will be automatically stored in TFRecord format, although loose tiles can also be saved by passing ``save_tiles=True``.

The full documentation for the ``extract_tiles`` function is given below:

.. autofunction:: slideflow.Project.extract_tiles
   :noindex:

ROIs
****

By default, slides with valid ROIs will only have tiles extracted from within ROIs, and slides without ROIs will have tiles extracted across the whole-slide image. To skip slides that are missing ROIs, set ``roi_method='inside'``. To ignore ROIs entirely and extract tiles from whole-slide images, set ``roi_method='ignore'``. You can alternatively extract *outside* the annotated ROIs with ``roi_method='outside'``.

Stain normalization
*******************

.. note::
    See :py:mod:`slideflow.norm` for comparisons & benchmarks of stain normalization methods.

.. image:: norm_compare/wsi_norm_compare.jpg

Image tiles can undergo digital H&E stain normalization either during tile extraction or in real-time during training. Real-time normalization adds CPU overhead during training and inference but offers greater flexibility, allowing you to test different normalization strategies without re-extracting tiles from your entire dataset.

To normalize tiles during tile extraction, use the ``normalizer`` and ``normalizer_source`` arguments; ``normalizer`` is the name of the algorithm to use. A path to a normalization reference image may optionally be provided through ``normalizer_source``. Available stain normalization algorithms include:

- **macenko**: M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, *IEEE International Symposium on Biomedical Imaging: From Nano to Macro*, 2009, pp. 1107–1110.
- **vahadane**: A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, *IEEE Transactions on Medical Imaging*, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
- **reinhard**: E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, *IEEE Computer Graphics and Applications*, vol. 21, no. 5, pp. 34–41, Sep. 2001.
- **reinhard_fast**: A modification of the Reinhard algorithm with the brightness standardization step removed for computational efficiency.
- **augment**: HSV colorspace augmentation.

.. code-block:: python

    P.extract_tiles(
      tile_px=299,
      tile_um=302,
      normalizer='reinhard'
    )

Alternatively, real-time normalization can be performed with all pipeline functions that process TFRecords. For example, real-time normalization during training is enabled by setting the appropriate hyperparameter:

.. code-block:: python

    from slideflow.model import ModelParams
    hp = ModelParams(..., normalizer='reinhard')

If a normalizer was used during model training, the appropriate information will be stored in the model metadata file, ``params.json``, located in the saved model folder. Any Slideflow function that uses this model will then process images using the same normalization strategy.

The normalizer interfaces can also be access directly through :class:`slideflow.norm.StainNormalizer`. See :py:mod:`slideflow.norm` for examples and more information.

Background filtering
********************

.. image:: otsu.png

|

Slide background can be detected and filtered by two types of methods - **tile-based methods** and **slide-based methods**.

Whitespace and grayspace filtering are two **tile-based methods** that detect the amount of whitespace or grayspace in a given image tile, discarding the tile if the content exceeds a set threshold. Whitespace is calculated using overall brightness for each pixel, then counting the fraction of pixels with a brightness above some threshold. Grayspace is calculated by converting RGB images to the HSV spectrum, then counting the fraction of pixels with a saturation below some threshold. This filtering is performed separately for each tile as it is being extracted. Grayspace filtering is the default background filtering behavior. The arguments ``whitespace_fraction``, ``whitespace_threshold``, ``grayspace_fraction``, and ``grayspace_threshold`` are used for these methods, as described in the documentation for the tile extraction function (:func:`slideflow.Dataset.extract_tiles`).

Alternatively, Otsu's thresholding is a **slide-based method** that distinguishes foreground (tissue) from background (empty slide). Otsu's thresholding is performed in the HSV colorspace and yields similar results to grayspace filtering. Otsu's thresholding is ~30% faster than grayspace filtering for slides with accessible downsample layers, but if downsample layers are not stored in a given slide or are inaccessible (e.g. ``enable_downsample=False``), grayspace filtering may be faster. To use Otsu's thresholding, set the argument ``qc='otsu'`` (and disable grayspace filtering by setting ``grayspace_threshold=1``)

Quality control
***************

.. image:: blur.png

|

In addition to background filtering, additional blur-detection quality control can be used to identify artifact (pen marks) or out-of-focus areas. If annotated Regions of Interest (ROIs) are not available for your dataset, blur detection should be enabled in order to ensure that high quality image tiles are extracted. If ROIs *are* available, it may be unnecessary. Blur detection may increase tile extraction time by 50% or more.

To use blur detection QC, set ``qc='blur'`` (or ``qc='both'`` if also using Otsu's thresholding).

If both Otsu's thresholding and blur detection are being used, Slideflow will automatically calculate Blur Burden, a metric used to assess the degree to which non-background tiles are either out-of-focus or contain artifact. In the tile extraction PDF report that is generated, the distribution of blur burden for slides in the dataset will be plotted on the first page. The report will contain the number of slides meeting criteria for warning, when the blur burden exceeds 5% for a given slide. A text file containing names of slides with high blur burden will be saved in the exported TFRecords directory. These slides should be manually reviewed to ensure they are of high enough quality to include in the dataset.

Performance optimization
************************

The ``libvips`` library is used for all slide reading and tile extraction. As tile extraction is heavily reliant on random access reading, significant performance gains can be experienced by either 1) moving all slides to an SSD, or 2) utilizing an SSD or ramdisk buffer (to which slides will be copied prior to extraction). The use of a ramdisk buffer can improve tile extraction speed by 10-fold or greater! To maximize performance, pass the buffer path to the argument ``buffer``.

Extraction reports
******************

Once tiles have been extracted, a PDF report will be generated with a summary and sample of tiles extracted from their corresponding slides. An example of such a report is given below. Reviewing this report may enable you to identify data corruption, artifacts with stain normalization, or suboptimal background filtering. The report is saved in the TFRecords directory.

.. image:: example_report_small.jpg

In addition to viewing reports after tile extraction, you may generate new reports on existing tfrecords with :func:`slideflow.Dataset.tfrecord_report`, by calling this function on a given dataset (see :ref:`dataset` for more information on datasets). For example:

.. code-block:: python

    dataset = P.dataset(tile_px=299, tile_um=302)
    dataset.tfrecord_report("/path/to/dest")

You can also generate reports for slides that have not yet been extracted by passing ``dry_run=True`` to :meth:`slideflow.Dataset.extract_tiles`.