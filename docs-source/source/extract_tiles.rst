.. _filtering:

Tile extraction
===============

The next step is tile extraction, which is accomplished using the ``extract_tiles()`` function. The only arguments required are ``tile_px`` and ``tile_um``, which determine the size of the extracted tiles in pixels and microns, respectively:

.. code-block:: python

    P.extract_tiles(tile_px=299, tile_um=302)

To filter according to a columns in your annotations file, pass a dictionary to ``filters``, with keys equal to column names and values equal to a list of all acceptable values you want to include. If this argument is not supplied, all valid slides will be extracted.

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

By default, slides with valid ROIs will only have tiles extracted from within ROIs, and slides without ROIs will have tiles extracted across the whole-slide image. To skip slides that are missing ROIs, use ``skip_missing_roi=True``. To ignore ROIs entirely and extract tiles from whole-slide images, pass ``roi_method='ignore'``. You can alternatively extract *outside* the annotated ROIs by passing ``roi_method='outside'``.

Stain Normalization
*******************

Tiles can be normalized to account for differing strengths of H&E staining, which has been shown to improve machine learning accuracy on some datasets. Several normalization algorithms exist, and none have shown clear superiority over the other. However, while tile normalization may improve training performance, some tiles and slides may be prone to artifacts as a result of normalization algorithms.

If you choose to use normalization, you may either normalize images to an internal H&E-stained control image contained within the pipeline, or you may explicitly provide a reference image for normalization.

Normalization can be performed at the time of tile extraction or in real-time during training. Real-time normalization adds CPU overhead and may increase training or inference times for some models, although it allows greater flexibility, as normalization strategies can be changed without re-extracting tiles from your entire dataset.

To normalize tiles during tile extraction, use the ``normalizer`` and ``normalizer_source`` arguments; ``normalizer`` is the name of the algorithm to use. A path to a normalization reference image may optionally be provided through ``normalizer_source``. Available stain normalization algorithms include:

- **macenko**: M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, *IEEE International Symposium on Biomedical Imaging: From Nano to Macro*, 2009, pp. 1107–1110.
- **vahadane**: A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, *IEEE Transactions on Medical Imaging*, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
- **reinhard**: E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, *IEEE Computer Graphics and Applications*, vol. 21, no. 5, pp. 34–41, Sep. 2001.
- **reinhard_fast**: A modification of the Reinhard algorithm with the brightness standardization step removed for computational efficiency.

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

If a normalizer was used during model training, the appropriate information will be stored in the model metadata file, `params.json`, located in the saved model folder. Any function within `slideflow` that uses this model will then process images using the same normalization strategy.

Background filtering
********************

Slide background can be detected and filtered by two types of methods - tile-based methods and slide-based methods.

Whitespace and grayspace filtering are two tile-based methods that detect the amount of whitespace or grayspace in a given tile, discarding the tile if the content exceeds a set threshold. Whitespace is calculated using overall brightness for each pixel, then counting the fraction of pixels with a brightness above some threshold. Grayspace is calculated by converting RGB images to the HSV spectrum, then counting the fraction of pixels with a saturation below some threshold. This filtering is performed separately for each tile as it is being extracted. Grayspace filtering is the default background filtering behavior. The arguments ``whitespace_fraction``, ``whitespace_threshold``, ``grayspace_fraction``, and ``grayspace_threshold`` are used for these methods, as described in the documentation for the tile extraction function (:func:`slideflow.Dataset.extract_tiles`).

Alternatively, Otsu's thresholding can be performed on the lowest downsample level for a whole slide. This method generates a mask that identifies areas of foreground and marks areas of background to be discarded. Otsu's thresholding is performed in the HSV colorspace, and generally yields identical results to grayspace filtering. Otsu's thresholding is ~30% faster than grayspace filtering for slides with accessible downsample layers, but if downsample layers are not stored in a given slide or are inaccessible (e.g. ``enable_downsample=False``, which should be set for any system that does not have a patched pixman library), grayspace filtering will be significantly faster. To use Otsu's thresholding, set the argument ``qc='otsu'`` (and disable grayspace filtering by setting ``grayspace_threshold=1``)

If you have pixman>0.38 and use slides with accessible downsample layers, Otsu's thresholding should be used. Otherwise, grayspace filtering will be faster.

Quality control
***************

In addition to background filtering, additional blur-detection quality control can be used to identify out-of-focus areas, or areas with artifact. If annotated Regions of Interest (ROIs) are not available for your dataset, blur detection quality control should be enabled in order to ensure that high quality image tiles are extracted. If ROIs *are* available, it may be unnecessary. Blur detection may increase tile extraction time by 50% or more.

To use blur detection QC, set ``qc='blur'`` (or ``qc='both'`` if also using Otsu's thresholding).

If both Otsu's thresholding and blur detection are being used, Slideflow will automatically calculate Blur Burden, a metric used to assess the degree to which non-background tiles are either out-of-focus or contain artifact. In the tile extraction PDF report that is generated, the distribution of blur burden for slides in the dataset will be plotted on the first page. The report will contain the number of slides meeting criteria for warning, when the blur burden exceeds 5% for a given slide. A text file containing names of slides with high blur burden will be saved in the exported TFRecords directory. These slides should be manually reviewed to ensure they are of high enough quality to include in the dataset.

Performance optimization
************************

The ``libvips`` library is used for all slide reading and tile extraction. As tile extraction is heavily reliant on random access reading, significant performance gains can be experienced by either 1) moving all slides to an SSD, or 2) utilizing an SSD or ramdisk buffer (to which slides will be copied prior to extraction). The use of a ramdisk buffer can improve tile extraction speed by 10-fold or greater! To maximize performance, pass the buffer path to the argument ``buffer``.

Multiprocessing and multithreading is used during tile extraction to maximize performance efficiency. The number of process workers and threads per worker can be manually specified with ``num_workers`` and ``num_threads``, respectively. Optimal results are generally seen by setting ``num_workers=2`` and ``num_threads`` equal to the number of CPU cores available. Tile extraction speed scales linearly with CPU core availability.

Extraction reports
******************

Once tiles have been extracted, a PDF report will be generated with a summary and sample of tiles extracted from their corresponding slides. An example of such a report is given below. It is generally good practice to review this report, as you may catch slides with data corruption, artifacts with stain normalization, or suboptimal whitespace/grayspace filtering. The report is saved in the project root directory.

In addition to viewing reports after tile extraction, you may generate new reports on existing tfrecords with :func:`slideflow.Dataset.tfrecord_report`, by calling this function on a given dataset (see :ref:`dataset` for more information on datasets). For example:

.. code-block:: python

    dataset = P.dataset(tile_px=299, tile_um=302)
    dataset.tfrecord_report("/path/to/dest")

You can also generate reports for slides that have not yet been extracted by passing ``dry_run=True`` to :meth:`slideflow.Dataset.extract_tiles`.