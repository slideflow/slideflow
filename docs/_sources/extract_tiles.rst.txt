.. _filtering:

Tile extraction
===============

Once a validation plan has been established, our next step is tile extraction, which is done with the ``extract_tiles()`` function. The only arguments required are ``tile_px`` and ``tile_um``, which determine the size of the extracted tiles in pixels and microns, respectively:

.. code-block:: python

    SFP.extract_tiles(tile_px=299, tile_um=302)

To filter according to a columns in your annotations file, pass a dictionary to ``filters``, with keys equal to column names and values equal to a list of all acceptable values you want to include. If this argument is not supplied, all valid slides will be extracted.

For example, to extract tiles only for slides that are labeled as "train" in the "dataset" column header in your annotations file, do:

.. code-block:: python

    SFP.extract_tiles(tile_px=299, tile_um=302, filters={"dataset": ["train"]})

To further filter by the annotation header "mutation_status", including only slides with the category "braf" or "ras", do:

.. code-block:: python

    SFP.extract_tiles(tile_px=299, tile_um=302, filters={"dataset": ["train"], "mutation_status": ["braf", "ras"]})

.. note::
    The ``filters`` argument can be also used for filtering input slides in many slideflow functions, including ``train()``, ``evaluate()``, ``generate_heatmaps()``, and ``generate_mosaic()``.

Tiles will be extracted at the specified pixel and micron size. Tiles will be automatically stored in TFRecord format, although loose tiles can also be saved by passing a destination path to the argument ``tiles_dir``.

The full documentation for the ``extract_tiles`` function is given below:

.. autofunction:: slideflow.project.Project.extract_tiles
   :noindex:

ROIs
****

By default, only tiles with valid ROIs will be extracted, and tiles will only be extracted from within annotated ROIs. To disable skipping of slides that lack ROIs, pass ``skip_missing_roi=False``. To ignore ROIs entirely and extract entire slides, pass ``roi_method='ignore'``. You can alternatively extract *outside* the annotated ROIs by passing ``roi_method='outside'``.

Normalization
*************

Tiles can be normalized to account for differing strengths of H&E staining, which has been shown to improve machine learning accuracy on some datasets. Several normalization algorithms exist, and none have shown clear superiority over the other. However, while tile normalization may improve training performance, some tiles and slides may be prone to artifacts as a result of normalization algorithms.

If you choose to use tile normalization, you may either normalize the tile to an internal H&E-stained image contained within the pipeline, or you may explicitly provide a reference image for normalization.

Normalization can be done on-the-fly or at the time of tile extraction prior to storage in TFRecords. On-the-fly normalization adds significant CPU overhead and is generally not recommended. Normalization can also be done at the time of extraction, which reduces CPU requirements during other pipeline functions. To normalize tiles during extraction, use the ``normalizer`` and ``normalizer_source`` arguments; ``normalizer`` is the name of the algorithm to use, and can include 'macenko', 'reinhard', or 'vahadane'. A path to a normalization reference image may optionally be provided through ``normalizer_source``.

.. code-block:: python

    SFP.extract_tiles(tile_px=299, tile_um=302, normalizer='reinhard')

Alternatively, real-time normalization can be performed with nearly any pipeline function that accepts TFRecord inputs. For example, to normalize tiles during training:

.. code-block:: python

    SFP.train(..., normalizer='reinhard', normalizer_source='/path/to/reference.png')

Whitespace/grayspace filtering
******************************

In order to filter out background tiles, either whitespace or grayspace filtering may be used. Whitespace filtering is performed by calculating overall brightness for each pixel, and counting the fraction of pixels with a brightness above some threshold. Grayspace filtering is performed by convering RGB pixels to the HSV (hue, saturation, value) colorspace, and counting the fraction of pixels with a saturation below some threshold.

To perform filtering at the time of tile extraction, use the arguments ``whitespace_fraction``, ``whitespace_threshold``, ``grayspace_fraction``, and ``grayspace_threshold``, as described in the documentation, :func:`slideflow.project.Project.extract_tiles`.

Performance optimization
************************

The ``libvips`` library is used for all slide reading and tile extraction. As tile extraction is heavily reliant on random access reading, significant performance gains can be experienced by either 1) moving all slides to an SSD, or 2) utilizing an SSD or ramdisk buffer (to which slides will be copied prior to extraction). The use of a ramdisk buffer can improve tile extraction speed by 10-fold or greater! To maximize performance, pass the buffer path to the argument ``buffer``.

Multiprocessing and multithreading is used during tile extraction to maximize performance efficiency. The number of process workers and threads per worker can be manually specified with ``num_workers`` and ``threads_per_worker``, respectively (both of which default to 4). Increasing the available threads per worker will improve CPU utilization for highly multi-core servers.

Extraction reports
******************

Once tiles have been extracted, a PDF report will be generated with a summary and sample of tiles extracted from their corresponding slides. An example of such a report is given below. It is generally good practice to review this report, as you may catch slides with data corruption, artifacts with stain normalization, or suboptimal whitespace/grayspace filtering. The report is saved in the project root directory.

In addition to viewing reports after tile extraction, you may generate new reports on existing tfrecords with :func:`slideflow.dataset.Dataset.tfrecord_report`, by calling this function on a given dataset (see :ref:`dataset` for more information on datasets). For example:

.. code-block:: python

    dataset = SFP.get_dataset(tile_px=299, tile_um=302)
    dataset.tfrecord_report("/path/to/dest")

You can also generate reports for slides that have not yet been extracted with :func:`slideflow.dataset.Dataset.slide_report`.