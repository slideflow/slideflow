.. currentmodule:: slideflow.dataset

.. _dataset:

slideflow.dataset
==================

The :class:`Dataset` class in this module is used to organize dataset sources, ROI annotations,
clinical annotations, and dataset processing.

Dataset Organization
---------------------

A *source* is a set of slides, corresponding Regions of Interest (ROI) annotations (if available), and any tiles
extracted from these slides, either as loose tiles or in the binary TFRecord format. Sources are defined in the
project dataset configuration JSON file, with the following format:

.. code-block:: json

    {
        "SOURCE":
        {
            "slides": "/directory",
            "roi": "/directory",
            "tiles": "/directory",
            "tfrecords": "/directory",
        }
    }

A single *dataset* can have multiple sources. One example of this might be if you were performing a pan-cancer analysis;
you would likely have a unique source for each cancer subtype, in order to keep each set of slides and tiles distinct.
Another example might be if you are analyzing slides from multiple institutions, and you want to ensure that you are
not mixing your training and evaluation datasets.

The :class:`Dataset` class is initialized from a dataset configuration file, a list of source names
to include from the configuration file, and tile size parameters (``tile_px`` and ``tile_um``). Clinical annotations can be
provided to this object, which can then be used to filter slides according to outcomes and perform a variety of other
class-aware functions.

Filtering
---------

Datasets can be filtered with several different filtering mechanisms:

- **filters**: A dictionary can be passed via the ``filters`` argument to a Dataset to perform filtering. The keys of this dictionary should be annotation headers, and the values of this dictionary indicate the categorical outcomes which should be included. Any slides with an outcome other than what is provided by this dict will be excluded.
- **filter_blank**: A list of headers can be provided to the ``filter_blank`` argument; any slide with a blank annotation in one of these columns will be excluded.
- **min_tiles**: An int can be provided to ``min_tiles``; any tfrecords with fewer than this number of tiles will be excluded.

Filters can be provided at the time of Dataset instantiation by passing to the initializer:

.. code-block:: python

    dataset = Dataset(..., filters={'HPV_status': ['negative', 'positive']})

... or with the :meth:`Dataset.filter` method:

.. code-block:: python

    dataset = dataset.filter(min_tiles=50)

Once applied, all dataset functions and parameters will reflect this filtering criteria, including the :attr:`Dataset.num_tiles` parameter.

Dataset Manipulation
--------------------

A number of different functions can be applied to Datasets in order to manipulate filters (:meth:`Dataset.filter`, :meth:`Dataset.remove_filter`, :meth:`Dataset.clear_filters`), balance datasets (:meth:`Dataset.balance`), or clip tfrecords to a maximum number of tiles (:meth:`Dataset.clip`). The full documentation of these functions is given below. Note: these functions return a Dataset copy with the functions applied, not to the original dataset. Thus, for proper use, assign the result of the function to the original dataset variable:

.. code-block:: python

    dataset = dataset.clip(50)

This also means that these functions can be chained for simplicity:

.. code-block:: python

    dataset = dataset.balance('HPV_status').clip(50)


Manifest
--------

The Dataset manifest is a dictionary mapping tfrecords to both the total number of slides, as well as the number of slides after any clipping or balancing. For example, after clipping:

.. code-block:: python

    dataset = dataset.clip(500)

... the :meth:`Dataset.manifest` function would return something like:

.. code-block:: json

    {
        "/path/tfrecord1.tfrecords":
        {
            "total": 1526,
            "clipped": 500
        },
        "/path/tfrecord2.tfrecords":
        {
            "total": 455,
            "clipped": 455
        }
    }

Training/Validation Splitting
-----------------------------

Datasets can be split into training and validation datasets with :meth:`Dataset.train_val_split`, with full documentation given below. The result of this function is two datasets - the first training, the second validation - each a separate instance of :class:`Dataset`.

Tile and TFRecord Processing
----------------------------

Datasets can also be used to process and extract tiles. Some example methods support tile and tfrecord processing include:

- :meth:`Dataset.extract_tiles`: Performs tile extraction for all slides in the dataset.
- :meth:`Dataset.extract_tiles_from_tfrecords`: Extract tiles from saved TFRecords, saving in loose .jpg or .png format to a folder.
- :meth:`Dataset.resize_tfrecords`: Resizes all images in TFRecords to a new size.
- :meth:`Dataset.split_tfrecords_by_roi`: Splits a set of extracted tfrecords according to whether tiles are inside or outside the slide's ROI.
- :meth:`Dataset.tfrecord_report`: Generates a PDF report of the tiles inside a collection of TFRecords.

Tensorflow & PyTorch Datasets
-----------------------------

Finally, Datasets can also return either a ``tf.data.Datasets`` or ``torch.utils.data.Dataloader`` object to quickly and easily create a deep learning dataset ready to be used as model input, with the :meth:`Dataset.tensorflow` and :meth:`Dataset.torch` methods, respectively.

.. automodule: slideflow.dataset

Dataset
--------

.. autoclass:: slideflow.Dataset
    :inherited-members: