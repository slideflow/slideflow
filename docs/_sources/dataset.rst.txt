.. currentmodule:: slideflow.dataset

slideflow.dataset
=====================

The :class:`slideflow.dataset.Dataset` class in this module is used to organize dataset sources, ROI annotations,
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

The :class:`slideflow.dataset.Dataset` class is initialized from a dataset configuration file, a list of source names
to include from the configuration file, and tile size parameters (tile_px and tile_um). Clinical annotations can be
provided to this object, which can then be used to filter slides according to outcomes and perform a variety of other
class-aware function.

The full documentation for this class and its constituent methods is given below.

.. automodule: slideflow.dataset

Dataset
--------

.. autoclass:: Dataset
    :inherited-members: