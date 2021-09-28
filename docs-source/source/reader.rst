.. currentmodule:: slideflow.io.reader

slideflow.io.reader
===================

The purpose of this module is to provide a performant, backend-agnostic TFRecord reader and interleaver to use as
input for Torch models. It uses `dareblopy <https://github.com/podgorskiy/DareBlopy>`_ for the binary TFRecord file
reading, and contains :func:`slideflow.io.reader.interleave` function built around this package capable of efficient interleaving
and balancing.

.. warning::
    If using this module to generate input for a Torch model, please note that performance may suffer if
    Tensorflow has also been been loaded. Many modules in this package are not loaded by default for this
    reason, in order to delay the loading of Tensorflow until needed in case backend-agnostic functions are required.

.. automodule:: slideflow.io.reader
    :members: