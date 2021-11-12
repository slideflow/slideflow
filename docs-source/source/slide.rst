.. currentmodule:: slideflow.slide

slideflow.slide
=====================

This module contains classes to load slides and extract tiles. For optimal performance, tile extraction should
generally not be performed by instancing these classes directly, but by calling either
:func:`slideflow.Project.extract_tiles` or :func:`slideflow.Dataset.extract_tiles`, which include performance
optimizations and additional functionality.

WSI
***
.. autoclass:: WSI
    :inherited-members:

TMA
***
.. autoclass:: TMA
    :inherited-members: