.. currentmodule:: slideflow.slide.qc

.. _qc:

slideflow.slide.qc
==================

This module contains functions for slide-level quality control, including Otsu's thresholding and Gaussian blur filtering. Quality control methods are used by passing a list of callables to the ``qc`` argument of ``.extract_tiles()``. They can also be directly applied to a :class:`slideflow.WSI` object with :meth:`slideflow.WSI.qc`.

.. code-block:: python

  import slideflow as sf
  from slideflow.slide import qc

  # Define custom QC options
  qc = [
    qc.Otsu(),
    qc.Gaussian(sigma=2)
  ]

  # Use this QC during tile extraction
  P.extract_tiles(qc=qc)

  # Alternatively, you can use the same QC directly on a WSI object
  wsi = sf.WSI(...)
  wsi.qc(qc).show()


Calculated QC masks can be automatically saved and loaded by using the :class:`slideflow.slide.qc.Save` and :class:`slideflow.slide.qc.Load` methods. By default, masks are saved in the same folder as whole-slide images.

.. code-block:: python

  from slideflow.slide import qc

  # Define a QC approach that auto-saves masks
  qc = [
    qc.Otsu(),
    qc.Save()
  ]
  P.extract_tiles(qc=qc)

  ...
  # Auto-load previously saved masks
  qc = [
    qc.Load()
  ]
  P.extract_tiles(qc=qc)

.. automodule:: slideflow.slide.qc
    :members:

Otsu
****
.. autoclass:: Otsu
    :inherited-members:

Gaussian
********
.. autoclass:: Gaussian
    :inherited-members:

Save
****
.. autoclass:: Save
    :inherited-members:

Load
****
.. autoclass:: Load
    :inherited-members: