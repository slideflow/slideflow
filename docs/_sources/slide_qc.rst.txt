.. currentmodule:: slideflow.slide.qc

.. _qc:

slideflow.slide.qc
==================

This module contains functions for slide-level quality control, including Otsu's thresholding and Gaussian blur filtering. Quality control methods are used by passing a list of callables to the ``qc`` argument of ``.extract_tiles()``. They can also be directly applied to a slide with :meth:`slideflow.WSI.qc`.

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

.. autoclass:: Otsu

.. autoclass:: Gaussian

.. autoclass:: Save

.. autoclass:: Load

.. autoclass:: StridedDL