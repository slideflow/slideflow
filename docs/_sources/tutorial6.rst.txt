.. currentmodule:: slideflow.slide

.. _tutorial6:

Tutorial 6: Custom slide filtering
==================================

In this brief tutorial, we'll take a look at how you can implement and preview bespoke slide-level filtering methods.

The slide-level filtering (QC) methods Slideflow currently supports include Otsu's thresholding and Gaussian blur filtering, which can be applied to a :class:`WSI` object with :meth:`WSI.qc`. If you have a custom filtering algorithm you would like to apply to a slide, you can now use :meth:`WSI.apply_qc_mask()` to apply a boolean mask to filter a slide.

For the purposes of this tutorial, we will generate a boolean mask using the already-available Otsu's thresholding algorithm, but you can replace this with whatever masking algorithm you like.

First, we'll load a slide:

.. code-block:: python

    import numpy as np
    import slideflow as sf

    wsi = sf.WSI('slide.svs', tile_px=299, tile_um=302)

Next, we'll apply Otsu's thresholding to get the boolean mask we'll use in subsequent steps, then remove the QC once we have the mask:

.. code-block:: python

    wsi.qc('otsu')
    qc_mask = np.copy(wsi.qc_mask)
    wsi.remove_qc()

Our mask should have two dimensions (y, x) and have a dtype of bool:

.. code-block:: bash

    >>> qc_mask.shape
    (1010, 2847)
    >>> qc_mask.dtype
    dtype('bool')

Our :class:`WSI` object now has no QC applied. We can manually apply this boolean mask with :meth:`WSI.apply_qc_mask()`:

.. code-block:: python

    wsi.apply_qc_mask(qc_mask)

And that's it! We can preview how our mask affects tile filtering by using :meth:`WSI.preview()`:

.. code-block:: python

    wsi.preview().show()
