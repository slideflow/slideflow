.. currentmodule:: slideflow.norm

slideflow.norm
===============

.. automodule:: slideflow.norm
   :members:

StainNormalizer
---------------

.. autoclass:: StainNormalizer
.. autofunction:: slideflow.norm.StainNormalizer.fit
.. autofunction:: slideflow.norm.StainNormalizer.get_fit
.. autofunction:: slideflow.norm.StainNormalizer.set_fit
.. autofunction:: slideflow.norm.StainNormalizer.transform
.. autofunction:: slideflow.norm.StainNormalizer.jpeg_to_jpeg
.. autofunction:: slideflow.norm.StainNormalizer.jpeg_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.png_to_png
.. autofunction:: slideflow.norm.StainNormalizer.png_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.rgb_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.tf_to_rgb
.. autofunction:: slideflow.norm.StainNormalizer.tf_to_tf
.. autofunction:: slideflow.norm.StainNormalizer.torch_to_torch

Example images
--------------

.. figure:: norm_compare/wsi_norm_compare.jpg

    Comparison of normalizers applied to a whole-slide image.

.. figure:: norm_compare/tile_norm_compare.jpg

    Comparison of normalizers applied to an image tile.

.. figure:: norm_compare/wsi_unnormalized.jpg

    Unnormalized whole-slide images.

.. figure:: norm_compare/wsi_reinhard_v1.jpg

    Whole-slide images normalized with **Reinhard**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_reinhard_v2.jpg

    Whole-slide images normalized with **Reinhard**, fit to preset "v2"

.. figure:: norm_compare/wsi_macenko_v1.jpg

    Whole-slide images normalized with **Macenko**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_macenko_v2.jpg

    Whole-slide images normalized with **Macenko**, fit to preset "v2"

.. figure:: norm_compare/wsi_vahadane_v1.jpg

    Whole-slide images normalized with **Vahadane**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_vahadane_v2.jpg

    Whole-slide images normalized with **Vahadane**, fit to preset "v2"

.. figure:: norm_compare/wsi_vahadane_spams_v1.jpg

    Whole-slide images normalized with **Vahadane (SPAMS)**, fit to preset "v1" (default)

.. figure:: norm_compare/wsi_vahadane_spams_v2.jpg

    Whole-slide images normalized with **Vahadane (SPAMS)**, fit to preset "v2"

.. figure:: norm_compare/tile_unnormalized.jpg

    Unnormalized image tiles.

.. figure:: norm_compare/tile_reinhard_v1.jpg

    Image tiles normalized with **Reinhard Mask**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_reinhard_v2.jpg

    Image tiles normalized with **Reinhard Mask**, fit to preset "v2"

.. figure:: norm_compare/tile_macenko_v1.jpg

    Image tiles normalized with **Macenko**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_macenko_v2.jpg

    Image tiles normalized with **Macenko**, fit to preset "v2"

.. figure:: norm_compare/tile_vahadane_v1.jpg

    Image tiles normalized with **Vahadane**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_vahadane_v2.jpg

    Image tiles normalized with **Vahadane**, fit to preset "v2"

.. figure:: norm_compare/tile_vahadane_spams_v1.jpg

    Image tiles normalized with **Vahadane (SPAMS)**, fit to preset "v1" (default)

.. figure:: norm_compare/tile_vahadane_spams_v2.jpg

    Image tiles normalized with **Vahadane (SPAMS)**, fit to preset "v2"