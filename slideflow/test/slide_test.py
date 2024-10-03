import multiprocessing as mp
import unittest

import numpy as np
import slideflow as sf
from PIL import Image
from slideflow import errors


class TestSlide(unittest.TestCase):
    def __init__(self, testname, path, tile_px=71):
        super().__init__(testname)
        self.wsi_path = path
        self.tile_px = tile_px
        self.kw = dict(tile_px=tile_px, tile_um=604)
        self.wsi = sf.WSI(self.wsi_path, roi_method='ignore', **self.kw)

    def _assert_is_image(self, img: np.ndarray):
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertTrue(img.shape == (self.tile_px, self.tile_px, 3))
        self.assertTrue(img.dtype == np.uint8)

    def _assert_is_pil(self, img: Image):
        self.assertTrue(isinstance(img, Image.Image))

    def test_load(self):
        self.assertTrue(self.wsi.estimated_num_tiles > 0)
        self.assertTrue(len(self.wsi.shape) == 2)
        self.assertTrue(all(s > 0 for s in self.wsi.shape))
        self.assertTrue(self.wsi.shape == self.wsi.grid.shape)
        self.assertTrue(self.wsi.grid.sum() == (self.wsi.shape[0] * self.wsi.shape[1]))
        self.assertTrue(len(self.wsi.dimensions) == 2)

    def test_qc(self):
        qc_wsi = sf.WSI(self.wsi_path, roi_method='ignore', **self.kw)
        qc_wsi.qc('both')

    def test_index(self):
        self._assert_is_image(self.wsi[0, 0])
        self._assert_is_image(self.wsi[self.wsi.shape[0]-1,
                                       self.wsi.shape[1]-1])
        with self.assertRaises(IndexError):
            self.wsi[self.wsi.shape[0], 0]
        with self.assertRaises(IndexError):
            self.wsi[0, self.wsi.shape[1]]

    def test_stride(self):
        wsi_stride2 = sf.WSI(self.wsi_path, roi_method='ignore', stride_div=2, **self.kw)
        self.assertTrue(abs(wsi_stride2.shape[0] - self.wsi.shape[0] * 2) <= 1)
        self.assertTrue(abs(wsi_stride2.shape[1] - self.wsi.shape[1] * 2) <= 1)

    def test_raises_roi_error(self):
        with self.assertRaises(errors.MissingROIError):
            sf.WSI(self.wsi_path, roi_method='inside', **self.kw)
        with self.assertRaises(errors.MissingROIError):
            sf.WSI(self.wsi_path, roi_method='outside', **self.kw)

    def test_thumb(self):
        self._assert_is_pil(self.wsi.thumb(mpp=4, low_res=True))
        self._assert_is_pil(self.wsi.thumb(width=100, low_res=True))
        self._assert_is_pil(self.wsi.thumb(mpp=4, rois=True, low_res=True))
        self._assert_is_pil(self.wsi.square_thumb(width=100))
        with self.assertRaises(ValueError):
            self.wsi.thumb(mpp=4, width=100, low_res=True)

    def test_preview(self):
        pool = mp.dummy.Pool(8)
        self._assert_is_pil(self.wsi.preview(show_progress=False, pool=pool))
        pool.close()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
