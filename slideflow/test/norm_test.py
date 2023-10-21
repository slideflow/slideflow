import unittest

import importlib.util
import sys
import os
import numpy as np
import slideflow as sf
from io import BytesIO
from PIL import Image
import cv2

try:
    import tensorflow as tf
    import slideflow.norm.tensorflow as tf_norm
except ImportError:
    tf_norm = None

try:
    import torch
    import slideflow.norm.torch as torch_norm
except ImportError:
    torch_norm = None


spams_loader = importlib.util.find_spec('spams')

class TestSlide(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.px = 71  # type: ignore
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)
        float_img = np.random.random((cls.px, cls.px, 3))  # type: ignore
        cls.img = (float_img * 255).clip(0, 255).astype(np.uint8)  # type: ignore
        with BytesIO() as output:
            Image.fromarray(cls.img).save(  # type: ignore
                output,
                format="JPEG",
                quality=100
            )
            cls.jpg = output.getvalue()  # type: ignore
        with BytesIO() as output:
            Image.fromarray(cls.img).save(output, format="PNG")  # type: ignore
            cls.png = output.getvalue()  # type: ignore
        if 'tensorflow' in sys.modules:
            cls.tf = tf.convert_to_tensor(cls.img)  # type: ignore
        if 'torch' in sys.modules:
            cls.torch = torch.from_numpy(cls.img)  # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore
        return super().tearDownClass()

    def _assert_valid_reinhard_fit(self, fit):
        self.assertIn('target_means', fit)
        self.assertIn('target_stds', fit)
        self.assertIsInstance(fit['target_means'], np.ndarray)
        self.assertIsInstance(fit['target_stds'], np.ndarray)
        self.assertEqual(fit['target_means'].shape, (3,))
        self.assertEqual(fit['target_stds'].shape, (3,))

    def _assert_valid_macenko_fit(self, fit):
        self.assertIn('stain_matrix_target', fit)
        self.assertIn('target_concentrations', fit)
        self.assertIsInstance(fit['stain_matrix_target'], np.ndarray)
        self.assertIsInstance(fit['target_concentrations'], np.ndarray)
        self.assertEqual(fit['stain_matrix_target'].shape, (3, 2))
        self.assertEqual(fit['target_concentrations'].shape, (2,))

    def _assert_valid_vahadane_fit(self, fit):
        self.assertIn('stain_matrix_target', fit)
        self.assertIsInstance(fit['stain_matrix_target'], np.ndarray)
        self.assertEqual(fit['stain_matrix_target'].shape, (2, 3))

    def _assert_valid_jpg(self, jpg):
        self.assertIsInstance(jpg, (str, bytes))
        cv_image = cv2.imdecode(
            np.fromstring(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        self.assertEqual(cv_image.shape, (self.px, self.px, 3))

    def _assert_valid_png(self, png):
        self.assertIsInstance(png, (str, bytes))
        cv_image = cv2.imdecode(
            np.fromstring(png, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        self.assertEqual(cv_image.shape, (self.px, self.px, 3))

    def _assert_valid_numpy(self, img):
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (self.px, self.px, 3))

    def _assert_valid_tf(self, img):
        self.assertIsInstance(img, tf.Tensor)
        self.assertEqual(img.shape, (self.px, self.px, 3))

    def _assert_valid_torch(self, img):
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (self.px, self.px, 3))

    def _test_reinhard_fit_to_numpy(self, norm):
        norm.fit(self.img)
        self._assert_valid_reinhard_fit(norm.get_fit())

    def _test_reinhard_fit_to_path(self, norm):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(pkg_dir, '../norm/norm_tile.jpg')
        norm.fit(img_path)
        self._assert_valid_reinhard_fit(norm.get_fit())

    def _test_reinhard_set_fit(self, norm):
        tgt_means = np.random.random((3,))
        tgt_stds = np.random.random((3,))
        norm.set_fit(target_means=tgt_means, target_stds=tgt_stds)
        fit = norm.get_fit()
        self._assert_valid_reinhard_fit(fit)
        self.assertTrue(np.allclose(tgt_means, fit['target_means']))
        self.assertTrue(np.allclose(tgt_stds, fit['target_stds']))

    def _test_macenko_fit_to_numpy(self, norm):
        norm.fit(self.img)
        self._assert_valid_macenko_fit(norm.get_fit())

    def _test_macenko_fit_to_path(self, norm):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(pkg_dir, '../norm/norm_tile.jpg')
        norm.fit(img_path)
        self._assert_valid_macenko_fit(norm.get_fit())

    def _test_macenko_set_fit(self, norm):
        HE = np.random.random((3, 2))
        C = np.random.random((2,))
        norm.set_fit(stain_matrix_target=HE, target_concentrations=C)
        fit = norm.get_fit()
        self._assert_valid_macenko_fit(fit)
        self.assertTrue(np.allclose(HE, fit['stain_matrix_target']))
        self.assertTrue(np.allclose(C, fit['target_concentrations']))

    def _test_vahadane_fit_to_numpy(self, norm):
        norm.fit(self.img)
        self._assert_valid_vahadane_fit(norm.get_fit())

    def _test_vahadane_fit_to_path(self, norm):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(pkg_dir, '../norm/norm_tile.jpg')
        norm.fit(img_path)
        self._assert_valid_vahadane_fit(norm.get_fit())

    def _test_vahadane_set_fit(self, norm):
        HE = np.random.random((2, 3))
        norm.set_fit(stain_matrix_target=HE)
        fit = norm.get_fit()
        self._assert_valid_vahadane_fit(fit)
        self.assertTrue(np.allclose(HE, fit['stain_matrix_target']))

    def _test_transform_numpy(self, norm):
        self._assert_valid_numpy(norm.transform(self.img))

    def _test_transform_tensorflow(self, norm):
        self._assert_valid_tf(norm.transform(self.tf))

    def _test_transform_torch(self, norm):
        self._assert_valid_torch(norm.transform(self.torch))

    def _test_jpeg_to_jpeg(self, norm):
        self._assert_valid_jpg(norm.jpeg_to_jpeg(self.jpg))

    def _test_png_to_png(self, norm):
        self._assert_valid_png(norm.png_to_png(self.png))

    def _test_jpeg_to_rgb(self, norm):
        self._assert_valid_numpy(norm.jpeg_to_rgb(self.jpg))

    def _test_png_to_rgb(self, norm):
        self._assert_valid_numpy(norm.png_to_rgb(self.png))

    def _test_rgb_to_rgb(self, norm):
        self._assert_valid_numpy(norm.rgb_to_rgb(self.img))

    def _test_torch_to_torch(self, norm):
        self._assert_valid_torch(norm.torch_to_torch(self.torch))

    def _test_tf_to_tf(self, norm):
        self._assert_valid_tf(norm.tf_to_tf(self.tf))

    def _test_tf_to_rgb(self, norm):
        self._assert_valid_numpy(norm.tf_to_rgb(self.tf))

    def _test_transforms(self, norm):
        self._test_transform_numpy(norm)
        self._test_jpeg_to_jpeg(norm)
        self._test_jpeg_to_rgb(norm)
        self._test_png_to_png(norm)
        self._test_png_to_rgb(norm)
        self._test_rgb_to_rgb(norm)

        if 'torch' in sys.modules:
            self._test_transform_torch(norm)
            self._test_torch_to_torch(norm)
        if 'tensorflow' in sys.modules:
            self._test_transform_tensorflow(norm)
            self._test_tf_to_rgb(norm)
            self._test_tf_to_tf(norm)

    def test_reinhard_numpy(self):
        norm = sf.norm.StainNormalizer('reinhard')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    def test_reinhard_fast_numpy(self):
        norm = sf.norm.StainNormalizer('reinhard_fast')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    def test_reinhard_mask_numpy(self):
        norm = sf.norm.StainNormalizer('reinhard_mask')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    def test_augment_numpy(self):
        norm = sf.norm.StainNormalizer('reinhard_mask')
        self._test_transforms(norm)

    def test_macenko_numpy(self):
        norm = sf.norm.StainNormalizer('macenko')
        self._test_transforms(norm)
        self._test_macenko_fit_to_numpy(norm)
        self._test_macenko_fit_to_path(norm)
        self._test_macenko_set_fit(norm)

    @unittest.skipIf(spams_loader is None, "SPAMS not installed")
    def test_vahadane_numpy(self):
        norm = sf.norm.StainNormalizer('vahadane')
        self._test_transforms(norm)
        self._test_vahadane_fit_to_numpy(norm)
        self._test_vahadane_fit_to_path(norm)
        self._test_vahadane_set_fit(norm)

    def test_vahadane_sklearn_numpy(self):
        norm = sf.norm.StainNormalizer('vahadane_sklearn')
        self._test_transforms(norm)
        self._test_vahadane_fit_to_numpy(norm)
        self._test_vahadane_fit_to_path(norm)
        self._test_vahadane_set_fit(norm)

    @unittest.skipIf(spams_loader is None, "SPAMS not installed")
    def test_vahadane_spams_numpy(self):
        norm = sf.norm.StainNormalizer('vahadane_spams')
        self._test_transforms(norm)
        self._test_vahadane_fit_to_numpy(norm)
        self._test_vahadane_fit_to_path(norm)
        self._test_vahadane_set_fit(norm)

    @unittest.skipIf(tf_norm is None, "Tensorflow not imported")
    def test_reinhard_tensorflow(self):
        norm = tf_norm.StainNormalizer('reinhard')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    @unittest.skipIf(tf_norm is None, "Tensorflow not imported")
    def test_reinhard_fast_tensorflow(self):
        norm = tf_norm.StainNormalizer('reinhard_fast')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    @unittest.skipIf(tf_norm is None, "Tensorflow not imported")
    def test_reinhard_mask_tensorflow(self):
        norm = tf_norm.StainNormalizer('reinhard_mask')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    @unittest.skipIf(tf_norm is None, "Tensorflow not imported")
    def test_reinhard_fast_mask_tensorflow(self):
        norm = tf_norm.StainNormalizer('reinhard_fast_mask')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    @unittest.skipIf(tf_norm is None, "Tensorflow not imported")
    def test_macenko_tensorflow(self):
        norm = tf_norm.StainNormalizer('macenko')
        self._test_transforms(norm)
        self._test_macenko_fit_to_numpy(norm)
        self._test_macenko_fit_to_path(norm)
        self._test_macenko_set_fit(norm)

    @unittest.skipIf(torch_norm is None, "Torch not imported")
    def test_reinhard_torch(self):
        norm = torch_norm.StainNormalizer('reinhard')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

    @unittest.skipIf(torch_norm is None, "Torch not imported")
    def test_reinhard_fast_torch(self):
        norm = torch_norm.StainNormalizer('reinhard_fast')
        self._test_transforms(norm)
        self._test_reinhard_fit_to_numpy(norm)
        self._test_reinhard_fit_to_path(norm)
        self._test_reinhard_set_fit(norm)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
