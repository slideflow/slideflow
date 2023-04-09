import unittest
import sys
import numpy as np
import slideflow as sf
from packaging import version
from parameterized import parameterized
from slideflow.util import log

try:
    import tensorflow as tf
    if version.parse(tf.__version__) < version.parse("2.0"):
        raise ImportError

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    from slideflow.model.tensorflow import ModelParams as TFModelParams
    from slideflow.model.tensorflow import Features as TFFeatures
    tf_models = list(TFModelParams.ModelDict.keys())
except ImportError:
    tf_models = []
    pass

try:
    import torch
    from slideflow.model.torch import ModelParams as TorchModelParams
    from slideflow.model.torch import Features as TorchFeatures
    torch_models = list(TorchModelParams.ModelDict.keys())
except ImportError:
    torch_models = []
    pass

# -----------------------------------------------------------------------------

class TestSlide(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        px = 299
        cls._orig_logging_level = sf.getLoggingLevel()  # type: ignore
        sf.setLoggingLevel(40)
        if 'tensorflow' in sys.modules:
            cls.img = tf.convert_to_tensor(np.random.random((2, px, px, 3)))
            cls.nasnet_img = tf.convert_to_tensor(np.random.random((2, 331, 331, 3)))
        if 'torch' in sys.modules:
            cls.img = torch.from_numpy(np.random.random((2, 3, px, px))).to(torch.float32)
            cls.nasnet_img = torch.from_numpy(np.random.random((2, 3, 331, 331))).to(torch.float32)

    @classmethod
    def tearDownClass(cls) -> None:
        sf.setLoggingLevel(cls._orig_logging_level)  # type: ignore
        return super().tearDownClass()

    def _get_tensorflow_px(self, arch, include_top):
        if 'nasnet' in arch.lower():
            return 331
        elif include_top:
            px_dict = {
                'resnet50': 224,
                'resnet101': 224,
                'resnet152': 224,
                'resnet50_v2': 224,
                'resnet101_v2': 224,
                'resnet152_v2': 224,
                'vgg16': 224,
                'vgg19': 224,
                'mobilenet': 224,
                'mobilenet_v2': 224,
                'efficientnet_v2b0': 224,
                'efficientnet_v2b1': 240,
                'efficientnet_v2b2': 260,
                'efficientnet_v2b3': 300,
                'efficientnet_v2s': 384,
                'efficientnet_v2m': 480,
                'efficientnet_v2l': 480,
                'densenet_121': 224,
                'densenet_169': 224,
                'densenet_201': 224
            }
            if arch in px_dict:
                return px_dict[arch]
            else:
                return 299
        else:
            return 299

    def _get_torch_px(self, arch):
        if 'nasnet' in arch.lower():
            return 331
        else:
            return 299

    def _test_arch_tensorflow(self, arch, stage, include_top=False):
        assert stage in ('train', 'eval')
        px = self._get_tensorflow_px(arch=arch, include_top=include_top)

        log.info("Testing {} (stage={}, top={}, px={}, backend={})".format(
            arch,
            stage,
            include_top,
            px,
            'tensorflow'
        ))

        # Test forward pass of single image
        img = tf.convert_to_tensor(np.random.random((2, px, px, 3)))
        hp = TFModelParams(
            tile_px=px,
            tile_um=302,
            model=arch,
            include_top=include_top,
            hidden_layers=0
        )
        model = hp.build_model(num_classes=1, pretrain='imagenet')
        res = model(img, training=(stage=='train'))
        self.assertIsInstance(res, tf.Tensor)
        self.assertTrue(res.shape == (2, 1))

        # Test features interface
        if 'nasnet' not in arch.lower():
            interface = TFFeatures.from_model(model, layers=['postconv'])
            res = interface(img)
            if not isinstance(res, list):
                res = [res]
            self.assertTrue(len(res) == 1)
            self.assertIsInstance(res[0], tf.Tensor)
            self.assertTrue(len(res[0].shape) > 1)
            self.assertTrue(res[0].shape[0] == 2)

        # Cleanup
        del model

    def _test_arch_torch(self, arch, stage, include_top=False):
        assert stage in ('train', 'eval')
        px = self._get_torch_px(arch=arch)

        log.info("Testing {} (stage={}, top={}, px={}, backend={})".format(
            arch,
            stage,
            include_top,
            px,
            'torch'
        ))

        # Test forward pass of single image
        img = torch.from_numpy(np.random.random((2, 3, px, px))).to(torch.float32)
        hp = TorchModelParams(
            tile_px=px,
            tile_um=302,
            model=arch,
            include_top=include_top,
            hidden_layers=0
        )
        model = hp.build_model(num_classes=1, pretrain='imagenet')
        if stage == 'eval':
            model.eval()
        res = model(img)
        self.assertIsInstance(res, torch.Tensor)
        self.assertTrue(res.shape == (2, 1))

        # Test features interface
        if 'nasnet' not in arch.lower():
            interface = TorchFeatures.from_model(model, layers=['postconv'], tile_px=px)
            res = interface(img)
            if not isinstance(res, list):
                res = [res]
            self.assertTrue(len(res) == 1)
            self.assertIsInstance(res[0], torch.Tensor)
            self.assertTrue(len(res[0].shape) > 1)
            self.assertTrue(res[0].shape[0] == 2)

        # Cleanup
        del model

    @unittest.skipIf('tensorflow' not in sys.modules, "Tensorflow not installed")
    @parameterized.expand(tf_models, skip_on_empty=True)
    def test_all_tensorflow_arch_training_notop(self, arch):
        self._test_arch_tensorflow(arch, 'train', include_top=False)

    @unittest.skipIf('tensorflow' not in sys.modules, "Tensorflow not installed")
    @parameterized.expand(tf_models, skip_on_empty=True)
    def test_all_tensorflow_arch_training_withtop(self, arch):
        self._test_arch_tensorflow(arch, 'train', include_top=True)

    @unittest.skipIf('tensorflow' not in sys.modules, "Tensorflow not installed")
    @parameterized.expand(tf_models, skip_on_empty=True)
    def test_all_tensorflow_arch_eval_notop(self, arch):
        self._test_arch_tensorflow(arch, 'eval', include_top=False)

    @unittest.skipIf('tensorflow' not in sys.modules, "Tensorflow not installed")
    @parameterized.expand(tf_models, skip_on_empty=True)
    def test_all_tensorflow_arch_eval_withtop(self, arch):
        self._test_arch_tensorflow(arch, 'eval', include_top=True)

    @unittest.skipIf('torch' not in sys.modules, "PyTorch not installed")
    @parameterized.expand(torch_models, skip_on_empty=True)
    def test_all_torch_arch_training_notop(self, arch):
        self._test_arch_torch(arch, 'train', include_top=False)

    @unittest.skipIf('torch' not in sys.modules, "PyTorch not installed")
    @parameterized.expand(torch_models, skip_on_empty=True)
    def test_all_torch_arch_training_withtop(self, arch):
        self._test_arch_torch(arch, 'train', include_top=True)

    @unittest.skipIf('torch' not in sys.modules, "PyTorch not installed")
    @parameterized.expand(torch_models, skip_on_empty=True)
    def test_all_torch_arch_eval_notop(self, arch):
        self._test_arch_torch(arch, 'eval', include_top=False)

    @unittest.skipIf('torch' not in sys.modules, "PyTorch not installed")
    @parameterized.expand(torch_models, skip_on_empty=True)
    def test_all_torch_arch_eval_withtop(self, arch):
        self._test_arch_torch(arch, 'eval', include_top=True)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
