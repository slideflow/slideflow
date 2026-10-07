"""Check raw-tile LoRA training with a small timm transformer."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import timm
    from slideflow.mil.models import NNMIL
    from slideflow.mil.train._lora import encode_tiles, predict_lora, train_lora
    from slideflow.model.extractors._lora import LoRAQKV, adapter_state_dict, apply_lora, init_lora
    has_dependencies = True
except ImportError:
    has_dependencies = False


@unittest.skipUnless(has_dependencies, 'PyTorch and timm required')
class TestLoraTraining(unittest.TestCase):

    def model(self):
        return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0,
                                 img_size=32, patch_size=8, embed_dim=48, depth=4,
                                 num_heads=4, reg_tokens=4, init_values=1e-5)

    def test_zero_init_and_adapter_roundtrip(self):
        torch.manual_seed(3)
        model = self.model().eval()
        x = torch.randn(2, 3, 32, 32)
        reference = encode_tiles(model, x)
        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        init_lora(model, first_block=2, rank=4, alpha=8)
        model.eval()
        torch.testing.assert_close(encode_tiles(model, x), reference, atol=1e-6, rtol=1e-6)
        self.assertTrue(all(p.requires_grad == ('attn.qkv.A' in n or 'attn.qkv.B' in n)
                            for n, p in model.named_parameters()))
        with self.assertRaises(ValueError):
            init_lora(model, first_block=2, rank=4)
        state = adapter_state_dict(model, first_block=2)
        other = self.model().eval()
        other.load_state_dict(base_state)
        apply_lora(other, state, first_block=2, rank=4, alpha=8)
        torch.testing.assert_close(encode_tiles(model, x), encode_tiles(other, x), atol=1e-6, rtol=1e-6)

    def test_training_and_frozen_control(self):
        torch.manual_seed(4)
        tiles = torch.randint(0, 255, (8, 3, 32, 32, 3), dtype=torch.uint8)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        batches = DataLoader(TensorDataset(tiles, labels), batch_size=2, shuffle=False)

        class Extractor:
            tag = 'hoptimus0'
            transform = staticmethod(lambda x: x / 255)

        extractor = Extractor()
        extractor.model = self.model()
        head = NNMIL(48, 1, hidden_dim=16, dropout_p=0)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'lora'
            history = train_lora(extractor, head, batches, val_batches=batches,
                                 epochs=2, first_block=2, rank=4, alpha=8,
                                 device='cpu', outdir=out)
            self.assertEqual(len(history), 2)
            self.assertTrue(all('val_loss' in row for row in history))
            self.assertTrue((out/'adapters.pt').exists())
            state = torch.load(out/'adapters.pt', weights_only=True)
            self.assertTrue(any(state[k].abs().sum() > 0 for k in state if k.endswith('.Bq')))
            fresh = self.model()
            apply_lora(fresh, str(out/'adapters.pt'), first_block=2, rank=4, alpha=8)
            self.assertTrue(isinstance(fresh.blocks[2].attn.qkv, LoRAQKV))

            control = Extractor()
            control.model = self.model()
            train_lora(control, NNMIL(48, 1, hidden_dim=16), batches,
                       epochs=1, adapt=False, device='cpu', outdir=Path(folder)/'control')
            self.assertFalse((Path(folder)/'control'/'adapters.pt').exists())

    def test_joint_rs_training(self):
        torch.manual_seed(5)
        tiles = torch.randint(0, 255, (8, 3, 32, 32, 3), dtype=torch.uint8)
        rs = torch.tensor([10., 32., 12., 35., 15., 28., 18., 30.])
        labels = (rs >= 26).long()
        batches = DataLoader(TensorDataset(tiles, labels, rs), batch_size=2)

        class Extractor:
            tag = 'hoptimus0'
            transform = staticmethod(lambda x: x / 255)

        extractor = Extractor()
        extractor.model = self.model()
        head = NNMIL(48, 3, hidden_dim=16)
        history = train_lora(extractor, head, batches, objective='joint',
                             positive_weight=1., epochs=1, first_block=2,
                             rank=4, alpha=8, device='cpu')
        self.assertTrue(torch.isfinite(torch.tensor(history[0]['train_loss'])))
        predictions = predict_lora(extractor, head, batches, objective='joint')
        self.assertEqual(len(predictions['rs_pred']), len(rs))
        self.assertEqual(predictions['labels'], labels.tolist())


@unittest.skipUnless(has_dependencies, 'PyTorch and timm required')
class TestLoraReproducibility(unittest.TestCase):

    def test_joint_weight_dtype_and_adapter_seed(self):
        import copy
        import numpy as np
        torch.manual_seed(9)
        base = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0,
                                img_size=16, patch_size=8, embed_dim=16, depth=2,
                                num_heads=2)
        original_head = NNMIL(16, 3, hidden_dim=8)
        tiles = torch.randint(0, 255, (4, 2, 16, 16, 3), dtype=torch.uint8)
        rs = torch.tensor([10., 30., 15., 35.])
        batches = DataLoader(TensorDataset(tiles, (rs >= 26).long(), rs), batch_size=2)
        states = []
        for _ in range(2):
            class Extractor:
                tag = 'tiny_vit'
                transform = staticmethod(lambda x: x / 255)
            extractor = Extractor()
            extractor.model = copy.deepcopy(base)
            history = train_lora(extractor, copy.deepcopy(original_head), batches,
                                 objective='joint', positive_weight=np.float64(1), epochs=1,
                                 first_block=1, rank=2, alpha=4, seed=123, device='cpu')
            self.assertTrue(torch.isfinite(torch.tensor(history[0]['train_loss'])))
            states.append(adapter_state_dict(extractor.model))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
