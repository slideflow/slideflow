"""Check nnMIL registration, checkpoint inference and extractor reconstruction."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import slideflow as sf
import slideflow.mil
from timm.models.vision_transformer import VisionTransformer

from slideflow.mil.models import NNMIL
from slideflow.mil.data import StratifiedShuffle
from slideflow.model.extractors._factory import build_extractor_from_cfg
from slideflow.model.extractors._lora import adapter_state_dict, apply_lora, init_lora


class TestNNMILIntegration(unittest.TestCase):

    def test_config_and_checkpoint_roundtrip(self):
        config = sf.mil.mil_config('nnmil', aggregation_level='patient',
                                  balanced_batches=False, n_strata=3,
                                  model_kwargs={'hidden_dim': 8})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            config_path = path/'mil_params.json'
            config_path.write_text(json.dumps(config.json_dump()))
            restored = sf.mil.load_mil_config(str(config_path), strict=True)
            self.assertEqual(restored.aggregation_level, 'patient')
            self.assertFalse(restored.model_config.balanced_batches)
            self.assertEqual(restored.model_config.n_strata, 3)
            model = config.build_model(16, 2).eval()
            weights = path/'head.pt'
            torch.save(model.state_dict(), weights)
            loaded, _ = sf.mil.load_model_weights(str(weights), config=restored,
                                                 input_shape=16, output_shape=2)
            loaded = loaded.cpu()
            bags = torch.randn(2, 5, 16)
            lens = torch.tensor([3, 5])
            torch.testing.assert_close(model(bags, lens), loaded(bags, lens))
            scores, attention, _ = restored.batched_predict(loaded, bags, attention=True, device='cpu')
            torch.testing.assert_close(scores, model(bags, lens=None).softmax(-1))
            self.assertTrue(torch.isfinite(attention).all())

    def test_uq_classification_and_regression(self):
        bags = torch.randn(2, 5, 16)
        for loss, n_out in [('cross_entropy', 2), ('mse', 1)]:
            config = sf.mil.mil_config('nnmil', loss=loss, model_kwargs={'hidden_dim': 8})
            model = config.build_model(16, n_out).eval()
            expected, std = model(bags, uq=True, uq_softmax=loss == 'cross_entropy')
            actual, _, actual_std = config.batched_predict(model, bags, uq=True, device='cpu')
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_std, std)
            if loss == 'mse':
                self.assertTrue(torch.any(actual_std > 0))

    def test_strata_edge_cases(self):
        self.assertEqual(StratifiedShuffle(np.array([]))([]), [])
        order = StratifiedShuffle(np.array([0, 0, 1, 1]))([1, 3])
        self.assertEqual(sorted(order), [1, 3])
        with self.assertRaises(ValueError):
            sf.mil.mil_config('nnmil', n_strata=0)
        config = sf.mil.mil_config('nnmil', loss='mse').model_config
        with self.assertRaises(ValueError):
            config._strata(np.array([1., np.nan]))
        self.assertEqual(len(np.unique(config._strata(np.ones(8)))), 1)

    def test_existing_model_config_unchanged(self):
        config = sf.mil.mil_config('attention_mil')
        self.assertIs(type(config.model_config), sf.mil.MILModelConfig)
        self.assertNotIn('balanced_batches', config.to_dict())

    def test_extractor_roundtrip(self):
        torch.manual_seed(3)
        base = VisionTransformer(img_size=224, patch_size=56, embed_dim=16,
                                 depth=2, num_heads=2, num_classes=0,
                                 reg_tokens=4, init_values=1e-5).eval()
        adapted = copy.deepcopy(base)
        init_lora(adapted, first_block=1, rank=2, alpha=4)
        with torch.no_grad():
            adapted.blocks[1].attn.qkv.Bq.normal_(std=.1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            adapters = path/'adapters.pt'
            torch.save(adapter_state_dict(adapted), adapters)
            for name in ['hoptimus0', 'mettle']:
                weights = path/f'{name}.pt'
                state = base.state_dict()
                if name == 'mettle':
                    state = dict(state)
                    state['head.weight'] = torch.zeros(2, 16)
                    state = {'state_dict': state}
                torch.save(state, weights)
                with patch('timm.create_model', side_effect=lambda *a, **k: copy.deepcopy(base)):
                    extractor = sf.build_feature_extractor(name, weights=weights, lora=adapters,
                        lora_first_block=1, lora_rank=2, lora_alpha=4,
                        device='cpu', mixed_precision=False, channels_last=False)
                    config = json.loads(json.dumps(extractor.dump_config()))
                    restored = build_extractor_from_cfg(config, device='cpu',
                                                       mixed_precision=False, channels_last=False)
                    x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
                    torch.testing.assert_close(extractor(x), restored(x))
                    self.assertEqual(extractor.preprocess_kwargs, {'standardize': False})

    def test_bad_adapter_does_not_mutate_encoder(self):
        base = VisionTransformer(img_size=16, patch_size=8, embed_dim=16,
                                 depth=2, num_heads=2, num_classes=0).double().eval()
        adapted = copy.deepcopy(base)
        init_lora(adapted, first_block=1, rank=2)
        self.assertEqual(adapted.blocks[1].attn.qkv.Aq.dtype, torch.float64)
        state = adapter_state_dict(adapted)
        state['0.attn.qkv.Aq'] = torch.zeros(1, 16)
        keys = list(base.state_dict())
        with self.assertRaises(ValueError):
            apply_lora(base, state, first_block=1, rank=2)
        self.assertEqual(list(base.state_dict()), keys)
        self.assertTrue(all(p.requires_grad for p in base.parameters()))


if __name__ == '__main__':
    unittest.main()
