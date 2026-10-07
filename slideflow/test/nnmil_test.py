import tempfile
import unittest

import numpy as np

try:
    import torch
    import slideflow.mil
    from slideflow.mil.models import NNMIL
    from slideflow.mil.data import StratifiedShuffle
    from slideflow.mil.train import _fastai
    has_torch = True
except ImportError:
    has_torch = False

# -----------------------------------------------------------------------------

@unittest.skipIf(not has_torch, "PyTorch not installed")
class TestNNMIL(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_registered(self):
        self.assertIn('nnmil', slideflow.mil.list_models())
        config = slideflow.mil.mil_config('nnmil')
        self.assertIsInstance(config.model_config, slideflow.mil.NNMILModelConfig)
        self.assertTrue(config.model_config.use_lens)

    def test_eval_subsets(self):
        model = NNMIL(1536, 2)
        subsets = model.eval_subsets
        # 256-dim windows every 64 dims over 1536 dims: starts 0, 64, ..., 1280
        self.assertEqual(tuple(subsets.shape), (21, 256))
        self.assertEqual(len(torch.unique(subsets)), 1536)
        # Fewer features than hidden_dim: one subset with every feature.
        self.assertEqual(tuple(NNMIL(64, 2).eval_subsets.shape), (1, 64))

    def test_forward_shapes(self):
        model = NNMIL(128, 3, hidden_dim=32)
        bags, lens = torch.randn(4, 50, 128), torch.full((4,), 50)
        model.train()
        self.assertEqual(tuple(model(bags, lens).shape), (4, 3))
        model.eval()
        with torch.no_grad():
            out, att = model(bags, lens, return_attention=True)
            mean, std = model(bags, lens, uq=True)
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertEqual(tuple(att.shape), (4, 50, 1))
        self.assertEqual(tuple(mean.shape), (4, 3))
        self.assertEqual(tuple(std.shape), (4, 3))

    def test_eval_is_deterministic(self):
        model = NNMIL(128, 2, hidden_dim=32).eval()
        bags = torch.randn(2, 30, 128)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(bags), model(bags)))

    def test_padding_is_ignored(self):
        model = NNMIL(128, 2, hidden_dim=32).eval()
        bag = torch.randn(1, 20, 128)
        padded = torch.cat([bag, torch.zeros(1, 12, 128)], dim=1)
        with torch.no_grad():
            ref = model(bag, torch.tensor([20]))
            out, att = model(padded, torch.tensor([20]), return_attention=True)
        self.assertTrue(torch.allclose(ref, out, atol=1e-6))
        self.assertTrue(torch.all(att[0, 20:] == 0))
        self.assertAlmostEqual(float(att[0, :20].sum()), 1.0, places=5)

    def test_calculate_attention(self):
        model = NNMIL(128, 2, hidden_dim=32).eval()
        bags, lens = torch.randn(3, 25, 128), torch.tensor([25, 10, 1])
        with torch.no_grad():
            _, att = model(bags, lens, return_attention=True)
            self.assertTrue(torch.allclose(att, model.calculate_attention(bags, lens)))

    def test_stratified_shuffle(self):
        strata = np.array([1] * 150 + [0] * 850)
        order = StratifiedShuffle(strata)(list(range(1000)))
        self.assertEqual(sorted(order), list(range(1000)))
        per_batch = [strata[order[i:i + 32]].sum() for i in range(0, 992, 32)]
        # expected 32 * 0.15 = 4.8 positives per batch
        self.assertTrue(all(3 <= n <= 7 for n in per_batch), per_batch)

    def test_regression_strata(self):
        config = slideflow.mil.mil_config('nnmil', loss='mse').model_config
        strata = config._strata(np.linspace(0, 1, 100))
        self.assertEqual(np.bincount(strata).tolist(), [25, 25, 25, 25])

    def test_train(self):
        n, n_feats = 120, 64
        y = np.array(['a'] * 90 + ['b'] * 30)
        bags = np.empty(n, dtype=object)
        for i in range(n):
            bag = torch.randn(np.random.randint(20, 60), n_feats)
            if y[i] == 'b':
                bag[:5, :8] += 2.0
            bags[i] = bag
        idx = np.random.permutation(n)
        config = slideflow.mil.mil_config(
            'nnmil', lr=1e-3, epochs=30, batch_size=16, bag_size=32,
            model_kwargs={'hidden_dim': 32}
        )
        with tempfile.TemporaryDirectory() as outdir:
            learner, (n_in, n_out) = _fastai.build_learner(
                config, bags, y, idx[:90], idx[90:], np.unique(y),
                outdir=outdir, device='cpu'
            )
            self.assertEqual((n_in, n_out), (n_feats, 2))
            _fastai.train(learner, config)
            model = learner.model.eval()
            with torch.no_grad():
                scores = [
                    torch.softmax(model(bags[i].unsqueeze(0), torch.tensor([len(bags[i])])), 1)[0, 1].item()
                    for i in idx[90:]
                ]
        from sklearn.metrics import roc_auc_score
        self.assertGreater(roc_auc_score(y[idx[90:]] == 'b', scores), 0.9)


if __name__ == '__main__':
    unittest.main()
