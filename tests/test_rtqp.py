"""Small regression checks for paper semantics and temporal leakage."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from models.config import ExperimentConfig, ablation_flags, split_times
from models.data import BatchCollator, inject_inputs, load_data
from models.diagnostics import binary_auc
from models.engine import construct_reliability, run_experiment
from models.model import RTQP, SDAE, TemporalEncoder, reliability


class RTQPRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = ExperimentConfig(rank=4, channels=4, hidden=16, sdae_width=8,
                                      epochs=1, pretrain_epochs=1, nmf_iterations=3,
                                      batch_size=16, max_train_samples=32, max_eval_samples=16,
                                      device="cpu", density=0.25)

    def test_paper_splits_and_case3(self):
        times = split_times(self.config)
        self.assertEqual(tuple(map(len, times.values())), (40, 11, 6))
        swept = split_times(replace(self.config, protocol="window_sweep", window=4))
        self.assertEqual(swept["train"][0], 10)
        self.assertEqual(swept["validation"], times["validation"])
        moving = split_times(replace(self.config, protocol="moving", task=2))
        self.assertEqual(moving["train"], tuple(range(23, 30)))
        self.assertEqual(moving["validation"], (30, 31))
        self.assertEqual(moving["test"], tuple(range(32, 48)))
        self.assertFalse(ablation_flags(3)["feature_enhancement"])
        self.assertTrue(ablation_flags(3)["enhanced_branch"])

    def test_nmf_latest_history_and_clean_labels(self):
        values = np.arange(64, dtype=np.float32)[:, None, None] + np.ones((64, 8, 12), np.float32)
        mask = np.ones_like(values, dtype=bool)
        visible, injected = inject_inputs(values, mask, 1.0, 0.5, 42)
        features = {"enhanced": visible.copy(), "confidence": np.ones_like(values),
                    "label_confidence": np.ones_like(values)}
        for name, count in (("original_user", 8), ("original_service", 12),
                            ("enhanced_user", 8), ("enhanced_service", 12)):
            features[name] = np.broadcast_to(np.arange(64)[:, None, None], (64, count, 4)).astype(np.float32)
        collate = BatchCollator(values, visible, mask, features, 7)
        batch = collate([(20, 3, 5)])
        self.assertEqual(batch["nmf_time"].item(), 19)
        self.assertTrue(torch.all(batch["nmf"] == 19))
        self.assertEqual(batch["target"].item(), values[20, 3, 5])
        self.assertTrue(np.all(injected))
        values[20:] = 99999
        visible[20:] = 99999
        after = collate([(20, 3, 5)])
        for name in ("x", "nmf"):
            torch.testing.assert_close(batch[name], after[name])

    def test_reliability_uses_no_future_or_batch_normalization(self):
        x = torch.tensor([[1., 2., 4.], [100., 50., 20.]])
        observed = torch.ones_like(x, dtype=torch.bool)
        scores, weights = reliability(x, torch.zeros_like(x), observed)
        single, _ = reliability(x[:1], torch.zeros_like(x[:1]), observed[:1])
        torch.testing.assert_close(scores[:1], single)
        self.assertEqual(weights[0, -1].item(), 0)
        values = np.random.default_rng(8).random((64, 8, 12)).astype(np.float32)
        mask = np.ones_like(values, dtype=bool)
        torch.manual_seed(2)
        sdae = SDAE(self.config).eval().requires_grad_(False)
        enhanced, confidence = np.zeros_like(values), np.zeros_like(values)
        construct_reliability(sdae, values, mask, 7, torch.device("cpu"), enhanced, confidence)
        changed = values.copy()
        changed[21:] = 10000
        enhanced2, confidence2 = np.zeros_like(values), np.zeros_like(values)
        construct_reliability(sdae, changed, mask, 7, torch.device("cpu"), enhanced2, confidence2)
        np.testing.assert_array_equal(enhanced[:21], enhanced2[:21])
        np.testing.assert_array_equal(confidence[:21], confidence2[:21])

    def test_confidence_attention_formula_and_missing_inputs(self):
        encoder = TemporalEncoder(self.config)
        x = torch.tensor([[1., 2., 3.]])
        mask = torch.ones_like(x, dtype=torch.bool)
        confidence = torch.tensor([[1., 0., 0.5]])
        _, plain = encoder(x, confidence, mask, weighted=False)
        feature, weighted = encoder(x, confidence, mask, weighted=True)
        expected = plain * confidence[:, None, None, :]
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(weighted, expected)
        self.assertTrue(torch.all(weighted[..., 1] == 0))
        self.assertTrue(torch.isfinite(feature).all())
        feature, weights = encoder(x * 0, confidence * 0, mask & False)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(feature == 0))

    def test_loss_denominator_is_number_of_samples(self):
        model = RTQP(self.config)
        model.forward = lambda batch: torch.tensor([2., 4.], requires_grad=True)
        batch = {"target": torch.zeros(2), "target_confidence": torch.tensor([1., 0.5])}
        self.assertEqual(model.loss(batch).item(), 6.)
        case6 = RTQP(replace(self.config, case=6))
        case6.forward = model.forward
        self.assertEqual(case6.loss(batch).item(), 10.)

    def test_index_base_and_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data.txt"
            path.write_text("0 0 0 1\n1 2 1 2\n", encoding="utf-8")
            x = load_data(path, Path(root) / "cache", (2, 2, 3))
            self.assertEqual(x[0, 0, 0], 1)
            self.assertEqual(x[1, 1, 2], 2)
            del x  # Release the Windows memory-mapped file before temp cleanup.
            path.write_text("1 1 1 1\n2 3 2 2\n", encoding="utf-8")
            x = load_data(path, Path(root) / "cache", (2, 2, 3))
            self.assertEqual(x[0, 0, 0], 1)
            del x
            path.write_text("0 0 0 1\n0 0 0 1\n", encoding="utf-8")
            x = load_data(path, Path(root) / "cache", (2, 2, 3))
            self.assertEqual(x[0, 0, 0], 1)
            del x
            path.write_text("0 0 0 1\n0 0 0 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_data(path, Path(root) / "cache", (2, 2, 3))

    def test_auc_ties(self):
        self.assertEqual(binary_auc([0, 1, 0, 1], [0, 1, 0, 1])["auroc"], 1)
        self.assertEqual(binary_auc([0, 1], [0.5, 0.5])["auroc"], 0.5)

    def test_two_stage_checkpoint_and_cache_reproducibility(self):
        values = np.random.default_rng(42).uniform(0.1, 2, (64, 8, 12)).astype(np.float32)
        values[np.random.default_rng(4).random(values.shape) < 0.1] = 0
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "cache"
            a = run_experiment(self.config, values, Path(root) / "first", cache)
            b = run_experiment(self.config, values, Path(root) / "second", cache)
            self.assertFalse(a["preprocess"]["cache_hit"])
            self.assertTrue(b["preprocess"]["cache_hit"])
            self.assertTrue(b["pretrain"]["cache_hit"])
            self.assertEqual(a["test"], b["test"])
            self.assertEqual(a["best_epoch"], 1)
            checkpoint = torch.load(a["checkpoint"], weights_only=False)
            model = RTQP(self.config)
            model.load_state_dict(checkpoint["model"])
            model.train()
            self.assertFalse(model.sdae.training)
            self.assertFalse(any(p.requires_grad for p in model.sdae.parameters()))
            self.assertTrue(np.isfinite(a["test"]["rmse"]))


if __name__ == "__main__":
    unittest.main()
