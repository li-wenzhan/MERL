import dataclasses
from pathlib import Path
import tempfile
import unittest

import torch

from merl.trust import (CalibrationBatch, ChunkFeatures, ResidualPredictor,
                        StageScheduler, TrustConfig, success_to_go, trust_scores)
from merl.imagination import Prediction, imagine, mixed_policy_loss, sample_chunks, score_chunks


def calibration(n=24):
    x = torch.linspace(0, 1, n)
    features = ChunkFeatures(
        x[:, None].repeat(1, 2), x[:, None, None].repeat(1, 4, 2),
        x[:, None, None].repeat(1, 4, 1), torch.ones(n),
        torch.zeros(n, 1), torch.ones(n, 4, dtype=torch.bool),
    )
    error = (0.05 + 0.3 * x)[:, None, None]
    return CalibrationBatch(
        features, features.imagined_latents + error.sqrt(),
        (0.1 + 0.2 * x)[:, None].repeat(1, 4), torch.zeros(n, 4),
        features.actions.clone(), tuple(f"trajectory-{i // 2}" for i in range(n)),
        tuple(f"anchor-{i}" for i in range(n)), tuple(f"anchor-{i}" for i in range(n)), "wm-v1",
    )


class TrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.batch = calibration()
        cls.predictor = ResidualPredictor(seed=7)
        cls.fit_stats = cls.predictor.fit(cls.batch, stage=1, steps=300)

    def test_exact_residual_definition(self):
        actual = self.batch.residuals()
        x = torch.linspace(0, 1, len(actual))
        torch.testing.assert_close(actual[:, 0], 0.05 + 0.3 * x)
        torch.testing.assert_close(actual[:, 1], 0.1 + 0.2 * x)

    def test_pair_mismatch_rejected(self):
        wrong = dataclasses.replace(self.batch, grounded_actions=self.batch.grounded_actions + 0.01)
        with self.assertRaisesRegex(ValueError, "identical action"):
            wrong.residuals()
        wrong = dataclasses.replace(self.batch, predicted_anchor_ids=("wrong",) * 24)
        with self.assertRaisesRegex(ValueError, "different anchors"):
            wrong.residuals()

    def test_heldout_excluded_and_stored_recursive_labels_allowed(self):
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.predictor.fit(dataclasses.replace(self.batch, split="evaluation"), stage=1)
        recursive = dataclasses.replace(self.batch.features, depth=torch.full((24,), 2))
        stats = self.predictor.fit(dataclasses.replace(self.batch, features=recursive), stage=1)
        self.assertEqual(stats["windows"], 24)
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.predictor.fit(dataclasses.replace(self.batch, features=recursive, split="evaluation"), stage=1)

    def test_mask_excludes_padding(self):
        mask = self.batch.features.valid_steps.clone()
        mask[:, -1] = False
        features = dataclasses.replace(self.batch.features, valid_steps=mask)
        grounded = self.batch.grounded_latents.clone()
        grounded[:, -1] = 1000
        batch = dataclasses.replace(self.batch, features=features, grounded_latents=grounded)
        torch.testing.assert_close(batch.residuals(), self.batch.residuals())

    def test_empty_and_holey_masks_rejected(self):
        for mask in (torch.zeros(24, 4, dtype=torch.bool), torch.tensor([[True, False, True, False]]).repeat(24, 1)):
            with self.assertRaises(ValueError):
                dataclasses.replace(self.batch.features, valid_steps=mask).matrix()

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            trust_scores(torch.tensor([[float("nan"), 0.]]))
        broken = self.batch.features.actions.clone()
        broken[0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            dataclasses.replace(self.batch.features, actions=broken).matrix()

    def test_priority_and_weight_match_formulas(self):
        r = torch.tensor([[0., 0.], [0.5, 0.5], [1., 1.]])
        cfg = TrustConfig(priority_epsilon=0.1, priority_exponent=2, weight_eta=3)
        scores = trust_scores(r, cfg)
        p = (r.sum(-1).double() + 0.1).pow(-2)
        torch.testing.assert_close(scores.probability, p / p.sum())
        torch.testing.assert_close(scores.weight, torch.exp(-3 * r.sum(-1)).clamp(0.05, 1))
        self.assertTrue(torch.all(scores.probability[:-1] > scores.probability[1:]))
        uniform = trust_scores(r, dataclasses.replace(cfg, priority_exponent=0))
        torch.testing.assert_close(uniform.probability, torch.full((3,), 1 / 3, dtype=torch.float64))

    def test_scheduler_formula_and_resume(self):
        s = StageScheduler(beta=0.5)
        self.assertEqual(s.update(0), (0.95, 32))
        ratio, horizon = s.update(2)  # EMA=1, confidence=0.5
        self.assertAlmostEqual(ratio, 0.5)
        self.assertEqual(horizon, 20)
        restored = StageScheduler(**dataclasses.asdict(s))
        self.assertEqual(s.update(4), restored.update(4))

    def test_success_to_go(self):
        result = success_to_go(torch.tensor([1, 0]), torch.tensor([6, 6]),
                               torch.tensor([[0, 4, 5], [0, 4, 5]]), gamma=0.5, horizon=3)
        torch.testing.assert_close(result, torch.tensor([[0.25, 0.5, 1.], [0., 0., 0.]]))

    def test_predictor_learns_and_is_frozen(self):
        self.assertLess(self.fit_stats["train_obs_mae"], 0.015)
        self.assertLess(self.fit_stats["train_proxy_mae"], 0.015)
        result = self.predictor.predict(self.batch.features, simulator_revision="wm-v1", stage=1)
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in self.predictor.model.parameters()))

    def test_stale_or_untrained_rejected(self):
        with self.assertRaises(RuntimeError):
            ResidualPredictor().predict(self.batch.features, simulator_revision="wm-v1", stage=1)
        for revision, stage in (("wm-v2", 1), ("wm-v1", 2)):
            with self.assertRaisesRegex(ValueError, "stale"):
                self.predictor.predict(self.batch.features, simulator_revision=revision, stage=stage)

    def test_prediction_cannot_access_future_labels(self):
        before = self.predictor.predict(self.batch.features, simulator_revision="wm-v1", stage=1)
        future = dataclasses.replace(self.batch, grounded_latents=self.batch.grounded_latents + 10,
                                     target_proxy=torch.ones_like(self.batch.target_proxy))
        after = self.predictor.predict(future.features, simulator_revision="wm-v1", stage=1)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_rng_and_checkpoint_roundtrip(self):
        state = torch.get_rng_state().clone()
        predictor = ResidualPredictor(seed=19)
        predictor.fit(self.batch, stage=1, steps=10)
        torch.testing.assert_close(torch.get_rng_state(), state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual.pt"
            predictor.save(path)
            restored = ResidualPredictor.load(path)
            torch.testing.assert_close(torch.get_rng_state(), state)
            torch.testing.assert_close(
                predictor.predict(self.batch.features, simulator_revision="wm-v1", stage=1),
                restored.predict(self.batch.features, simulator_revision="wm-v1", stage=1), rtol=0, atol=0,
            )

    def test_recursive_rollout_uses_predictions_and_partial_horizon(self):
        observed = []
        def policy(obs, instruction):
            observed.append(float(obs[0]))
            return torch.ones(4, 1)
        def simulator(obs, actions, instruction):
            count = len(actions)
            frames = obs[None] + torch.arange(1, count + 1)[:, None]
            return Prediction(frames, frames, torch.full((count,), 0.5))
        chunks = imagine(anchor=torch.zeros(2), instruction="toy", horizon=10, chunk_size=4,
                         policy=policy, simulator=simulator, encode_anchor=lambda x: x,
                         predictor=self.predictor, stage_context=torch.zeros(1),
                         simulator_revision="wm-v1", stage=1)
        self.assertEqual(observed, [0., 4., 8.])
        self.assertEqual([len(c.actions) for c in chunks], [4, 4, 2])
        self.assertEqual(chunks[-1].features.valid_steps.tolist(), [[True, True, False, False]])
        self.assertEqual([c.features.depth.item() for c in chunks], [1, 2, 3])
        scores = score_chunks(chunks)
        self.assertEqual(sample_chunks(scores, 8, generator=torch.Generator().manual_seed(3)).shape, (8,))

    def test_trust_scales_gradient_without_renormalizing_weights(self):
        real = torch.tensor([2., 4.], requires_grad=True)
        imag = torch.tensor([3., 5.], requires_grad=True)
        weights = torch.tensor([0.2, 0.8], requires_grad=True)
        mixed_policy_loss(real, imag, weights, 0.25).backward()
        torch.testing.assert_close(real.grad, torch.tensor([0.375, 0.375]))
        torch.testing.assert_close(imag.grad, torch.tensor([0.025, 0.1]))
        self.assertIsNone(weights.grad)


if __name__ == "__main__":
    unittest.main()
