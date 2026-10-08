import dataclasses
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from merl.checkpoint import atomic_save, rng_state, restore_rng
from merl.paper import (PaperConfig, branch_coefficients, clipped_chunk_loss, grouped_advantages,
                        load_grounded_trajectory, save_grounded_trajectory)
from merl.proxy import soft_progress_loss


class PaperTests(unittest.TestCase):
    def test_equal_grounded_budget_and_independent_controls(self):
        config = PaperConfig()
        self.assertEqual(config.grounded_trajectories * config.grounded_step_cap, 3072)
        for mode in ("MBRL", "ONLINE_MBRL", "MFRL"):
            self.assertFalse(config.for_mode(mode).stage_trust)
            self.assertFalse(config.for_mode(mode).chunk_trust)
        self.assertTrue(config.for_mode("STATIC_TRUST").chunk_trust)
        for kwargs in (dict(horizon_max=384), dict(grounded_step_cap=513), dict(proxy_discount=0)):
            with self.assertRaises(ValueError):
                dataclasses.replace(config, **kwargs)

    def test_transition_export_roundtrip_and_soft_temporal_targets(self):
        with tempfile.TemporaryDirectory() as root:
            observations = np.arange(4, dtype=np.uint8)[:, None, None, None].repeat(3, -1)
            path = save_grounded_trajectory(Path(root) / "episode.npz", observations=observations,
                                             executed_actions=np.ones((3, 7)), instruction="place", success=True,
                                             task_id=0, trial_id=0, stage=1)
            item = load_grounded_trajectory(path, dataclasses.replace(PaperConfig(), proxy_discount=.5))
            torch.testing.assert_close(item.target_proxy, torch.tensor([.25, .5, 1.]))
            self.assertEqual(item.observations.shape[0], item.actions.shape[0] + 1)
            with self.assertRaises(ValueError):
                save_grounded_trajectory(Path(root) / "invalid.npz", observations=observations[:-1],
                                          executed_actions=np.ones((3, 7)), instruction="place", success=True,
                                          task_id=0, trial_id=0, stage=1)

    def test_soft_classifier_supervision_does_not_round_progress(self):
        logits = torch.zeros(1, 2, 2, requires_grad=True)
        target = torch.tensor([[.25, float("nan")]])
        valid = torch.tensor([[True, False]])
        loss = soft_progress_loss(logits, target, valid)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[[-.25, .25], [0., 0.]]]))
        self.assertAlmostEqual(float(loss), np.log(2), places=6)

    def test_global_mixed_chunk_gradient_matches_paper_despite_unequal_counts(self):
        new = torch.zeros(5, 3, requires_grad=True)
        old = torch.zeros_like(new)
        advantages = torch.tensor([[1., 1., 999.], [-1., -1., 999.],
                                   [2., 2., 999.], [2., 2., 999.], [2., 2., 999.]])
        mask = torch.tensor([[True, True, False]]).repeat(5, 1)
        terms = clipped_chunk_loss(new, old, advantages, mask, .2, .2)
        weights = torch.tensor([1., 1., .1, .2, .3])
        coefficients = branch_coefficients(torch.tensor([False, False, True, True, True]), weights, .6, 2, 3)
        loss = (terms * coefficients).sum()
        loss.backward()
        expected = -advantages * mask * coefficients[:, None]
        torch.testing.assert_close(new.grad, expected)
        self.assertAlmostEqual(float(coefficients[2:].sum()), .12, places=6)

    def test_masked_nan_does_not_poison_policy_gradients(self):
        new = torch.tensor([[0., float("nan")]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = clipped_chunk_loss(new, torch.zeros_like(new), torch.ones_like(new), mask, .2, .2).sum()
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[-1., 0.]]))
        with self.assertRaises(FloatingPointError):
            clipped_chunk_loss(torch.tensor([[1000.]]), torch.zeros(1, 1), torch.ones(1, 1),
                               torch.ones(1, 1, dtype=torch.bool), .2, .2)

    def test_checkpoint_retention_protects_preexisting_and_outside_files(self):
        from merl.paper_trainer import prune_owned_checkpoints
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "new_run"
            owned = []
            for stage in (1, 2):
                actor = root / "actor" / f"global_step_{stage}"
                actor.mkdir(parents=True)
                (actor / "tensor.pt").write_text("owned")
                state = root / "paper_state" / f"completed_stage_{stage}.pt"
                state.parent.mkdir(exist_ok=True)
                state.write_text("complete")
                owned.append((str(actor), None, str(state)))
            existing = Path(folder) / "previous_run"
            existing.mkdir()
            prune_owned_checkpoints(root, owned, 1)
            self.assertFalse((root / "actor/global_step_1").exists())
            self.assertTrue((root / "actor/global_step_2/tensor.pt").is_file())
            self.assertTrue(existing.is_dir())
            with self.assertRaises(ValueError):
                prune_owned_checkpoints(root, [(str(existing), None, None), *owned], 1)
            self.assertTrue(existing.is_dir())

    def test_group_statistics_precede_replay_and_do_not_mix_branches(self):
        values = grouped_advantages(torch.tensor([0., 1., .2, .4]), ["real", "real", "imag", "imag"])
        torch.testing.assert_close(values, torch.tensor([-1., 1., -1., 1.]), atol=2e-5, rtol=0)
        with self.assertRaises(ValueError):
            grouped_advantages(torch.tensor([.2]), ["unpaired"])

    def test_full_runtime_restores_optimizer_and_rng(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], lr=.03)
        parameter.sum().backward()
        optimizer.step()
        state = rng_state()
        expected = (torch.rand(2), np.random.rand(2), random.random())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "runtime.pt"
            atomic_save(dict(optimizer=optimizer.state_dict(), rng=state), path)
            payload = torch.load(path, weights_only=True)
            other = torch.optim.AdamW([torch.nn.Parameter(parameter.clone())], lr=1)
            other.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            torch.testing.assert_close(torch.rand(2), expected[0])
            np.testing.assert_equal(np.random.rand(2), expected[1])
            self.assertEqual(random.random(), expected[2])
            self.assertEqual(other.param_groups[0]["lr"], .03)
            torch.testing.assert_close(other.state_dict()["state"][0]["exp_avg"], optimizer.state_dict()["state"][0]["exp_avg"])


if __name__ == "__main__":
    unittest.main()
