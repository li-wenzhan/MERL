from dataclasses import replace
import unittest

import torch

from merl.stored_calibration import (
    HistorySimulator, StoredTrajectory, build_calibration_batch, context_at,
)


class StoredCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = StoredTrajectory("episode-1", "move", torch.arange(7.)[:, None],
                                           torch.ones(6, 1), torch.arange(6.) / 5, "calibration")
        self.seen = []

    def step(self, context, actions):
        self.seen.append((context.copy(), actions.clone()))
        return context.anchor + actions.cumsum(0), torch.full((len(actions),), 0.5)

    def build(self, trajectory=None, windows=((0, 2), (0, 5))):
        return build_calibration_batch(
            [trajectory or self.trajectory], windows, history_size=2, chunk_size=3,
            stage_context=torch.tensor([1.]), simulator_revision="sim-1",
            step=self.step, encode=lambda x: x.float() * 2,
        )

    def test_temporal_alignment_and_partial_window(self):
        batch = self.build()
        self.assertTrue(torch.equal(self.seen[0][0].observations, torch.tensor([[1.], [2.]])))
        self.assertTrue(torch.equal(batch.grounded_latents[0], torch.tensor([[6.], [8.], [10.]])))
        self.assertTrue(torch.equal(batch.features.valid_steps[1], torch.tensor([True, False, False])))
        torch.testing.assert_close(batch.residuals(), torch.tensor([[0., (0.1+0.1+0.3)/3], [0., 0.5]]))
        self.assertEqual(batch.anchor_ids, ("episode-1@transition:2", "episode-1@transition:5"))

    def test_changing_future_labels_cannot_change_prediction_inputs(self):
        first = self.build(windows=((0, 2),))
        observations = self.trajectory.observations.clone()
        observations[3:] += 100
        changed = replace(self.trajectory, observations=observations, target_proxy=torch.zeros(6))
        second = self.build(changed, windows=((0, 2),))
        torch.testing.assert_close(first.features.matrix(), second.features.matrix())
        torch.testing.assert_close(first.predicted_proxy, second.predicted_proxy)
        self.assertGreater(float(second.residuals()[0, 0]), 0)
        self.assertTrue(torch.equal(self.seen[0][0].observations, self.seen[1][0].observations))

    def test_heldout_and_invalid_windows_rejected_before_simulation(self):
        for split in ("evaluation", "train", "recursive_matched_future"):
            with self.assertRaises(ValueError):
                self.build(replace(self.trajectory, split=split))
        for windows in (((0, 1),), ((0, 6),), ((0, 2), (0, 2)), ((-1, 2),)):
            with self.assertRaises(ValueError):
                self.build(windows=windows)
        self.assertEqual(self.seen, [])

    def test_recursive_history_is_prediction_only_and_action_aligned(self):
        simulator = HistorySimulator(context_at(self.trajectory, 2, 2), self.step, lambda x: x)
        first = simulator(torch.tensor([2.]), torch.tensor([[10.], [20.]]), "move")
        simulator(first.observations[-1], torch.tensor([[30.]]), "move")
        context, _ = self.seen[-1]
        torch.testing.assert_close(context.observations, torch.tensor([[12.], [32.]]))
        torch.testing.assert_close(context.actions, torch.tensor([[10.], [20.]]))
        with self.assertRaises(ValueError):
            simulator(torch.tensor([0.]), torch.ones(1, 1), "move")
        torch.testing.assert_close(self.trajectory.observations, torch.arange(7.)[:, None])

    def test_requires_explicit_post_action_targets(self):
        with self.assertRaises(ValueError):
            self.build(replace(self.trajectory, observations=torch.zeros(6, 1)))
        with self.assertRaises(ValueError):
            self.build(replace(self.trajectory, target_proxy=torch.zeros(7)))


if __name__ == "__main__":
    unittest.main()
