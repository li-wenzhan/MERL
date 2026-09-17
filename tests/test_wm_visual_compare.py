"""Fixed-action recursive comparisons may not consume future ground truth."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from merl.episode_artifacts import save_episode
from merl.wm_visual_compare import load_reference, predict_reference, visual_metrics


class WMVisualCompareTests(unittest.TestCase):
    def test_recursive_context_alignment_and_future_gt_isolation(self):
        observations = np.arange(25, dtype=np.uint8)[:, None, None, None].repeat(3, axis=3)
        actions = np.arange(24, dtype=np.float32)[:, None].repeat(7, axis=1)
        calls = []
        def predict(history, past, future, seed):
            calls.append((history.copy(), past.copy(), future.copy(), seed))
            return np.full((len(future), 1, 1, 3), int(history[-1, 0, 0, 0]) + 10, dtype=np.uint8), np.zeros(len(future))
        options = dict(start=4, horizon=18, history_size=4, chunk_size=8, rollout="recursive", seed=9, predict=predict)
        frames, _ = predict_reference(observations, actions, **options)
        self.assertEqual([len(c[2]) for c in calls], [8, 8, 2])
        np.testing.assert_array_equal(calls[0][0][:, 0, 0, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(calls[0][1][:, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(calls[1][0][:, 0, 0, 0], [14] * 4)
        np.testing.assert_array_equal(calls[1][2][:, 0], np.arange(12, 20))
        changed = observations.copy()
        changed[5:] = 200
        repeated, _ = predict_reference(changed, actions, **options)
        np.testing.assert_array_equal(repeated, frames)
        teacher, _ = predict_reference(changed, actions, **dict(options, rollout="teacher_forced"))
        self.assertFalse(np.array_equal(teacher, frames))

    def test_saved_executed_actions_roundtrip_and_misalignment_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
            actions = np.full((3, 7), -1, dtype=np.float32)
            meta = dict(task_id=0, trial_id=10, environment_steps=3, valid=True,
                        observation_source="real_environment", protocol_id="test")
            folder = save_episode(tmp, frames, meta, executed_actions=actions)
            obs, actual, record, _ = load_reference(folder / "episode.json")
            np.testing.assert_array_equal(actual, actions)
            self.assertEqual(len(obs), 4)
            self.assertEqual(record["split"], "evaluation")
            with self.assertRaises(ValueError):
                save_episode(tmp, frames, meta, executed_actions=actions[:2])
            record["split"] = "train"
            (folder / "episode.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_reference(folder / "episode.json")

    def test_pixel_metrics_use_identical_frames_and_normalized_scale(self):
        zero = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        self.assertTrue(visual_metrics(zero, zero)["identical_pixels"])
        score = visual_metrics(zero, np.full_like(zero, 255))
        self.assertEqual(score["pixel_mse"], 1.0)
        self.assertEqual(score["psnr_db"], 0.0)
