"""Training-data isolation and resumable simulator initialization."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from merl.algorithm import MERLConfig, save_grounded_trajectory
from merl.checkpoint import restore_rng
from merl.train_simulator import check_resume, save_training, training_data
from merl.launch import build_settings, parser


class SimulatorTrainingTests(unittest.TestCase):
    def test_collection_exports_training_only_with_selected_normalization(self):
        args = parser().parse_args(["--mode", "MFRL", "--job", "collect", "--experiment", "init",
                                    "--vla-init", "/models/init", "--collection-dir", "/data/init",
                                    "--split", "wm_train", "--unnorm-key", "libero_10_no_noops"])
        settings, _ = build_settings(args)
        self.assertTrue(settings["trainer.rollout_before_train"])
        self.assertFalse(settings["actor_rollout_ref.world_model.enable"])
        self.assertEqual(settings["actor_rollout_ref.rollout.unnorm_key"], "libero_10_no_noops")
        self.assertTrue(settings["actor_rollout_ref.rollout.grounded_export_dir"].endswith("trajectories"))
        self.assertEqual(settings["data.rollout_trial_offset"], 0)
        args.split = "wm_eval_fixed_mini"
        settings, _ = build_settings(args)
        self.assertNotIn("actor_rollout_ref.rollout.grounded_export_dir", settings)
        self.assertEqual(settings["data.rollout_trial_offset"], 10)
        args.collection_trial_offset = 20
        settings, _ = build_settings(args)
        self.assertEqual(settings["data.rollout_trial_offset"], 20)
        args.collection_trial_offset = 49
        with self.assertRaisesRegex(ValueError, "50-state"):
            build_settings(args)

    def test_training_rejects_heldout_and_duplicate_trajectories(self):
        config = MERLConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            save_grounded_trajectory(path, observations=np.zeros((10, 16, 16, 3), np.uint8),
                                     executed_actions=np.zeros((9, 7)), instruction="Put the cup away.",
                                     success=True, task_id=0, trial_id=0, stage=0)
            items, inventory = training_data(directory, config)
            self.assertEqual(len(items), 1)
            self.assertEqual(len(inventory[str(path.resolve())]), 64)
            duplicate = path.with_name("duplicate.npz")
            duplicate.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                training_data(directory, config)
            duplicate.unlink()
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                metadata["split"] = "evaluation"
                values = dict(observations=data["observations"], actions=data["actions"],
                              metadata=np.asarray(json.dumps(metadata)))
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "held-out"):
                training_data(directory, config)

    def test_resumed_optimizer_and_rng_match_uninterrupted_update(self):
        torch.manual_seed(12)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def update(net, optim):
            optim.zero_grad()
            net(torch.rand(4, 3)).square().mean().backward()
            optim.step()

        update(model, optimizer)
        contract = dict(config=asdict(MERLConfig()), data_sha256={"episode.npz": "abc"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            weights, state = save_training(model, optimizer, output, 1, contract)
            payload = torch.load(state, weights_only=True)
            check_resume(payload, contract)
            with self.assertRaisesRegex(ValueError, "identical"):
                check_resume(payload, dict(contract, data_sha256={"episode.npz": "changed"}))
            update(model, optimizer)
            restored = torch.nn.Linear(3, 1)
            restored.load_state_dict(torch.load(weights, weights_only=True))
            resumed = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            resumed.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            update(restored, resumed)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(json.loads((output / "latest.json").read_text())["step"], 1)


if __name__ == "__main__":
    unittest.main()
