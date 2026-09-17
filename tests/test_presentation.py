"""Check artifact provenance, complete panels and actual training evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from merl.episode_artifacts import save_episode
from merl.presentation_report import build_report, load_panel, training_evidence
from merl.presentation_run import main


class PresentationTests(unittest.TestCase):
    def test_continue_on_error_preserves_failures_and_runs_remaining_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "comparison"
            protocol = dict(id="panel", task_ids=[0], trials=3, trial_ids=[10, 11, 12])
            argv = ["presentation", "--sft-checkpoint", tmp, "--wm-checkpoint", tmp,
                    "--output", str(root), "--continue-on-error"]
            with patch("sys.argv", argv), patch("merl.presentation_run.protocol_for", return_value=protocol), \
                 patch("merl.presentation_run.subprocess.run", side_effect=[
                     SimpleNamespace(returncode=17), SimpleNamespace(returncode=0),
                     SimpleNamespace(returncode=0)]) as run, \
                 patch("merl.presentation_report.build_report") as report:
                with self.assertRaises(SystemExit) as result:
                    main()
            self.assertEqual(result.exception.code, 1)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(report.call_count, 3)
            for index, mode in enumerate(("MFRL", "MBRL", "MERL")):
                info = json.loads((root / mode / "run_info.json").read_text())
                self.assertEqual(info["status"], "failed" if index == 0 else "completed")
                self.assertNotIn("--smoke", info["command"])
                self.assertIn("actor_rollout_ref.rollout.eval_max_steps=512", info["command"])
                self.assertIn("data.eval_trial_offset=10", info["command"])
                self.assertIn("trainer.max_training_seconds=900", info["command"])
                self.assertIn("actor_rollout_ref.model.checkpoint_format=hf_full_state_dict", info["command"])

    def test_failed_mode_stops_remaining_modes_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "comparison"
            argv = ["presentation", "--sft-checkpoint", tmp, "--wm-checkpoint", tmp,
                    "--modes", "MBRL", "ONLINE_MBRL", "MERL", "--output", str(root)]
            with patch("sys.argv", argv), patch("merl.presentation_run.protocol_for", return_value={"id": "panel"}), \
                 patch("merl.presentation_run.subprocess.run", return_value=SimpleNamespace(returncode=17)) as run, \
                 patch("merl.presentation_report.build_report") as report:
                with self.assertRaises(SystemExit) as result:
                    main()
            self.assertEqual(result.exception.code, 1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(report.call_count, 1)
            self.assertFalse((root / "ONLINE_MBRL").exists())
            self.assertEqual(json.loads((root / "MBRL/run_info.json").read_text())["exit_code"], 17)

    def test_saved_video_keyframes_and_incomplete_panel(self):
        import imageio.v2 as imageio
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = dict(id="frozen-panel", task_ids=[0], trials=2, trial_ids=[10, 11], horizon=512)
            (root / "protocol.json").write_text(json.dumps(protocol))
            mode = root / "MFRL"
            frames = [np.full((32, 32, 3), i * 40, dtype=np.uint8) for i in range(4)]
            metadata = dict(task_id=0, trial_id=10, protocol_id=protocol["id"], label="MFRL",
                            observation_source="real_environment", global_step=2, success=True,
                            valid=True, environment_steps=3, failure_reason="")
            folder = save_episode(mode / "episodes/step_000002", frames, metadata)
            saved = json.loads((folder / "episode.json").read_text())
            self.assertEqual(saved["frame_count"], 4)
            np.testing.assert_array_equal(np.asarray(Image.open(folder / "frame_00003.png")), frames[-1])
            reader = imageio.get_reader(folder / saved["video"])
            try:
                self.assertEqual(reader.count_frames(), 4)
            finally:
                reader.close()
            info = dict(label="MFRL", mode="MFRL", status="completed", job="train",
                        experiment_dir=str(root / "run"), assets_unchanged=True)
            (mode / "run_info.json").write_text(json.dumps(info))
            first = build_report(root)[0]
            self.assertIsNone(first["success_rate"])
            self.assertFalse(first["evaluation_complete"])
            save_episode(mode / "episodes/step_000002", frames[:2], dict(metadata, trial_id=11, success=False))
            result = build_report(root)[0]
            self.assertEqual(result["success_rate"], 0.5)
            self.assertEqual(result["completed_outer_steps"], 0)
            self.assertTrue(any("gradient" in warning for warning in result["warnings"]))
            self.assertTrue((root / "report/task_00_trial_10.mp4").exists())
            self.assertTrue((root / "report/task_00_trial_11.png").exists())
            info["status"] = "failed"
            (mode / "run_info.json").write_text(json.dumps(info))
            with patch("merl.presentation_report.render_trial") as render:
                self.assertIsNone(build_report(root)[0]["success_rate"])
                render.assert_not_called()
            save_episode(mode / "episodes/step_000002", [], metadata)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_panel(mode, protocol)

    def test_training_evidence_uses_production_metric_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"train/global_step": 0, "actor/grad_norm": 0.4,
                   "actor/optimizer_step_count": 2,
                   "wm/actor_input_imag_token_count": 56,
                   "wm/actor_input_imag_weight_mean": 0.1,
                   "wm/update/steps_done": 2}
            (root / "run_0.log").write_text("header\n2026\t0\t" + json.dumps(row) + "\n")
            result = training_evidence({"experiment_dir": tmp})
            self.assertEqual(result["completed_outer_steps"], 1)
            self.assertEqual(result["imagined_actor_tokens_logged"], 56)
            self.assertEqual(result["imagined_actor_weight_max"], 0.1)
            self.assertEqual(result["world_model_update_steps"], 2)

    def test_reject_invalid_rgb_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "uint8 RGB"):
                save_episode(tmp, [np.zeros((4, 4))], dict(task_id=0, trial_id=0))
