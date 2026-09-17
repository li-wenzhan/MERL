"""Launcher provenance must not require Git, a checkout or a GPU in this test."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from merl.launch import main


class LaunchRuntimeTests(unittest.TestCase):
    def test_manifest_and_exit_status_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "weights"
            checkpoint.mkdir()
            (checkpoint / "dataset_statistics.json").write_text('{"libero_10":{}}')
            (checkpoint / "model.safetensors.index.json").write_text('{}')
            resolved = {"actor_rollout_ref": {"world_model": {"enable": False, "fixed_eval_enabled": False}}}
            for exit_code in (0, 17):
                with self.subTest(exit_code=exit_code):
                    name = f"run_{exit_code}"
                    argv = ["launch", "--mode", "MFRL", "--sft-checkpoint", str(checkpoint),
                            "--experiment", name, "--output-root", str(root), "--actor-gpus", "1"]
                    process = Mock(stdout=iter(["training output\n"]))
                    process.wait.return_value = exit_code
                    process.poll.return_value = exit_code
                    def run(command, **kwargs):
                        self.assertNotEqual(command[0], "git")
                    with patch("sys.argv", argv), patch("merl.launch.platform.system", return_value="Linux"), \
                         patch("merl.launch.compose_config", return_value=resolved), \
                         patch("merl.launch.subprocess.check_output", side_effect=AssertionError("Git is unavailable")), \
                         patch("merl.launch.subprocess.run", side_effect=run), \
                         patch("merl.launch.subprocess.Popen", return_value=process), \
                         patch("merl.launch.importlib.metadata.version", return_value="test"), \
                         patch("torch.cuda.device_count", return_value=1), \
                         patch("torch.cuda.get_device_name", return_value="test GPU"), \
                         patch("torch.cuda.get_device_properties", return_value=Mock(total_memory=80 * 1024**3)), \
                         patch.dict(os.environ, {"MERL_CODE_REVISION": "manual-label", "MERL_CONSOLE_LOG": "/logs/job.log"}):
                        with self.assertRaises(SystemExit) as result:
                            main()
                    self.assertEqual(result.exception.code, exit_code)
                    run_dir = root / "MFRL" / name
                    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
                    self.assertEqual(manifest["status"], "completed" if exit_code == 0 else "failed")
                    self.assertEqual(manifest["exit_code"], exit_code)
                    self.assertNotIn("git_commit", manifest)
                    self.assertEqual(manifest["code_revision_label"], "manual-label")
                    self.assertEqual(manifest["console_log"], "/logs/job.log")
                    self.assertTrue(any(p.endswith("merl/launch.py") or p.endswith("merl\\launch.py")
                                        for p in manifest["source_hashes"]))
                    self.assertTrue(all(len(digest) == 64 for digest in manifest["source_hashes"].values()))
                    self.assertEqual((run_dir / "run.log").read_text(), "training output\n")
