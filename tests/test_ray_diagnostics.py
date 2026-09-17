"""A pending startup must fail visibly and preserve useful node-local evidence."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from merl.ray_diagnostics import RayLogCapture, startup_get


class RayDiagnosticsTests(unittest.TestCase):
    def test_timeout_reports_stage_and_resources_and_worker_errors_propagate(self):
        class GetTimeoutError(Exception):
            pass
        ray = SimpleNamespace(get=Mock(side_effect=GetTimeoutError),
                              exceptions=SimpleNamespace(GetTimeoutError=GetTimeoutError),
                              cluster_resources=lambda: {"GPU": 4}, available_resources=lambda: {"GPU": 1})
        with patch.dict("sys.modules", {"ray": ray}), \
                patch("merl.ray_diagnostics.time.monotonic", side_effect=[0, 0, 1, 2]):
            with self.assertRaisesRegex(TimeoutError, "placement.*GPU.*ray_logs"):
                startup_get("pending", stage="placement", timeout=2, interval=1)
        ray.get.side_effect = RuntimeError("worker import failed")
        with patch.dict("sys.modules", {"ray": ray}):
            with self.assertRaisesRegex(RuntimeError, "worker import failed"):
                startup_get("failed", stage="actor init")
        ray.get.side_effect = None
        ray.get.return_value = ["rank0", "rank1", "rank2"]
        with patch.dict("sys.modules", {"ray": ray}):
            self.assertEqual(startup_get("ready", stage="actor init"), ray.get.return_value)

    def test_capture_retains_bounded_tails_and_observes_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "ray/session_001/logs"
            logs.mkdir(parents=True)
            source = logs / "worker-1.err"
            source.write_bytes(b"0123456789")
            (logs / "not-a-log.bin").write_bytes(b"do not copy")
            capture = RayLogCapture(root / "ray", root / "saved", tail_bytes=6)
            capture.snapshot()
            target = root / "saved/session_001/logs/worker-1.err"
            self.assertEqual(target.read_bytes(), b"456789")
            source.write_bytes(b"0123456789failure")
            capture.snapshot()
            self.assertEqual(target.read_bytes(), b"ailure")
            self.assertFalse((target.parent / "not-a-log.bin").exists())
            record = json.loads((root / "saved/index.json").read_text())["files"][0]
            self.assertEqual((record["source_bytes"], record["retained_bytes"]), (17, 6))

    def test_capture_context_takes_final_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with RayLogCapture(root / "ray", root / "saved"):
                logs = root / "ray/session_001/logs"
                logs.mkdir(parents=True)
                (logs / "raylet.out").write_text("final failure")
            self.assertEqual((root / "saved/session_001/logs/raylet.out").read_text(), "final failure")
