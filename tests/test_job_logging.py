"""Real Bash integration checks for the ACP console wrapper."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_logged.sh"


@unittest.skipUnless(platform.system() == "Linux", "ACP logger requires Linux Bash")
class JobLoggingTests(unittest.TestCase):
    def test_stdout_stderr_failures_and_distinct_logs(self):
        with tempfile.TemporaryDirectory(prefix="acp logs ") as tmp:
            env = dict(os.environ, ACP_LOG_DIR=tmp)
            for code in (0, 17):
                result = subprocess.run(["bash", str(SCRIPT), sys.executable, "-u", "-c",
                                         f"import sys; print('early stdout'); print('early stderr', file=sys.stderr); sys.exit({code})"],
                                        env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn("early stdout", result.stdout)
                self.assertIn("early stderr", result.stdout)
            logs = list(Path(tmp).glob("*.log"))
            self.assertEqual(len(logs), 2)
            for log in logs:
                text = log.read_text()
                self.assertIn("early stdout", text)
                self.assertIn("early stderr", text)
                self.assertIn("finished_utc=", text)
            statuses = [json.loads(p.read_text()) for p in Path(tmp).glob("*.status.json")]
            self.assertEqual(sorted(s["exit_code"] for s in statuses), [0, 17])
            self.assertTrue(all(s["logging_exit_code"] == 0 for s in statuses))

    def test_missing_interpreter_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(["bash", str(SCRIPT), "/nonexistent/merl-python"],
                                    env=dict(os.environ, ACP_LOG_DIR=tmp), capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 127)
            self.assertIn("/nonexistent/merl-python", next(Path(tmp).glob("*.log")).read_text())

    def test_logging_failure_is_not_reported_as_job_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tee = root / "tee"
            tee.write_text("#!/bin/bash\ncat >/dev/null\nexit 23\n")
            tee.chmod(0o755)
            result = subprocess.run(["bash", str(SCRIPT), sys.executable, "-c", "print('ok')"],
                                    env=dict(os.environ, ACP_LOG_DIR=str(root / "logs"),
                                             PATH=str(root) + os.pathsep + os.environ["PATH"]),
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 23)
            status = json.loads(next((root / "logs").glob("*.status.json")).read_text())
            self.assertEqual(status["command_exit_code"], 0)
            self.assertEqual(status["logging_exit_code"], 23)
