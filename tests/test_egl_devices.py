"""Ray's restricted CUDA visibility must survive LIBERO child configuration."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from verl.utils.libero_runtime import resolve_egl_device_id
from merl.modes import require_online_real_batch


class EGLDeviceTests(unittest.TestCase):
    def test_auto_selects_visible_identifiers_including_nonzero_and_reordered_devices(self):
        for visible, expected in (("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"),
                                  ("2,5,7", "2"), ("", "0")):
            with self.subTest(visible=visible):
                selected = resolve_egl_device_id("auto", visible)
                self.assertEqual(selected, expected)
                if visible:
                    self.assertIn(selected, visible.split(","))
        self.assertEqual(resolve_egl_device_id("5", "2,5,7"), "5")
        for requested, visible in (("0", "2"), ("1", "10"), ("invalid", "0"), ("auto", "GPU-uuid")):
            with self.assertRaises(ValueError):
                resolve_egl_device_id(requested, visible)

    def test_production_wrapper_inherits_rank_visibility_and_honors_child_override(self):
        path = Path(__file__).resolve().parents[1] / "verl/workers/rollout/rob_rollout_wm_pro.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_mujoco_egl_device_id")
        scope = dict(os=os, Any=Any)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        resolve = scope[node.name]
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2", "MERL_LIBERO_EGL_DEVICE_ID": "auto"}):
            self.assertEqual(resolve(SimpleNamespace(), visible_devices_override=""), "2")
            self.assertEqual(resolve(SimpleNamespace(), visible_devices_override="0,1,2,3"), "0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
            with self.assertRaises(ValueError):
                resolve(SimpleNamespace(mujoco_egl_device_id="0"))

    def test_online_updates_reject_the_observed_partial_real_batch(self):
        for batch in (None, [], [object(), object()]):
            with self.assertRaisesRegex(RuntimeError, "real collection incomplete"):
                require_online_real_batch(batch, 6)
        require_online_real_batch([object()] * 6, 6)
