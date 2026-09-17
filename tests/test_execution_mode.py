"""Exercise the entrypoint's mode boundary without starting Ray."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class ExecutionModeTests(unittest.TestCase):
    def test_policy_evaluation_disables_simulation_before_resource_planning(self):
        path = Path(__file__).resolve().parents[1] / "verl/trainer/main_ppo.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_configure_policy_evaluation")
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        configure = scope[node.name]
        for mode in ("MFRL", "MBRL", "MERL", "ONLINE_MBRL"):
            for evaluation in (True, False):
                wm = SimpleNamespace(enable=True, fine_tune=True, fixed_eval_enabled=True)
                config = SimpleNamespace(trainer=SimpleNamespace(train_mode=mode, val_only=evaluation),
                                         actor_rollout_ref=SimpleNamespace(world_model=wm))
                self.assertEqual(configure(config), evaluation)
                self.assertEqual((wm.enable, wm.fine_tune, wm.fixed_eval_enabled), (not evaluation,) * 3)
