"""Exercise production loop boundaries without allocating distributed workers."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


class TrainingBudgetTests(unittest.TestCase):
    def test_both_loops_stop_at_the_budget_including_resumed_runs(self):
        path = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayTrainer")
        methods = {n.name: n for n in trainer.body if isinstance(n, ast.FunctionDef)}
        scope = {}
        exec(compile(ast.Module(body=[methods["_training_step_limit_reached"]], type_ignores=[]),
                     str(path), "exec"), scope)
        check = scope["_training_step_limit_reached"]
        for name in ("fit", "fit_wm_v5"):
            # Retain the actual epoch guard and batch-loop condition. Replace expensive
            # worker calls with three successful outer updates per simulated epoch.
            epoch_loop = copy.deepcopy(next(n for n in methods[name].body
                                            if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                                            and n.target.id == "epoch"))
            batch_loop = next(n for n in epoch_loop.body if isinstance(n, ast.While))
            batch_loop.body = ast.parse("global_steps += 1\nbatches += 1\nif batches == 3: break").body
            epoch_loop.body = [epoch_loop.body[0], *ast.parse("batches = 0").body, batch_loop]
            program = ast.fix_missing_locations(ast.Module(body=[epoch_loop], type_ignores=[]))
            code = compile(program, str(path), "exec")
            for limit, initial, expected in ((None, 0, 9), (1, 0, 1), (4, 0, 4),
                                             (4, 3, 4), (4, 4, 4), (4, 6, 6)):
                with self.subTest(mode=name, limit=limit, initial=initial):
                    config = SimpleNamespace(trainer={"total_training_steps": limit})
                    owner = SimpleNamespace(config=config)
                    owner._training_step_limit_reached = lambda step: check(owner, step)
                    class TrainerConfig(dict):
                        total_epochs = 3
                    config.trainer = TrainerConfig(config.trainer)
                    runtime = {"self": owner, "start_epoch": 0, "global_steps": initial}
                    exec(code, runtime)
                    self.assertEqual(runtime["global_steps"], expected)
