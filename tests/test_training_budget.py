"""Exercise production loop boundaries without allocating distributed workers."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


class TrainingBudgetTests(unittest.TestCase):
    def test_wall_time_budget_is_checked_at_update_boundaries(self):
        path = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_training_step_limit_reached")
        clock = SimpleNamespace(monotonic=lambda: 160.0)
        scope = {"time": clock}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        owner = SimpleNamespace(config=SimpleNamespace(trainer={"max_training_seconds": 60}),
                                _training_started=100.0)
        check = scope["_training_step_limit_reached"]
        self.assertTrue(check(owner, 1))
        clock.monotonic = lambda: 159.0
        self.assertFalse(check(owner, 1))
        owner.config.trainer["max_training_seconds"] = 0
        clock.monotonic = lambda: 9999.0
        self.assertFalse(check(owner, 1))

    def test_real_training_dispatch_preserves_the_requested_horizon(self):
        path = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit")
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_get_positive_int_attr")
        setup = None
        for node in ast.walk(fit):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if (isinstance(statement, ast.Assign)
                        and ast.unparse(statement.targets[0]) == "gen_batch.meta_info"):
                    setup = body[index:next(i for i in range(index + 1, len(body))
                                            if isinstance(body[i], ast.Assign)
                                            and ast.unparse(body[i].targets[0]) == "gen_batch_output")]
        self.assertIsNotNone(setup)
        code = compile(ast.Module(body=[helper, *setup], type_ignores=[]), str(path), "exec")
        for horizon in (16, 384, None):
            with self.subTest(horizon=horizon):
                rollout = SimpleNamespace(train_max_steps=horizon)
                owner = SimpleNamespace(
                    config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)),
                    tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0))
                batch = SimpleNamespace(meta_info={})
                exec(code, {"self": owner, "gen_batch": batch, "n_samples": 4, "global_steps": 7})
                self.assertEqual(batch.meta_info.get("max_steps"), horizon)
                self.assertEqual(batch.meta_info["n_samples"], 4)
                self.assertEqual(batch.meta_info["global_steps"], 7)

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
