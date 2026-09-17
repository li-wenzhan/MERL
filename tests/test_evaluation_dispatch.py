"""Exercise validation padding through the real DataProto rank-split contract."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from verl import DataProto


class EvaluationDispatchTests(unittest.TestCase):
    def test_export_flags_survive_real_rank_splits_and_exclude_only_padding(self):
        path = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_validate")
        loop = next(node for node in method.body if isinstance(node, ast.For))
        prefix = []
        for node in loop.body:
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "test_output_gen_batch":
                break
            prefix.append(node)
        program = compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec")
        owner = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0),
                                actor_rollout_wg=SimpleNamespace(world_size=3))
        for size in (1, 2, 3, 4):
            with self.subTest(size=size):
                data = dict(task_id=torch.arange(size).unsqueeze(1),
                            task_suite_name=np.full(size, "libero_10", dtype=object))
                scope = dict(self=owner, test_data=data, np=np, torch=torch,
                             DataProto=DataProto, global_steps=2, eval_rollout_max_steps=512)
                exec(program, scope)
                dispatch = scope["dispatch_batch"]
                chunks = dispatch.chunk(3)
                self.assertEqual(len(chunks), 3)
                flags = np.concatenate([part.non_tensor_batch["evaluation_keep"] for part in chunks])
                self.assertEqual(flags.dtype, object)
                self.assertEqual(flags.tolist(), [True] * size + [False] * ((-size) % 3))
                self.assertEqual(scope["test_batch"].non_tensor_batch["evaluation_keep"].tolist(), [True] * size)
