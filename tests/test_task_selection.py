"""Test production enumeration without importing the simulator or GPU stack."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import unittest

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def load_dataset():
    tree = ast.parse((ROOT / "verl/utils/dataset/rob_dataset.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LIBERO_Dataset")
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": lambda: SimpleNamespace(n_tasks=10)})
    scope = {"torch": torch, "Dataset": Dataset, "_require_libero_benchmark": lambda: benchmark}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rob_dataset.py", "exec"), scope)
    return scope["LIBERO_Dataset"]


class TaskSelectionTests(unittest.TestCase):
    def test_selection_preserves_exact_task_trial_pairs(self):
        dataset = load_dataset()("libero_10", num_trials_per_task=3, task_ids=[7, 2])
        self.assertEqual([(int(row["task_id"]), int(row["trial_id"])) for row in dataset],
                         [(7, 0), (7, 1), (7, 2), (2, 0), (2, 1), (2, 2)])

    def test_default_and_invalid_selections(self):
        cls = load_dataset()
        self.assertEqual(len(cls("libero_10", num_trials_per_task=2)), 20)
        for ids in ([], [0, 0], [-1], [10], [True], [1.5]):
            with self.assertRaises(ValueError):
                cls("libero_10", task_ids=ids)

    def test_evaluation_offset_keeps_online_training_states_disjoint(self):
        cls = load_dataset()
        train = cls("libero_10", num_trials_per_task=3, task_ids=[0])
        valid = cls("libero_10", num_trials_per_task=3, task_ids=[0], train_val="valid", trial_offset=10)
        self.assertEqual([int(row["trial_id"]) for row in train], [0, 1, 2])
        self.assertEqual([int(row["trial_id"]) for row in valid], [10, 11, 12])
        for offset in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                cls("libero_10", trial_offset=offset)

    def test_rollout_guard_never_relabels_tasks(self):
        tree = ast.parse((ROOT / "verl/workers/rollout/rob_rollout_wm_pro.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sanitize_task_ids")
        node.decorator_list = []
        scope = {"torch": torch, "DataProto": SimpleNamespace, "Optional": Optional, "List": List}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "rollout.py", "exec"), scope)
        prompts = SimpleNamespace(batch={"task_id": torch.tensor([[0], [9]])})
        with self.assertRaises(ValueError):
            scope["sanitize_task_ids"](prompts, [0])
        self.assertEqual(prompts.batch["task_id"].tolist(), [[0], [9]])
        self.assertIs(scope["sanitize_task_ids"](prompts, [0, 9]), prompts)
