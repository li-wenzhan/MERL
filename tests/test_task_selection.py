"""Test production enumeration without importing the simulator or GPU stack."""

import ast
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import typing
from typing import List, Optional
import unittest

import torch
import numpy as np
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
    def test_original_collection_uses_existing_assets_without_perturbations(self):
        tree = ast.parse((ROOT / "verl/workers/rollout/rob_rollout_wm_pro.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "_preprocess_libero_pro_task_suite")
        tasks = [SimpleNamespace(bddl_file="cup.bddl", init_states_file="cup.pruned_init", language="Move cup")]
        benchmark = {"libero_10": lambda: SimpleNamespace(get_num_tasks=lambda: 1, get_task=lambda i: tasks[i])}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bddl, init = root / "bddl/libero_10", root / "init/libero_10"
            bddl.mkdir(parents=True)
            init.mkdir(parents=True)
            (bddl / "cup.bddl").write_text("(:language Put the cup away.)", encoding="utf-8")
            state = init / "cup.pruned_init"
            state.write_bytes(b"state fixture")
            config = dict(bddl_files_path=str(bddl.parent), init_file_dir=str(init.parent))
            scope = {**vars(typing), "Path": Path, "os": os, "re": re,
                     "load_libero_pro_config": lambda _: config,
                     "_get_libero_pro_benchmark_dict": lambda _: benchmark}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "rollout.py", "exec"), scope)
            worker = SimpleNamespace(config=SimpleNamespace(libero_pro_eval_config_path="original.yaml"))
            prompts = SimpleNamespace(non_tensor_batch={"task_suite_name": np.asarray(["libero_10"], dtype=object)})
            descriptions = scope[node.name](worker, prompts)
            self.assertEqual(descriptions, [["Put the cup away."]])
            self.assertEqual(prompts.non_tensor_batch["task_suite_name"].tolist(), ["libero_10"])
            state.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "original LIBERO assets"):
                scope[node.name](worker, prompts)
            self.assertTrue((bddl / "cup.bddl").exists())

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
