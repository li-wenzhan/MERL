"""Check update failure semantics with a small CPU actor and the actual method."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class OptimizerBoundaryTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "verl/workers/actor/dp_rob.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"_optimizer_step", "_force_actor_optimizer_single_tensor_step",
                 "_repair_actor_optimizer_tensor_contract"}
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {"torch": torch, "FSDP": type("UnusedFSDP", (), {})}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
        actor_type = type("CPUActor", (), {name: scope[name] for name in names})
        self.actor = actor_type()
        self.actor.config = SimpleNamespace(grad_clip=1.0)
        self.actor.actor_module = torch.nn.Linear(2, 1, bias=False).double()
        self.actor.actor_optimizer = torch.optim.AdamW(self.actor.actor_module.parameters(), lr=0.01)
        self.param = next(self.actor.actor_module.parameters())
        self.param.grad = torch.ones_like(self.param)

    def test_state_repair_precedes_a_real_update(self):
        self.actor._optimizer_step()
        state = self.actor.actor_optimizer.state[self.param]
        state["exp_avg"] = state["exp_avg"].float()
        state["exp_avg_sq"] = state["exp_avg_sq"].float()
        before = self.param.detach().clone()
        self.param.grad = torch.ones_like(self.param)
        self.assertTrue(torch.isfinite(self.actor._optimizer_step()))
        self.assertEqual(state["exp_avg"].dtype, self.param.dtype)
        self.assertEqual(state["step"].item(), 2)
        self.assertFalse(torch.equal(before, self.param))

    def test_failure_propagates_without_retry(self):
        self.actor.actor_optimizer.step = Mock(side_effect=RuntimeError("found dtype mismatch"))
        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            self.actor._optimizer_step()
        self.actor.actor_optimizer.step.assert_called_once()

    def test_nonfinite_gradient_cannot_update_weights(self):
        before = self.param.detach().clone()
        self.param.grad.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            self.actor._optimizer_step()
        self.assertTrue(torch.equal(before, self.param))
        self.assertFalse(self.actor.actor_optimizer.state)

    def test_guarded_skip_does_not_touch_optimizer_state(self):
        self.actor._optimizer_step(skip_step=True)
        self.assertFalse(self.actor.actor_optimizer.state)
        self.assertIsNone(self.param.grad)
