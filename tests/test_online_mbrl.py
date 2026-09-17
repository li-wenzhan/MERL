"""Online MBRL keeps simulated actor updates but trains the simulator on real data."""

import ast
from pathlib import Path
import unittest

import torch
from merl.launch import build_settings, parser
from merl.modes import online_mbrl_actor_contract, validate_online_mbrl, require_online_wm_sync

ROOT = Path(__file__).resolve().parents[1]


class OnlineMBRLTests(unittest.TestCase):
    def test_all_rollout_workers_must_load_the_updated_simulator(self):
        require_online_wm_sync([{"loaded": True}] * 3)
        for results in (None, [], [{"loaded": True}, {"loaded": False}], [None]):
            with self.assertRaises(RuntimeError):
                require_online_wm_sync(results)

    def test_profile_rejects_trust_and_frozen_simulator_overrides(self):
        args = parser().parse_args(["--mode", "ONLINE_MBRL", "--sft-checkpoint", "/sft",
                                   "--wm-checkpoint", "/wm", "--experiment", "test"])
        cfg, _ = build_settings(args)
        prefix = "actor_rollout_ref.world_model."
        wm = {key[len(prefix):]: value for key, value in cfg.items() if key.startswith(prefix)}
        validate_online_mbrl(wm)
        for field, value in (("fine_tune", False), ("weak_update_enable", True),
                             ("wm_real_anchor_reward", True), ("imag_advantage_abs_clip", 2),
                             ("use_wm_reward_proxy", False), ("wm_warmup_steps", 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_online_mbrl(dict(wm, **{field: value}))

    def test_unit_actor_weights_preserve_padding_and_gradients(self):
        batch = dict(is_wm=torch.ones(2), is_weight=torch.tensor([0.2, 0.0]),
                     response_mask=torch.tensor([[1, 0], [1, 1]]))
        masks = batch["response_mask"].clone()
        online_mbrl_actor_contract(batch)
        parameter = torch.ones(2, 2, requires_grad=True)
        (parameter * batch["is_weight"][:, None] * masks).sum().backward()
        torch.testing.assert_close(parameter.grad, masks.float())
        torch.testing.assert_close(batch["response_mask"], masks)
        batch["is_wm"][0] = 0
        with self.assertRaises(RuntimeError):
            online_mbrl_actor_contract(batch)

    def test_production_mode_gate_collects_real_data_for_online_baseline(self):
        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit_wm_v5")
        update = next(n for n in method.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "update_wm_effective")
        mode_setup = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                          and isinstance(n.body[0], ast.Assign)
                          and ast.unparse(n.body[0].targets[0]) == "planned_r_wm_prev"
                          and '"ONLINE_MBRL"' in ast.unparse(n.test).replace("'", '"'))
        for mode, real_prompts, update_expected in (("MBRL", 0, False), ("ONLINE_MBRL", 3, True)):
            scope = dict(train_mode=mode, update_wm=True, batch_size=3, n_samples=2, total_needed=6)
            exec(compile(ast.Module(body=[update, mode_setup], type_ignores=[]), "mode_contract", "exec"), scope)
            self.assertEqual(scope["real_collection_prompt_target"], real_prompts)
            self.assertEqual(scope["valid_batch_target_size"], 6)
            self.assertEqual(scope["update_wm_effective"], update_expected)
        calibration = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                           and "train_mode == 'ONLINE_MBRL'" in ast.unparse(n.test)
                           and "calibration_real_batch" in ast.unparse(n.test))
        self.assertTrue(eval(compile(ast.Expression(calibration.test), "calibration", "eval"),
                             dict(train_mode="ONLINE_MBRL", weak_update_active=False,
                                  weak_update_calibration_step=False, real_collection_prompt_target=3,
                                  calibration_real_batch=None, _is_valid_dataproto=lambda value: value is not None)))
