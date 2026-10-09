import os
import importlib.util
from unittest.mock import patch
import unittest

from merl.launch import build_settings, compose_config, hydra_args, parser, runtime_env


class LaunchTests(unittest.TestCase):
    def args(self, mode="MERL", extra=()):
        return parser().parse_args(["--mode", mode, "--sft-checkpoint", "/models/sft",
                                   "--wm-checkpoint", "/models/wm.pt", "--experiment", "test", *extra])

    def test_modes_require_and_select_trained_simulator(self):
        for mode in ("MERL", "MBRL", "MFRL", "ONLINE_MBRL"):
            cfg, _ = build_settings(self.args(mode))
            self.assertEqual(cfg["actor_rollout_ref.world_model.enable"], mode != "MFRL")
            self.assertEqual(cfg["actor_rollout_ref.world_model.fine_tune"], mode in ("MERL", "ONLINE_MBRL"))
            self.assertEqual(cfg["actor_rollout_ref.world_model.load_from_ckpt"], mode != "MFRL")
        args = self.args()
        args.wm_checkpoint = None
        with self.assertRaises(ValueError):
            build_settings(args)

    def test_evaluation_is_real_only_and_collection_has_no_updates(self):
        cfg, _ = build_settings(self.args(extra=("--job", "evaluate")))
        self.assertFalse(cfg["actor_rollout_ref.world_model.enable"])
        self.assertTrue(cfg["trainer.val_only"])
        cfg, _ = build_settings(self.args("MFRL", ("--job", "collect", "--shared-wm-eval", "/data/fixed")))
        self.assertTrue(cfg["trainer.rollout_before_train"])
        self.assertFalse(cfg["trainer.rollout_do_sample"])
        self.assertEqual(cfg["trainer.rollout_train_split"], "wm_eval_fixed_mini")

    def test_actor_scaling_and_controlled_overrides(self):
        cfg, _ = build_settings(self.args("MFRL", ("--actor-gpus", "1")))
        self.assertEqual(cfg["actor_rollout_ref.ref.log_prob_micro_batch_size"], 1)
        self.assertEqual(cfg["actor_rollout_ref.actor.ppo_mini_batch_size"], 2)
        values = hydra_args(cfg, ["--", "data.n_samples=2"])
        self.assertEqual([x for x in values if x.startswith("++data.n_samples=")], ["++data.n_samples=2"])
        with self.assertRaises(ValueError):
            hydra_args(cfg, ["trainer.n_gpus_per_node=8"])
        with self.assertRaises(ValueError):
            hydra_args(cfg, ["actor_rollout_ref.world_model={enable:true}"])
        with self.assertRaises(ValueError):
            hydra_args(cfg, ["trainer.val_only=false"])

    def test_preserves_scheduler_visible_devices_and_network_selection(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abc,GPU-def", "NCCL_SOCKET_IFNAME": "eth0"}, clear=True):
            env = runtime_env()
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-abc,GPU-def")
            self.assertEqual(env["NCCL_SOCKET_IFNAME"], "eth0")
            self.assertEqual(env["MERL_LIBERO_EGL_DEVICE_ID"], "auto")
            self.assertNotEqual(env.get("GLOO_SOCKET_IFNAME"), "bond0")

    @unittest.skipUnless(importlib.util.find_spec("hydra"), "configuration composition requires hydra-core")
    def test_actual_hydra_composition_and_single_gpu_contract(self):
        for mode in ("MFRL", "MBRL", "STATIC_TRUST", "ONLINE_MBRL", "MERL"):
            cfg, _ = build_settings(self.args(mode))
            resolved = compose_config(hydra_args(cfg, []))
            self.assertEqual(resolved["merl"]["grounded_trajectories"], 6)
            self.assertEqual(resolved["trainer"]["n_gpus_per_node"], 3)
            self.assertEqual(resolved["actor_rollout_ref"]["world_model"]["mixed_precision"], "bf16")
            self.assertEqual(resolved["actor_rollout_ref"]["actor"]["clip_ratio_high"],
                             resolved["actor_rollout_ref"]["actor"]["clip_ratio_low"])
        cfg, _ = build_settings(self.args("MFRL", ("--actor-gpus", "1")))
        with self.assertRaisesRegex(ValueError, "three actor GPUs"):
            compose_config(hydra_args(cfg, []))
        compose_config(hydra_args(cfg, ["actor_rollout_ref.actor.fsdp_config.model_dtype=bf16"]))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            compose_config(hydra_args(cfg, ["actor_rollout_ref.actor.clip_ratio_high=.28"]))
        for extra in (("--job", "evaluate", "--actor-gpus", "1"),
                      ("--job", "collect", "--actor-gpus", "1", "--shared-wm-eval", "/data/fixed")):
            cfg, _ = build_settings(self.args("MFRL", extra))
            compose_config(hydra_args(cfg, []))


if __name__ == "__main__":
    unittest.main()
