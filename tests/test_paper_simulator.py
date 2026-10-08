"""Production simulator boundaries without loading video-model dependencies."""

from dataclasses import asdict, replace
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from merl.paper import PaperConfig
from merl.paper_simulator import PaperSimulator
from merl.stored_calibration import PastContext, StoredTrajectory


class FakeVAE(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=1.)

    def encode(self, images):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=images[:, :1].repeat(1, 4, 1, 1)))


class FakeProxy(torch.nn.Module):
    def predict_score(self, images, actions):
        self.images, self.actions = images.clone(), actions.clone()
        return (images.mean((1, 2, 3)) + 1) / 2


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = torch.nn.Linear(1, 1)
        self.vae = FakeVAE()
        self.reward_classifier = FakeProxy()
        self.pipeline = object()
        self.tokenizer = self.text_encoder = None

    def action_encoder(self, actions, *args, **kwargs):
        return actions[..., :2]


class SimulatorTests(unittest.TestCase):
    def simulator(self):
        args = SimpleNamespace(num_history=8, num_frames=8, height=4, width=6,
                               num_inference_steps=2, decode_chunk_size=8, guidance_scale=2,
                               fps=4, motion_bucket_id=127, frame_level_cond=True, his_cond_zero=False)
        return PaperSimulator(FakeModel(), args, "cpu", asdict(PaperConfig()), "MERL")

    def test_training_windows_separate_pre_action_proxy_and_post_action_visual_targets(self):
        simulator = self.simulator()
        frames = torch.arange(12, dtype=torch.uint8)[:, None, None, None].repeat(1, 4, 6, 3)
        actions = torch.arange(11, dtype=torch.float32)[:, None].repeat(1, 7)
        item = StoredTrajectory("training", "place", frames, actions, torch.arange(11) / 11, "calibration")
        batch = simulator.training_window(item, 8)
        torch.testing.assert_close(batch["img"][0, 7], simulator.images(frames[8:9])[0])
        torch.testing.assert_close(batch["img"][0, 8:11], simulator.images(frames[9:12]))
        torch.testing.assert_close(batch["proxy_img"][0, 8:11], simulator.images(frames[8:11]))
        torch.testing.assert_close(batch["action"][0, 8:11], actions[8:11])
        self.assertEqual(batch["valid_steps"][0].tolist(), [True] * 11 + [False] * 5)
        self.assertTrue((batch["action"][0, 11:] == 0).all())

    def test_inference_proxy_uses_anchor_then_predictions_in_training_range(self):
        simulator = self.simulator()
        history = torch.full((8, 4, 6, 3), 255, dtype=torch.uint8)
        commands = torch.tensor([[.2, .3, 0, 0, 0, 0, 1.], [.4, .5, 0, 0, 0, 0, -1.]])
        context = PastContext(history, torch.zeros(8, 7), "place", "past")

        class Pipeline:
            @staticmethod
            def __call__(pipeline, **kwargs):
                return [np.zeros((kwargs["num_frames"], 4, 6, 3), dtype=np.float32)], None

        module = SimpleNamespace(CtrlWorldDiffusionPipeline=Pipeline)
        with patch.dict(sys.modules, {"modules.ctrl_world.models.pipeline_ctrl_world": module}):
            frames, scores = simulator.step(context, commands)
        self.assertEqual(frames.dtype, torch.uint8)
        torch.testing.assert_close(scores, torch.tensor([1., 0.]))
        torch.testing.assert_close(simulator.model.reward_classifier.actions, commands[:, :2])
        torch.testing.assert_close(simulator.model.reward_classifier.images[:, 0, 0, 0], torch.tensor([1., -1.]))

    def test_grounded_export_rejects_bad_types_instead_of_silent_image_conversion(self):
        from merl.paper import save_grounded_trajectory
        with self.assertRaises(ValueError):
            save_grounded_trajectory("unused.npz", observations=np.zeros((2, 4, 6, 3)),
                                     executed_actions=np.ones((1, 7)), instruction="place",
                                     success=True, task_id=0, trial_id=0, stage=1)

    def test_request_seed_is_independent_of_order_and_restores_model_rng(self):
        simulator = self.simulator()
        simulator.config = replace(simulator.config, chunk_trust=False)
        simulator.stage, simulator.revision = 1, "stage:1"
        def step(context, commands):
            return torch.randint(256, (len(commands), 4, 6, 3), dtype=torch.uint8), torch.rand(len(commands))
        simulator.step = step
        arguments = (np.zeros((8, 4, 6, 3), dtype=np.uint8), np.zeros((8, 7)), np.zeros((3, 7)),
                     "place", 2, 1, "stage:1")
        before = torch.random.get_rng_state().clone()
        first = simulator.predict_chunk(*arguments, seed=101)
        simulator.predict_chunk(*arguments, seed=202)
        second = simulator.predict_chunk(*arguments, seed=101)
        np.testing.assert_array_equal(first["observations"], second["observations"])
        torch.testing.assert_close(before, torch.random.get_rng_state())

    def test_restore_reconstructs_stage_context_and_schedule(self):
        original, restored = self.simulator(), self.simulator()
        original.stage, original.revision = 3, "MERL/stage:3"
        original.ratio, original.horizon = original.scheduler.update(.25)
        state = original.state_dict()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.stage_context, torch.tensor([3.]))
        self.assertEqual((restored.ratio, restored.horizon), (original.ratio, original.horizon))
        self.assertEqual(restored.scheduler, original.scheduler)
        self.assertEqual(restored.revision, original.revision)
        state["mode"] = "MBRL"
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
