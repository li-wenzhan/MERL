"""Stored-data simulator adaptation and no-oracle calibration on the WM GPU.

The actor sends past RGB/action context and proposed executed commands. This
interface never owns a LIBERO environment or reads future trajectory frames.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .paper import PaperConfig, load_grounded_trajectory
from .stored_calibration import PastContext, build_calibration_batch
from .trust import ChunkFeatures, ResidualPredictor, StageScheduler, TrustConfig, trust_scores


class PaperSimulator:
    def __init__(self, model, wm_args, device, config, mode):
        self.model = model.module if hasattr(model, "module") else model
        self.wm_args = wm_args
        self.device = torch.device(device)
        self.dtype = getattr(wm_args, "dtype_obj", next(self.model.unet.parameters()).dtype)
        self.config = PaperConfig.from_dict(config).for_mode(mode)
        self.mode = mode
        c = self.config
        if wm_args.num_history != c.history_size or wm_args.num_frames != c.chunk_size:
            raise ValueError("WM history/frame configuration must match camera-ready chunks")
        self.scheduler = StageScheduler(c.error_beta, c.reference_error, c.scheduler_kappa,
                                        c.ratio_min, c.ratio_max, c.horizon_min, c.horizon_max)
        self.predictor = ResidualPredictor(c.residual_hidden_dim, c.seed)
        self.trust = TrustConfig(c.alpha_obs, c.alpha_proxy, c.priority_epsilon,
                                 c.priority_exponent, c.weight_eta, c.weight_min)
        self.revision = None
        self.stage = None

    def images(self, frames):
        frames = torch.as_tensor(frames)
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("simulator RGB input must be uint8 [N,H,W,3]")
        value = frames.permute(0, 3, 1, 2).to(self.device, torch.float32) / 127.5 - 1
        return F.interpolate(value, (self.wm_args.height, self.wm_args.width),
                             mode="bilinear", align_corners=False).to(self.dtype)

    @torch.no_grad()
    def encode(self, frames):
        """Same frozen deterministic VAE encoder for GT and decoded predictions.

        Preserve spatial information with a fixed 4x6 pool (96 features). Using
        posterior sampling or raw diffusion latents would change the metric.
        """
        self.model.vae.eval()
        result = []
        for batch in torch.as_tensor(frames).split(8):
            latent = self.model.vae.encode(self.images(batch)).latent_dist.mean
            latent = latent.float() * self.model.vae.config.scaling_factor
            result.append(F.adaptive_avg_pool2d(latent, (4, 6)).flatten(1).cpu())
        return torch.cat(result)

    @torch.no_grad()
    def action_latent(self, actions, instruction):
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
            return self.model.action_encoder(torch.as_tensor(actions)[None].to(self.device, self.dtype),
                                             [instruction], self.model.tokenizer, self.model.text_encoder,
                                             frame_level_cond=self.wm_args.frame_level_cond)

    @torch.no_grad()
    def proxy(self, frames, action_latent):
        # Both training and inference use [-1,1] RGB, never a [0,1] shortcut.
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
            return self.model.reward_classifier.predict_score(
                self.images(frames), action_latent.reshape(-1, action_latent.shape[-1])
            ).float().cpu()

    @torch.no_grad()
    def step(self, context: PastContext, actions):
        from modules.ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        h, c = self.config.history_size, self.config.chunk_size
        if (context.observations.ndim != 4 or context.observations.dtype != torch.uint8
                or context.observations.shape[0] != h or context.observations.shape[-1] != 3
                or context.actions.shape != (h, 7) or not torch.isfinite(context.actions).all()
                or actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= c
                or not torch.isfinite(actions).all()):
            raise ValueError("camera-ready simulator requires past RGB[H], executed history[H,7] and 1..8 finite commands")
        self.model.eval()
        history = self.images(context.observations)
        history_latent = self.model.vae.encode(history).latent_dist.mean * self.model.vae.config.scaling_factor
        encoded_actions = self.action_latent(torch.cat((context.actions, actions)), context.instruction)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
            frames, _ = CtrlWorldDiffusionPipeline.__call__(
                self.model.pipeline, image=history_latent[-1:],
                text=encoded_actions, history=history_latent[None], width=self.wm_args.width,
                height=self.wm_args.height, num_frames=len(actions),
                num_inference_steps=self.wm_args.num_inference_steps,
                decode_chunk_size=min(self.wm_args.decode_chunk_size, len(actions)),
                max_guidance_scale=self.wm_args.guidance_scale, fps=self.wm_args.fps,
                motion_bucket_id=self.wm_args.motion_bucket_id, mask=None,
                output_type="frame", return_dict=False,
                frame_level_cond=self.wm_args.frame_level_cond, his_cond_zero=self.wm_args.his_cond_zero,
            )
        value = torch.from_numpy(np.clip(np.asarray(frames[0]) * 255, 0, 255).round().astype(np.uint8))
        # Keep the policy camera resolution stable through recursive rollouts.
        size = tuple(context.anchor.shape[:2])
        if tuple(value.shape[1:3]) != size:
            value = F.interpolate(value.permute(0, 3, 1, 2).float(), size, mode="bilinear",
                                  align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        before = torch.cat((context.anchor[None], value[:-1]))
        scores = self.proxy(before, encoded_actions[:, -len(actions):])
        return value, scores

    def training_window(self, item, start):
        c = self.config
        h, n = c.history_size, c.chunk_size
        count = min(n, len(item.actions) - start)
        outcomes = item.observations[start - h + 1:start + count + 1]
        before = item.observations[start - h:start + count]
        actions = item.actions[start - h:start + count]
        targets = item.target_proxy[start - h:start + count]
        valid = torch.arange(h + n) < h + count
        if count < n:
            outcomes = torch.cat((outcomes, outcomes[-1:].repeat(n - count, 1, 1, 1)))
            before = torch.cat((before, before[-1:].repeat(n - count, 1, 1, 1)))
            actions = torch.cat((actions, torch.zeros(n - count, actions.shape[1])))
            targets = torch.cat((targets, torch.zeros(n - count)))
        return dict(img=self.images(outcomes)[None], proxy_img=self.images(before)[None],
                    action=actions[None].to(self.device), reward=targets[None].to(self.device),
                    valid_steps=valid[None].to(self.device), text=[item.instruction])

    def update(self, items, optimizer, accelerator):
        c = self.config
        eligible = [item for item in items if len(item.actions) > c.history_size]
        if not eligible:
            raise RuntimeError("no complete stored history available for simulator update")
        generator = torch.Generator().manual_seed(c.seed + 1009 * self.stage)
        self.model.train()
        self.model.vae.eval()
        self.model.text_encoder.eval()
        self.model.image_encoder.eval()
        sums = torch.zeros(3)
        for _ in range(c.simulator_steps):
            item = eligible[int(torch.randint(len(eligible), (), generator=generator))]
            start = int(torch.randint(c.history_size, len(item.actions), (), generator=generator))
            batch = self.training_window(item, start)
            optimizer.zero_grad(set_to_none=True)
            # The tested CUDA runtime produces NaN visual gradients with its
            # default fused SDPA backend, despite a finite forward loss. Math
            # SDPA keeps reduced-precision intermediates in FP32. Include
            # backward because gradient checkpointing recomputes attention.
            with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
                terms, _ = self.model(batch)
                loss = terms["loss_noise"] + c.proxy_loss_weight * terms["loss_reward"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite camera-ready simulator loss")
                accelerator.backward(loss)
            try:
                accelerator.unscale_gradients(optimizer=optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.wm_args.max_grad_norm,
                                                      error_if_nonfinite=True)
            except RuntimeError as exc:
                invalid = [name for name, parameter in self.model.named_parameters()
                           if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                raise FloatingPointError(f"non-finite simulator gradient in {invalid[:8]}; update aborted") from exc
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite simulator gradient; update aborted")
            optimizer.step()
            sums += torch.tensor([float(loss.detach()), float(terms["loss_noise"].detach()),
                                  float(terms["loss_reward"].detach())])
            del terms, loss, batch
        optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        return {**dict(zip(("wm/loss", "wm/visual_loss", "wm/proxy_loss"), (sums / c.simulator_steps).tolist())),
                "wm/update/steps_done": c.simulator_steps}

    def prepare_stage(self, paths, stage, output_dir, optimizer, accelerator):
        c = self.config
        if len(paths) != c.grounded_trajectories or len(set(paths)) != len(paths):
            raise ValueError("every stage requires the full distinct grounded trajectory budget")
        items = [load_grounded_trajectory(path, c) for path in paths]
        self.stage = int(stage)
        self.revision = f"{self.mode}/stage:{stage}"
        self.stage_context = torch.tensor([float(stage)])
        metrics = {}
        if self.mode in ("MERL", "ONLINE_MBRL"):
            metrics.update(self.update(items, optimizer, accelerator))
        self.model.eval()
        windows = []
        for index, item in enumerate(items):
            if len(item.actions) <= c.history_size:
                continue
            end = max(c.history_size, len(item.actions) - c.calibration_depth * c.chunk_size)
            starts = torch.linspace(c.history_size, end, c.calibration_windows_per_trajectory + 2)[1:-1]
            windows.extend((index, start) for start in sorted(set(starts.long().tolist())))
        if not windows:
            raise RuntimeError("stored trajectories are too short to anchor imagination")
        if c.stage_trust or c.chunk_trust:
            batch = build_calibration_batch(items, windows, history_size=c.history_size,
                                            chunk_size=c.chunk_size, max_depth=c.calibration_depth,
                                            stage_context=self.stage_context, simulator_revision=self.revision,
                                            step=self.step, encode=self.encode)
            fit = self.predictor.fit(batch, stage=stage, steps=c.residual_fit_steps,
                                     learning_rate=c.residual_learning_rate)
            # Stage proxy mismatch is measured on GROUNDED pre-action pairs;
            # chunk residual targets use imagined pairs. They are distinct.
            depth_one = batch.features.depth == 1
            visual = float(batch.residuals()[depth_one, 0].mean())
            proxy_errors = []
            for index, start in windows:
                item = items[index]
                count = min(c.chunk_size, len(item.actions) - start)
                actions = item.actions[start - c.history_size:start + count]
                latent = self.action_latent(actions, item.instruction)[:, -count:]
                scores = self.proxy(item.observations[start:start + count], latent)
                proxy_errors.append((scores - item.target_proxy[start:start + count]).abs().mean())
            proxy_error = float(torch.stack(proxy_errors).mean())
            error = c.alpha_obs * visual + c.alpha_proxy * proxy_error
            metrics.update({"trust/stage_visual_error": visual, "trust/stage_proxy_error": proxy_error,
                            "trust/measured_stage_error": error,
                            **{f"trust/calibration_{k}": v for k, v in fit.items() if isinstance(v, (float, int))}})
            ratio, horizon = self.scheduler.update(error) if c.stage_trust else (c.fixed_ratio, c.fixed_horizon)
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            self.predictor.save(output / f"residual_stage_{stage:06d}.pt")
        else:
            ratio, horizon = c.fixed_ratio, c.fixed_horizon
        self.ratio, self.horizon = ratio, horizon
        metrics.update({"trust/ratio": ratio, "trust/horizon_environment_steps": horizon,
                        "trust/calibration_environment_calls": 0})
        # Driver can choose independently among stored grounded anchors.
        anchors = [dict(path=paths[index], start=start) for index, start in windows]
        return dict(metrics=metrics, ratio=ratio, horizon=horizon, anchors=anchors,
                    simulator_revision=self.revision, stage=stage)

    @torch.no_grad()
    def predict_chunk(self, observations, history_actions, actions, instruction, depth, stage, revision, seed=None):
        if stage != self.stage or revision != self.revision:
            raise RuntimeError("stale actor simulator revision")
        if not 1 <= int(depth) <= 4:
            raise ValueError("paper imagination depth must be 1..4")
        context = PastContext(torch.as_tensor(observations), torch.as_tensor(history_actions),
                              instruction, "actor-past-only-context")
        actions = torch.as_tensor(actions)
        # Isolate diffusion RNG from the arrival order of requests from three
        # actor ranks. Restoring RNG here also keeps optimizer resume stable.
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(int(seed))
            frames, proxy = self.step(context, actions)
        if self.config.chunk_trust:
            n, c = len(actions), self.config.chunk_size
            anchor = self.encode(context.anchor[None])
            future = torch.zeros(c, anchor.shape[-1])
            future[:n] = self.encode(frames)
            padded_actions = torch.zeros(c, actions.shape[-1])
            padded_actions[:n] = actions
            features = ChunkFeatures(anchor, future[None], padded_actions[None], torch.tensor([depth]),
                                     self.stage_context[None], (torch.arange(c) < n)[None])
            residual = self.predictor.predict(features, simulator_revision=revision, stage=stage)
        else:
            residual = torch.zeros(1, 2)
        return dict(observations=frames.numpy(), proxy=proxy.numpy(), residuals=residual[0].numpy())

    def state_dict(self):
        return dict(config=asdict(self.config), mode=self.mode, stage=self.stage,
                    revision=self.revision, scheduler=asdict(self.scheduler),
                    ratio=self.ratio, horizon=self.horizon)

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config) or state["mode"] != self.mode:
            raise ValueError("simulator resume configuration differs from the completed run")
        if state["stage"] is None or not state["revision"]:
            raise ValueError("only a prepared simulator stage can be restored")
        if not 0 <= state["ratio"] <= 1 or not self.config.chunk_size <= state["horizon"] <= 4 * self.config.chunk_size:
            raise ValueError("invalid restored simulator ratio/horizon")
        self.scheduler = StageScheduler(**state["scheduler"])
        self.stage, self.revision = state["stage"], state["revision"]
        self.stage_context = torch.tensor([float(self.stage)])
        self.ratio, self.horizon = state["ratio"], state["horizon"]
