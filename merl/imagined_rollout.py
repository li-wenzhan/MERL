"""Policy-driven recursive imagination, with no environment calls or futures."""

from pathlib import Path
import hashlib

import numpy as np
import torch

from .algorithm import MERLConfig, load_grounded_trajectory


@torch.no_grad()
def generate_imagination(rollout, prompts):
    import ray
    from verl import DataProto
    from verl.utils.libero_pro_utils import invert_gripper_action, normalize_gripper_action
    from merl.episode_artifacts import save_episode

    config = MERLConfig.from_dict(prompts.meta_info["merl_config"])
    stage = int(prompts.meta_info["global_steps"])
    horizon = int(prompts.meta_info["imagination_horizon"])
    revision = prompts.meta_info["simulator_revision"]
    simulator = ray.get_actor("world_model_trainer")
    rollout.module.eval()
    records = []
    groups = []
    residuals, scores, lengths, depths = [], [], [], []
    for path, start, candidate_id in zip(prompts.non_tensor_batch["anchor_path"],
                                        prompts.batch["anchor_start"].flatten().tolist(),
                                        prompts.non_tensor_batch["candidate_id"]):
        item = load_grounded_trajectory(path, config)
        # Copy only past context; keep future arrays out of the rollout loop.
        from .stored_calibration import context_at
        original_context = context_at(item, start, config.history_size)
        instruction = item.instruction
        anchor_id = original_context.anchor_id
        del item
        for sample in range(config.imagined_group_size):
            context = original_context.copy()
            elapsed, depth = 0, 1
            frames = [context.anchor.numpy().copy()]
            executed = []
            while elapsed < horizon:
                device = next(rollout.module.parameters()).device
                inputs = rollout.process_input([dict(full_image=context.anchor.numpy())], [instruction])
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                token_data = rollout._generate_one_step_oft(inputs)
                raw_actions = token_data["action"]
                if torch.is_tensor(raw_actions):
                    raw_actions = raw_actions.detach().cpu().numpy()
                commands = np.asarray(raw_actions)[0].copy()
                commands = np.stack([invert_gripper_action(normalize_gripper_action(a.copy(), binarize=True))
                                     for a in commands]).astype(np.float32)
                count = min(config.chunk_size, horizon - elapsed)
                request_seed = config.seed + int(hashlib.sha256(f"{candidate_id}/{sample}/{depth}".encode()).hexdigest()[:8], 16)
                output = ray.get(simulator.predict_imagined_chunk.remote(
                    context.observations.numpy(), context.actions.numpy(), commands[:count],
                    instruction, depth, stage, revision, request_seed))
                prediction = torch.from_numpy(output["observations"])
                proxy = torch.from_numpy(output["proxy"])
                if prediction.shape != (count, *context.anchor.shape) or proxy.shape != (count,):
                    raise RuntimeError("merl simulator output/action alignment failed")
                if not torch.isfinite(proxy).all() or ((proxy < 0) | (proxy > 1)).any():
                    raise RuntimeError("merl proxy must return bounded finite progress")
                record = {key: token_data[key].detach().cpu()[:, None]
                          for key in ("responses", "input_ids", "attention_mask", "pixel_values")}
                records.append(record)
                groups.append(f"{candidate_id}/depth:{depth}")
                residuals.append(torch.from_numpy(output["residuals"]))
                # Local mean success-to-go is an explicit bounded chunk score.
                scores.append(float(proxy.mean()))
                lengths.append(count)
                depths.append(depth)
                frames.extend(list(output["observations"]))
                executed.extend(list(commands[:count]))
                h = config.history_size
                context = type(context)(torch.cat((context.observations, prediction))[-h:],
                                        torch.cat((context.actions, torch.from_numpy(commands[:count])))[-h:],
                                        instruction, anchor_id)
                elapsed += count
                depth += 1
            # Keep generated observations for public comparison workflows.
            directory = prompts.meta_info.get("imagination_export_dir")
            if directory:
                name = hashlib.sha256(f"{candidate_id}/{sample}".encode()).hexdigest()[:12]
                save_episode(Path(directory) / f"stage_{stage:06d}" / name, frames,
                             dict(task_id=-1, trial_id=sample, instruction=instruction, success=False,
                                  valid=True, observation_source="world_model", stage=stage,
                                  simulator_revision=revision, anchor_id=anchor_id, candidate_id=candidate_id, proxy_only=True,
                                  environment_steps=0, imagined_steps=horizon,
                                  label=prompts.meta_info.get("train_mode", "MERL")),
                             executed_actions=executed)
    if not records:
        raise RuntimeError("merl imagination returned no valid chunks")
    tensors = {key: torch.cat([row[key] for row in records], 0) for key in records[0]}
    n = len(records)
    tensors.update(finish_step=torch.tensor(lengths), valid_response_tokens=torch.tensor(lengths) * 7,
                   proxy_score=torch.tensor(scores), rollout_depth=torch.tensor(depths),
                   predicted_residuals=torch.stack(residuals), is_wm=torch.ones(n), is_weight=torch.ones(n))
    return DataProto.from_dict(tensors=tensors, non_tensors={"candidate_group": np.asarray(groups, dtype=object)})
