"""MERL outer loop on the existing Ray/FSDP actor infrastructure.

Grounded collection -> simulator update -> stored calibration -> frozen
recursive imagination -> chunk replay -> ONE independently normalized mixed
policy update.
"""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .checkpoint import atomic_save
from .algorithm import MERLConfig, branch_coefficients, grouped_advantages
from .trust import TrustConfig, trust_scores


def resume_contract(settings):
    """Protect normalization, collection and optimizer contracts across runs."""
    import hashlib
    from omegaconf import OmegaConf
    actor = settings.actor_rollout_ref
    def plain(value):
        return OmegaConf.to_container(value, resolve=True)
    stats = Path(actor.model.path) / "dataset_statistics.json"
    evaluation = Path(actor.rollout.libero_pro_eval_config_path)
    optim = plain(actor.actor.optim)
    optim.pop("total_training_steps", None)  # Derived from the new outer-stage target.
    return dict(data={key: settings.data.get(key) for key in ("task_suite_name", "num_trials_per_task", "eval_trial_offset")},
                actor_optim=optim,
                actor_contract={key: actor.actor.get(key) for key in ("clip_ratio_low", "clip_ratio_high", "grad_clip")},
                model_dtype=actor.actor.fsdp_config.get("model_dtype", "fp32"),
                model={key: actor.model.get(key) for key in ("vla", "action_token_len", "action_chunks_len")},
                rollout={key: plain(actor.rollout[key]) if OmegaConf.is_config(actor.rollout[key]) else actor.rollout.get(key)
                         for key in ("temperature", "do_sample", "center_crop", "allowed_task_ids", "unnorm_key")},
                statistics_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
                evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest())


def real_chunks(batch, config):
    from verl import DataProto
    scores = batch.batch["complete"].float()
    groups = batch.non_tensor_batch["uid"].tolist()
    advantages = grouped_advantages(scores, groups)
    episodes, offsets, counts = [], [], []
    for episode, length in enumerate(batch.batch["finish_step"].tolist()):
        for start in range(0, int(length), config.chunk_size):
            episodes.append(episode)
            offsets.append(start // config.chunk_size)
            counts.append(min(config.chunk_size, length - start))
    if not episodes:
        raise RuntimeError("grounded collection produced no valid action chunks")
    tensors = {key: batch.batch[key][episodes, offsets][:, None]
               for key in ("responses", "input_ids", "attention_mask", "pixel_values")}
    n, tokens = len(episodes), tensors["responses"].shape[-1]
    tensors.update(finish_step=torch.tensor(counts), valid_response_tokens=torch.tensor(counts) * 7,
                   is_wm=torch.zeros(n), is_weight=torch.ones(n),
                   advantages=advantages[episodes, None].expand(-1, tokens).clone())
    return DataProto.from_dict(tensors=tensors)


def _require_workers(result, field):
    rows = [result] if isinstance(result, dict) else result
    if not rows or any(not row.get(field) for row in rows):
        raise RuntimeError(f"Incomplete distributed {field} operation: {result}")


def prune_owned_checkpoints(root, owned, keep):
    """Prune only completed checkpoints created by this invocation.

    Resumed source runs and pre-existing files never enter `owned`.
    Verification precedes every recursive removal, even for internal paths.
    """
    if keep < 0:
        raise ValueError("checkpoint retention must be nonnegative")
    while keep and len(owned) > keep:
        paths = owned[0]
        root = Path(root).resolve()
        resolved = [Path(path).resolve() for path in paths if path is not None]
        for path in resolved:
            if path == root or root not in path.parents or path.parent.parent != root:
                raise ValueError(f"checkpoint cleanup escaped the owned run: {path}")
        for path in resolved:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        owned.pop(0)


def fit(trainer):
    import ray
    from verl import DataProto
    from verl.utils.dataset.rob_dataset import collate_fn
    from verl.utils.tracking import Tracking
    from omegaconf import OmegaConf

    settings = trainer.config
    mode = settings.trainer.train_mode
    config = MERLConfig.from_dict(OmegaConf.to_container(settings.merl, resolve=True)).for_mode(mode)
    world = trainer.actor_rollout_wg.world_size
    if world not in (1, 3) or config.grounded_trajectories != 6:
        raise ValueError("MERL collection uses three groups of two trajectories with 1 or 3 actor GPUs")
    for count in (config.real_chunks_per_update, config.imagined_chunks_per_update):
        if count % world:
            raise ValueError("independent branch chunk counts must divide evenly across actor GPUs")
    logger = Tracking(project_name=mode, experiment_name=settings.trainer.experiment_name,
                      default_backend=settings.trainer.logger, local_dir=settings.trainer.default_local_dir,
                      wandb_mode=settings.trainer.wandb_mode,
                      config=OmegaConf.to_container(settings, resolve=True))
    root = Path(settings.trainer.default_local_dir)
    contract = resume_contract(settings)
    generator = torch.Generator().manual_seed(config.seed)
    trainer.actor_rollout_wg.init_training_runtime(config.seed)
    simulator = None
    if mode != "MFRL":
        trainer._ensure_wm_trainer_initialized()
        simulator = trainer.wm_trainer
        ray.get(simulator.setup_simulator.remote(asdict(config), mode))
    start_stage, transitions = 0, 0
    resume = str(settings.trainer.get("resume_from", "") or "")
    if resume:
        source = Path(resume)
        state = torch.load(source, map_location="cpu", weights_only=True)
        if (state["format_version"] != 1 or state["mode"] != mode or state["config"] != asdict(config)
                or state.get("run_contract") != contract):
            raise ValueError("merl resume requires identical mode and research configuration")
        if (settings.actor_rollout_ref.actor.optim.warmup_style != "constant"
                and state["planned_stages"] != int(settings.trainer.total_training_steps)):
            raise ValueError("a nonconstant LR schedule cannot change the planned stage budget on resume")
        _require_workers(trainer.actor_rollout_wg.load_checkpoint(state["actor"]), "loaded")
        _require_workers(trainer.actor_rollout_wg.load_training_runtime(state["actor"]), "loaded")
        if simulator is not None:
            ray.get(simulator.load_training_runtime.remote(state["simulator"]))
        generator.set_state(state["generator"])
        start_stage, transitions = state["stage"], state["grounded_transitions"]
    total_stages = int(settings.trainer.total_training_steps)
    if total_stages <= start_stage:
        raise ValueError("total stages must exceed the completed resume stage")
    if settings.trainer.val_before_train and not start_stage:
        logger.log(data=trainer._validate(global_steps=0), step=0)
    started = time.monotonic()
    output = root / "training_state"
    output.mkdir(parents=True, exist_ok=True)
    owned_checkpoints = []
    for stage in range(start_stage + 1, total_stages + 1):
        stage_started = time.monotonic()
        metrics = {"train/global_step": stage}
        indices = torch.randint(len(trainer.train_dataset), (3,), generator=generator).tolist()
        prompts = DataProto.from_single_dict(collate_fn([trainer.train_dataset[i] for i in indices]))
        prompts.non_tensor_batch["uid"] = np.asarray([f"stage:{stage}/group:{i}" for i in range(3)], dtype=object)
        prompts.meta_info.update(n_samples=2, max_steps=config.grounded_step_cap, global_steps=stage,
                                 recompute_log_prob=False, validate=False,
                                 grounded_export_dir=str(root / "grounded"))
        print(f"[merl] stage={stage}/{total_stages} grounded_collection_begin", flush=True)
        grounded = trainer.actor_rollout_wg.generate_sequences(prompts)
        if len(grounded) != config.grounded_trajectories:
            raise RuntimeError("merl grounded budget is incomplete; no actor/simulator update allowed")
        if (grounded.batch["finish_step"] < 1).any() or any(grounded.non_tensor_batch["placeholder_reason"]):
            raise RuntimeError("invalid grounded episodes cannot be filtered out of the matched budget")
        paths = grounded.non_tensor_batch["trajectory_path"].tolist()
        stage_transitions = int(grounded.batch["finish_step"].sum())
        if stage_transitions > config.grounded_trajectories * config.grounded_step_cap:
            raise RuntimeError("grounded transition allowance exceeded")
        transitions += stage_transitions
        metrics.update({"budget/grounded_trajectories": 6 * stage, "budget/grounded_transitions": transitions,
                        "budget/stage_grounded_transitions": stage_transitions,
                        "train/grounded_success": float(grounded.batch["complete"].float().mean())})
        metrics["timing/grounded_collection"] = time.monotonic() - stage_started
        candidates = real_chunks(grounded, config)
        del grounded
        real_indices = torch.randint(len(candidates), (config.real_chunks_per_update,), generator=generator)
        real = candidates.slice(real_indices)
        ratio, imagined_count = 0., 0
        imagined = None
        if simulator is not None:
            stamp = time.monotonic()
            print(f"[merl] stage={stage} simulator_update_and_stored_calibration_begin", flush=True)
            prepared = ray.get(simulator.prepare_simulation_stage.remote(paths, stage, str(output)))
            metrics.update(prepared["metrics"])
            metrics["timing/simulator_update_and_calibration"] = time.monotonic() - stamp
            anchors = prepared["anchors"]
            chosen = torch.randint(len(anchors), (3,), generator=generator).tolist()
            selected = [anchors[i] for i in chosen]
            imagined_prompts = DataProto.from_dict(
                tensors={"anchor_start": torch.tensor([x["start"] for x in selected])[:, None]},
                non_tensors={"anchor_path": np.asarray([x["path"] for x in selected], dtype=object),
                             "candidate_id": np.asarray([f"stage:{stage}/imag_group:{i}" for i in range(3)], dtype=object)})
            imagined_prompts.meta_info.update(recursive_imagination=True, merl_config=asdict(config),
                                               imagination_horizon=prepared["horizon"], simulator_revision=prepared["simulator_revision"],
                                               global_steps=stage, train_mode=mode, recompute_log_prob=False,
                                               imagination_export_dir=str(root / "imagination"))
            print(f"[merl] stage={stage} frozen_recursive_imagination_begin", flush=True)
            stamp = time.monotonic()
            imagined = trainer.actor_rollout_wg.generate_sequences(imagined_prompts)
            values = grouped_advantages(imagined.batch["proxy_score"], imagined.non_tensor_batch["candidate_group"].tolist())
            imagined.batch["advantages"] = values[:, None].expand(-1, imagined.batch["responses"].shape[-1]).clone()
            trust = trust_scores(imagined.batch["predicted_residuals"], TrustConfig(
                config.alpha_obs, config.alpha_proxy, config.priority_epsilon,
                config.priority_exponent, config.weight_eta, config.weight_min))
            probabilities = trust.probability if config.chunk_trust else torch.full((len(imagined),), 1 / len(imagined))
            imagined.batch["is_weight"] = trust.weight if config.chunk_trust else torch.ones(len(imagined))
            indices = torch.multinomial(probabilities, config.imagined_chunks_per_update,
                                        replacement=True, generator=generator)
            metrics.update({"trust/candidate_chunks": len(imagined), "trust/chunk_weight_mean": float(imagined.batch["is_weight"].mean()),
                            "trust/chunk_error_mean": float(trust.error.mean()),
                            "train/imagined_progress_mean": float(imagined.batch["proxy_score"].mean()),
                            "timing/recursive_imagination": time.monotonic() - stamp})
            # Preserve every candidate's admission and weighting evidence.
            atomic_save(dict(residuals=trust.residuals, error=trust.error, probability=probabilities,
                             weights=imagined.batch["is_weight"], selected=indices,
                             scores=imagined.batch["proxy_score"], depth=imagined.batch["rollout_depth"],
                             revision=prepared["simulator_revision"]), output / f"replay_stage_{stage:06d}.pt")
            imagined = imagined.slice(indices)
            imagined_count, ratio = len(imagined), prepared["ratio"]
            metrics.update({"wm/actor_input_imag_token_count": int(imagined.batch["valid_response_tokens"].sum()),
                            "wm/actor_input_imag_weight_mean": float(imagined.batch["is_weight"].mean()) * ratio})
        common = list(real.batch.keys())
        parts = [real] + ([imagined.select(batch_keys=common, non_tensor_batch_keys=[],
                                          meta_info_keys=[])] if imagined is not None else [])
        mixed = DataProto.concat(parts)
        mixed.batch["branch_coefficient"] = branch_coefficients(mixed.batch["is_wm"].bool(), mixed.batch["is_weight"],
                                                               ratio, len(real), imagined_count)
        # Randomize rank assignment without changing branch normalization.
        mixed = mixed.slice(torch.randperm(len(mixed), generator=generator))
        mixed.meta_info.update(chunk_objective=True, temperature=settings.actor_rollout_ref.rollout.temperature,
                               pad_token_id=trainer.tokenizer.pad_token_id)
        stamp = time.monotonic()
        mixed = mixed.union(trainer.actor_rollout_wg.compute_chunk_log_prob(mixed))
        actor = trainer.actor_rollout_wg.update_actor(mixed)
        from verl.trainer.ppo.ray_trainer import reduce_metrics
        metrics.update(reduce_metrics(actor.meta_info["metrics"]))
        metrics["timing/actor_update"] = time.monotonic() - stamp
        # Retain update evidence even if a later checkpoint/evaluation fails.
        with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(stage=stage, metrics=metrics), allow_nan=False) + "\n")
        elapsed = time.monotonic() - started
        limit = float(settings.trainer.get("max_training_seconds", 0) or 0)
        final = stage == total_stages or (limit > 0 and elapsed >= limit)
        save_freq = int(settings.trainer.save_freq)
        if final or (save_freq > 0 and stage % save_freq == 0):
            actor_path = str(root / "actor" / f"global_step_{stage}")
            simulator_path = str(root / "world_model" / f"global_step_{stage}") if simulator is not None else None
            trainer.actor_rollout_wg.save_checkpoint(actor_path)
            _require_workers(trainer.actor_rollout_wg.save_training_runtime(actor_path), "saved")
            if simulator is not None:
                ray.get(simulator.save_training_runtime.remote(simulator_path))
            state_path = output / f"completed_stage_{stage:06d}.pt"
            atomic_save(dict(format_version=1, mode=mode, config=asdict(config), stage=stage,
                             run_contract=contract,
                             planned_stages=total_stages,
                             grounded_transitions=transitions, generator=generator.get_state(),
                             actor=actor_path, simulator=simulator_path), state_path)
            (output / "latest.json").write_text(json.dumps(dict(completed_stage=stage, state=str(state_path),
                                                               actor=actor_path, simulator=simulator_path), indent=2) + "\n")
            owned_checkpoints.append((actor_path, simulator_path, str(state_path)))
            prune_owned_checkpoints(root, owned_checkpoints, int(settings.trainer.checkpoint_keep))
        test_freq = int(settings.trainer.test_freq)
        if (test_freq > 0 and stage % test_freq == 0) or (final and settings.trainer.get("final_val_after_train", True)):
            metrics.update(trainer._validate(global_steps=stage))
        metrics["timing/stage_seconds"] = time.monotonic() - stage_started
        logger.log(data=metrics, step=stage)
        with (output / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(stage=stage, metrics=metrics), allow_nan=False) + "\n")
        print(f"[merl] stage={stage} complete elapsed_seconds={metrics['timing/stage_seconds']:.1f}", flush=True)
        if final:
            break
