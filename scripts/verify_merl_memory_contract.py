# MERL memory / imagined-contract verification.
# 1. Replay DataProto strips rollout media and clones tensors.
# 2. Real/WM replay samples align and concat without video tensors.
# 3. PPO filter keeps only samples with valid response tokens.
# 4. Reward manager uses valid_response_tokens rather than raw finish_step.
# 5. task_description stays sample-aligned through union and strict guards.
# 6. plot_log_metrics derives the canonical reward/sample-efficiency views.
# Expected result: prints PASS and exits 0.

import importlib.util
import numpy as np
import torch
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import (
        align_keys_between_pools,
        clone_dataproto_for_replay,
        inject_actor_token_contract,
        union_prompt_and_rollout_output,
    )
    from verl.trainer.main_ppo import RobRewardManager
    from verl.trainer.ppo.dataproto_filter import DataProtoFilter
    from verl.utils.task_description_contract import normalize_task_descriptions
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing training dependency while running memory-contract verification: "
        f"{exc.name}. Run this script in the MeRL training environment with "
        "`PYTHONPATH=$PWD python scripts/verify_merl_memory_contract.py`."
    ) from exc


def _load_plot_log_metrics_module():
    module_path = os.path.join(REPO_ROOT, "scripts", "plot_log_metrics.py")
    spec = importlib.util.spec_from_file_location("plot_log_metrics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_dataproto(batch_size: int, *, is_wm: bool) -> DataProto:
    traj_len = 4
    action_token_len = 7
    response_len = traj_len * action_token_len
    tensors = {
        "task_id": torch.arange(batch_size, dtype=torch.long).view(batch_size, 1),
        "trial_id": torch.zeros(batch_size, 1, dtype=torch.long),
        "responses": torch.arange(batch_size * response_len, dtype=torch.long).view(batch_size, traj_len, action_token_len),
        "input_ids": torch.ones(batch_size, traj_len, 8, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, traj_len, 8, dtype=torch.long),
        "pixel_values": torch.ones(batch_size, traj_len, 3, 16, 16, dtype=torch.float32),
        "action": torch.ones(batch_size, 8, 7, dtype=torch.float32),
        "finish_step": torch.full((batch_size,), 4, dtype=torch.long),
        "complete": torch.ones(batch_size, dtype=torch.bool),
        "is_dummy": torch.zeros(batch_size, traj_len, dtype=torch.bool),
        "old_log_probs": torch.zeros(batch_size, response_len, dtype=torch.float32),
        "advantages": torch.ones(batch_size, response_len, dtype=torch.float32),
        "is_weight": torch.ones(batch_size, dtype=torch.float32),
        "is_wm": torch.full((batch_size,), 1.0 if is_wm else 0.0, dtype=torch.float32),
        "valid_response_tokens": torch.full((batch_size,), response_len, dtype=torch.long),
    }
    if is_wm:
        tensors.update(
            {
                "rm_scores": torch.zeros(batch_size, traj_len, action_token_len, dtype=torch.float32),
                "wm_pred_valid": torch.ones(batch_size, dtype=torch.bool),
                "wm_obs_error": torch.full((batch_size,), 0.05, dtype=torch.float32),
                "wm_done_error": torch.zeros(batch_size, dtype=torch.float32),
                "wm_valid_response_tokens": torch.full((batch_size,), response_len, dtype=torch.long),
                "wm_dummy_step_count": torch.zeros(batch_size, dtype=torch.long),
                "wm_placeholder_step_count": torch.zeros(batch_size, dtype=torch.long),
                "video": torch.ones(batch_size, 12, 32, 32, 3, dtype=torch.uint8),
                "env_video": torch.ones(batch_size, 12, 32, 32, 3, dtype=torch.uint8),
                "env_dones": torch.zeros(batch_size, 12, dtype=torch.long),
            }
        )
    else:
        tensors.update(
            {
                "video": torch.ones(batch_size, 8, 32, 32, 3, dtype=torch.uint8),
                "env_video": torch.ones(batch_size, 8, 32, 32, 3, dtype=torch.uint8),
                "env_dones": torch.zeros(batch_size, 8, dtype=torch.long),
            }
        )

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "uid": np.array([f"{'wm' if is_wm else 'real'}-{i}" for i in range(batch_size)], dtype=object),
            "task_suite_name": np.array(["libero_10"] * batch_size, dtype=object),
            "task_descriptions": np.array([f"task-{i}" for i in range(batch_size)], dtype=object),
        },
        meta_info={"source": "memory_contract_test"},
    )


def _make_reward_manager() -> RobRewardManager:
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model=SimpleNamespace(action_token_len=7, action_chunks_len=1)
        ),
        verifier=SimpleNamespace(reward_coef=1.0),
    )
    return RobRewardManager(num_examine=0, config=config)


def _assert_raises_value_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("Expected ValueError was not raised.")


def main() -> None:
    plot_mod = _load_plot_log_metrics_module()

    real = _make_dataproto(2, is_wm=False)
    wm = _make_dataproto(3, is_wm=True)

    light_real = clone_dataproto_for_replay(real)
    light_wm = clone_dataproto_for_replay(wm)

    for dp in (light_real, light_wm):
        assert "video" not in dp.batch
        assert "env_video" not in dp.batch
        assert "env_dones" not in dp.batch
        assert "action" in dp.batch
        assert "task_id" in dp.batch
        assert "trial_id" in dp.batch
        assert "task_suite_name" in dp.non_tensor_batch
        assert dp.batch["input_ids"].device.type == "cpu"

    real.batch["input_ids"].zero_()
    assert int(light_real.batch["input_ids"].sum().item()) > 0

    aligned_real, aligned_wm = align_keys_between_pools(light_real, light_wm)
    mixed = DataProto.concat([aligned_real, aligned_wm])

    assert len(mixed) == 5
    assert "wm_obs_error" in mixed.batch
    assert "wm_done_error" in mixed.batch
    assert "wm_pred_valid" in mixed.batch
    assert "task_id" in mixed.batch
    assert "trial_id" in mixed.batch
    assert "video" not in mixed.batch
    assert mixed.non_tensor_batch["uid"].shape[0] == len(mixed)
    assert mixed.non_tensor_batch["task_suite_name"].shape[0] == len(mixed)

    filtered = DataProtoFilter.filter_ppo_samples(
        mixed,
        require_pixel_values=False,
        action_token_len=7,
        require_valid_response_tokens=True,
    )
    assert len(filtered) == len(mixed)
    assert int(filtered.batch["is_wm"].sum().item()) == 3
    contract_stats = inject_actor_token_contract(
        filtered,
        action_token_len=7,
        strict=True,
        context="verify_actor_contract",
    )
    assert "is_wm_token_mask" in filtered.batch
    assert "is_weight_token" in filtered.batch
    assert contract_stats["imag_sample_count"] == 3
    assert contract_stats["imag_token_count"] == 3 * 28
    assert contract_stats["real_token_count"] == 2 * 28
    assert contract_stats["imag_weight_mean"] > 0
    masked_replay = clone_dataproto_for_replay(filtered)
    assert "is_wm_token_mask" in masked_replay.batch
    assert "is_weight_token" in masked_replay.batch

    broken_wm = clone_dataproto_for_replay(wm)
    broken_wm.batch["wm_valid_response_tokens"].zero_()
    broken_filtered = DataProtoFilter.filter_ppo_samples(
        broken_wm,
        require_pixel_values=False,
        action_token_len=7,
        require_valid_response_tokens=True,
    )
    assert len(broken_filtered) == 0

    prompt = _make_dataproto(2, is_wm=False)
    rollout = _make_dataproto(2, is_wm=False)
    rollout.batch["task_id"] = rollout.batch["task_id"] + 100
    rollout.non_tensor_batch["task_descriptions"] = np.array(
        ["pick up the mug", "open the drawer"], dtype=object
    )
    rollout.meta_info["task_descriptions"] = ["wrong-a", "wrong-b"]
    merged = union_prompt_and_rollout_output(
        prompt,
        rollout,
        context="verify_conflicting_identity",
    )
    assert "task_id" in merged.batch
    assert torch.equal(
        merged.batch["task_id"], torch.arange(2, dtype=torch.long).view(2, 1)
    )
    assert list(merged.non_tensor_batch["task_descriptions"]) == [
        "pick up the mug",
        "open the drawer",
    ]
    assert "task_descriptions" not in merged.meta_info

    assert normalize_task_descriptions(None, 2, context="verify_task_desc") == [
        "",
        "",
    ]
    _assert_raises_value_error(
        normalize_task_descriptions,
        ["only-one-task"],
        2,
        context="verify_task_desc_mismatch",
    )

    reward_manager = _make_reward_manager()

    real_reward = _make_dataproto(1, is_wm=False)
    real_reward.batch["finish_step"].fill_(4)
    real_reward.batch["valid_response_tokens"].fill_(14)
    real_reward.batch["acc"] = torch.tensor([1.0], dtype=torch.float32)
    reward_tensor_dict, reward_metrics = reward_manager(real_reward)
    gt_scores = reward_tensor_dict["gt_scores"]
    assert torch.count_nonzero(gt_scores).item() == 1
    assert torch.isclose(gt_scores[0, 13], torch.tensor(1.0, dtype=gt_scores.dtype))
    assert torch.isclose(
        reward_tensor_dict["all"][0, 13], torch.tensor(1.0, dtype=gt_scores.dtype)
    )
    assert torch.isclose(
        torch.tensor(reward_metrics["verifier"], dtype=gt_scores.dtype),
        torch.tensor(1.0, dtype=gt_scores.dtype),
    )

    wm_reward = _make_dataproto(1, is_wm=True)
    wm_reward.batch["finish_step"].fill_(4)
    wm_reward.batch["valid_response_tokens"].fill_(14)
    wm_reward.batch["wm_valid_response_tokens"].fill_(14)
    wm_reward.batch["rm_scores"] = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32
    )
    reward_tensor_dict, reward_metrics = reward_manager(wm_reward)
    rm_scores = reward_tensor_dict["rm_scores"]
    nonzero_idx = torch.nonzero(rm_scores[0], as_tuple=False).flatten().tolist()
    assert nonzero_idx == [6, 13]
    assert torch.isclose(rm_scores[0, 6], torch.tensor(0.1, dtype=rm_scores.dtype))
    assert torch.isclose(rm_scores[0, 13], torch.tensor(0.2, dtype=rm_scores.dtype))
    assert torch.isclose(
        torch.tensor(reward_metrics["reward_model"], dtype=rm_scores.dtype),
        torch.tensor(0.3, dtype=rm_scores.dtype),
        atol=1e-6,
    )

    per_step = {
        0: {
            "rollout/num_samples": 12.0,
            "rollout/success_rate": 0.25,
            "train_reward/verifier": 1.25,
        },
        1: {
            "rollout/real_num_samples": 11.0,
            "rollout/num_samples": 12.0,
            "wm/ratio_wm": 0.5,
            "train_reward/reward_all": 0.8,
            "train_reward/reward_model": 0.2,
            "train_reward/verifier": 0.6,
            "wm/loss": 0.4,
        },
        2: {
            "rollout/num_samples": 12.0,
            "wm/ratio_wm": 0.5,
            "train_reward/reward_model": 0.7,
            "wm/loss_ema": 0.3,
        },
    }
    steps = [0, 1, 2]
    plot_mod.add_derived_metrics(per_step, steps)
    assert per_step[0]["sample_efficiency/cum_real_env_samples"] == 12.0
    assert per_step[1]["sample_efficiency/cum_real_env_samples"] == 23.0
    assert per_step[2]["sample_efficiency/cum_real_env_samples"] == 23.0
    assert per_step[0]["train_reward/main"] == 1.25
    assert per_step[1]["train_reward/main"] == 0.8
    assert per_step[2]["train_reward/main"] == 0.7

    plot_result = plot_mod.ParseResult(
        steps=steps,
        per_step=per_step,
        files=[],
        record_count=3,
        bad_line_count=0,
        duplicate_step_count=0,
        step_sources={step: "verify" for step in steps},
    )
    assert plot_mod.resolve_key("train_reward/main", plot_result) == "train_reward/main"
    assert (
        plot_mod.resolve_key("rollout/real_success_rate", plot_result)
        == "rollout/success_rate"
    )
    assert "train_reward/verifier" not in plot_mod.PRESETS["merl"]["01_policy_success_rate"]
    assert "train_reward/main" in plot_mod.PRESETS["merl"]["01b_train_reward"]
    assert "train_reward/main" in plot_mod.PRESETS["mbrl"]["01b_train_reward"]
    assert "train_reward/main" in plot_mod.PRESETS["mfrl"]["01b_train_reward"]
    assert "wm/ratio_wm" in plot_mod.PRESETS["mbrl"]["03b_wm_ratio_scheduler"]
    assert "wm/loss" in plot_mod.PRESETS["merl"]["09_wm_train_update"]

    print("PASS: MERL replay/reward/task_description/plot contracts are valid.")


if __name__ == "__main__":
    main()
