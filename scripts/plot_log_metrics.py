#!/usr/bin/env python3
# Updated for current MERL/MBRL/MFRL tracker logs.
# - Merge multi-record-per-step JSON logs and resumed run log files.
# - Plot the current WM, MERL scheduler, policy, actor, critic, and timing metrics.
# - Keep the old --keys single-plot interface for ad-hoc debugging.

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_PATTERNS = ("run_*.log", "log.txt")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRACKER_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?P<step>\d+)\s+(?P<payload>\{.*\})\s*$"
)
CONSOLE_STEP_RE = re.compile(r"(?:^|\s)step:(?P<step>\d+)(?P<body>.*)$")

ALIASES = {
    "eval/success_rate/all": (
        "val/success_rate/all",
        "success_rate/all",
        "val/test_score/all",
        "test_score/all",
    ),
    "eval/success_rate/libero_10": (
        "val/success_rate/libero_10",
        "success_rate/libero_10",
        "val/test_score/libero_10",
        "test_score/libero_10",
    ),
    "rollout/real_success_rate": (
        "rollout/real_success_rate",
        "train/real_success_rate",
        "rollout/success_rate",
    ),
    "rollout/wm_proxy_success_rate": (
        "rollout/wm_proxy_success_rate",
        "train/wm_proxy_success_rate",
    ),
    "train_reward": (
        "train_reward/main",
        "train_reward/reward_all",
        "train_reward/all",
        "train_reward/reward_model",
        "train_reward/verifier",
    ),
    "train_reward/main": (
        "train_reward/main",
        "train_reward/reward_all",
        "train_reward/all",
        "train_reward/reward_model",
        "train_reward/verifier",
    ),
    "train_reward/reward_all": (
        "train_reward/reward_all",
        "train_reward/all",
    ),
    "wm/eval/clips": (
        "wm/eval/clip",
        "wm/eval/clips",
    ),
    "wm/eval/full/clips": (
        "wm/eval/full/clip",
        "wm/eval/full/clips",
    ),
    "wm/eval/fid": (
        "wm/eval/fid",
        "wm/eval/FID",
    ),
    "wm/eval/full/fid": (
        "wm/eval/full/fid",
        "wm/eval/full/FID",
    ),
    "wm/eval/fvd": (
        "wm/eval/fvd",
        "wm/eval/FVD",
    ),
    "wm/eval/full/fvd": (
        "wm/eval/full/fvd",
        "wm/eval/full/FVD",
    ),
}

LEARNING_SR_AUC_KEYS = (
    "learning_metrics/eval_success_rate_all_sr",
    "learning_metrics/eval_success_rate_all_auc",
    "learning_metrics/rollout_real_success_rate_sr",
    "learning_metrics/rollout_real_success_rate_auc",
)

LEARNING_R2T_KEYS = (
    "learning_metrics/eval_success_rate_all_r2t_0p3",
    "learning_metrics/eval_success_rate_all_r2t_0p4",
    "learning_metrics/rollout_real_success_rate_r2t_0p3",
    "learning_metrics/rollout_real_success_rate_r2t_0p4",
)


PRESETS: Mapping[str, OrderedDict[str, Sequence[str]]] = {
    "merl": OrderedDict(
        [
            ("00_learning_sr_auc", LEARNING_SR_AUC_KEYS),
            ("00b_learning_r2t", LEARNING_R2T_KEYS),
            (
                "01_policy_success_rate",
                (
                    "eval/success_rate/all",
                    "eval/success_rate/libero_10",
                    "rollout/real_success_rate",
                    "rollout/wm_proxy_success_rate",
                    "rollout/success_rate",
                ),
            ),
            (
                "01b_train_reward",
                (
                    "train_reward/main",
                    "train_reward/reward_all",
                    "train_reward/reward_model",
                    "train_reward/reward_model_raw",
                    "train_reward/reward_model_wm_mean",
                    "train_reward/reward_model_wm_raw_mean",
                    "train_reward/verifier",
                ),
            ),
            (
                "02_success_vs_real_samples",
                (
                    "eval/success_rate/all",
                    "eval/success_rate/libero_10",
                    "rollout/real_success_rate",
                ),
            ),
            (
                "03_delta_sample_efficiency",
                (
                    "sample_efficiency/delta_eval_success_rate_all_per_1k_real_samples",
                    "sample_efficiency/delta_eval_success_rate_libero_10_per_1k_real_samples",
                    "sample_efficiency/delta_rollout_real_success_rate_per_1k_real_samples",
                ),
            ),
            (
                "04_merl_ratio_scheduler",
                (
                    "wm/ratio_wm",
                    "wm/ratio_wm_target",
                    "wm/ratio_real",
                    "wm/ratio_real_target",
                    "wm/ratio_wm_prev",
                    "wm/ratio_signal_target",
                ),
            ),
            (
                "05_merl_confidence_weight",
                (
                    "wm/chunk_confidence_mean",
                    "wm/chunk_confidence_min",
                    "wm/chunk_confidence_max",
                    "wm/confidence_ema",
                    "wm/pred_valid_ratio",
                    "wm/sample_weight_mean",
                    "wm/merl_imagined_weight_mean_before",
                    "wm/merl_imagined_weight_mean_after",
                ),
            ),
            (
                "05b_merl_grpo_source_split",
                (
                    "wm/grpo_uid_source_split_applied",
                    "wm/grpo_uid_cross_source_shared_count",
                    "wm/grpo_uid_real_group_count",
                    "wm/grpo_uid_imag_group_count",
                    "wm/grpo_uid_singleton_group_count",
                    "wm/effective_grpo_group_count",
                    "wm/grpo_uid_mean_group_size",
                ),
            ),
            (
                "05c_merl_group_aware_actor_mix",
                (
                    "wm/group_aware_enabled",
                    "wm/group_aware_target_wm_sample_count",
                    "wm/group_aware_current_anchor_candidate_count",
                    "wm/group_aware_selected_wm_sample_count",
                    "wm/group_aware_real_replaced_count",
                    "wm/group_aware_target_zero_kept_current_real",
                    "wm/group_aware_fallback_real_batch",
                ),
            ),
            (
                "06_merl_chunk_error",
                (
                    "wm/chunk_obs_error_mean",
                    "wm/chunk_done_error_mean",
                    "wm/ratio_signal",
                    "wm/ratio_signal_ema",
                    "wm/ratio_signal_ema_for_scheduler",
                ),
            ),
            (
                "07_merl_horizon",
                (
                    "wm/current_imag_horizon",
                    "wm/next_imag_horizon",
                    "wm/rollout_n_samples",
                ),
            ),
            (
                "08_merl_replay_mix",
                (
                    "wm/num_wm_sample",
                    "wm/num_real_sample",
                    "wm/target_wm_sample_float",
                    "wm/target_wm_sample_count",
                    "wm/ratio_carry_in",
                    "wm/ratio_carry_out",
                    "wm/real_avail",
                    "wm/wm_avail",
                    "wm/mixed_size",
                ),
            ),
            (
                "09_wm_train_update",
                (
                    "wm/loss",
                    "wm/loss_ema",
                    "wm/ratio_signal",
                    "wm/ratio_signal_ema",
                    "wm/update/steps_done",
                    "wm/update/data_num_shards",
                ),
            ),
            (
                "10_wm_eval_psnr",
                (
                    "wm/eval/psnr",
                    "wm/eval/full/psnr",
                ),
            ),
            (
                "11_wm_eval_ssim",
                (
                    "wm/eval/ssim",
                    "wm/eval/full/ssim",
                ),
            ),
            (
                "12_wm_eval_lpips",
                (
                    "wm/eval/lpips",
                    "wm/eval/full/lpips",
                ),
            ),
            (
                "13_wm_eval_video_metrics",
                (
                    "wm/eval/clips",
                    "wm/eval/full/clips",
                    "wm/eval/fid",
                    "wm/eval/full/fid",
                    "wm/eval/fvd",
                    "wm/eval/full/fvd",
                ),
            ),
            (
                "13_wm_eval_reward",
                (
                    "wm/eval/reward_MSE",
                    "wm/eval/full/reward_MSE",
                    "wm/eval/reward_MAE",
                    "wm/eval/full/reward_MAE",
                    "wm/eval/reward_Correlation",
                    "wm/eval/full/reward_Correlation",
                ),
            ),
            (
                "14_wm_eval_done",
                (
                    "wm/eval/done_step_mae",
                    "wm/eval/full/done_step_mae",
                    "wm/eval/termination_event_accuracy",
                    "wm/eval/full/termination_event_accuracy",
                    "wm/eval/pred_terminate_rate",
                    "wm/eval/full/pred_terminate_rate",
                    "wm/eval/true_terminate_rate",
                    "wm/eval/full/true_terminate_rate",
                ),
            ),
            (
                "15_actor_grpo_loss",
                (
                    "actor/pg_loss",
                    "actor/pg_loss_real",
                    "actor/pg_loss_imag",
                ),
            ),
            (
                "16_imagined_ppo_contract",
                (
                    "wm/rollout_valid_token_count",
                    "wm/pre_filter_valid_imag_token_count",
                    "wm/valid_imag_token_count",
                    "wm/actor_input_imag_token_count",
                    "wm/actor_input_imag_weight_mean",
                    "actor/imag_token_count",
                    "actor/imag_weight_mean",
                    "actor/imag_adv_abs_mean",
                    "actor/imag_reward_mean",
                ),
            ),
            (
                "16b_imagined_reward_guard",
                (
                    "train_reward/reward_model",
                    "train_reward/reward_model_raw",
                    "train_reward/verifier",
                    "train_reward/reward_model_disabled",
                    "wm/merl_imagined_weight_mean_before",
                    "wm/merl_imagined_weight_mean_after",
                    "wm/merl_imagined_return_weight_mean",
                ),
            ),
            (
                "17_rollout_health",
                (
                    "rollout/success_rate",
                    "rollout/real_success_rate",
                    "rollout/wm_proxy_success_rate",
                    "rollout/finish_step_mean",
                    "rollout/full_horizon_ratio",
                    "rollout/dummy_step_ratio",
                    "rollout/dummy_sample_ratio",
                    "rollout/represented_horizon",
                ),
            ),
            (
                "18_actor_grpo_stability",
                (
                    "actor/pg_clipfrac",
                    "actor/ppo_kl",
                    "actor/grad_norm",
                    "actor_after/entropy_loss_eval",
                    "critic/kl",
                ),
            ),
            (
                "18b_actor_hard_guard",
                (
                    "actor/ppo_kl",
                    "actor/ppo_kl_hard_limit",
                    "actor/ppo_kl_hard_max",
                    "actor/ppo_kl_hard_skip_count",
                    "actor/ppo_kl_hard_total_count",
                    "actor/ppo_kl_hard_skip_ratio",
                ),
            ),
            (
                "18c_actor_lr_health",
                (
                    "actor/lr(1e-4)",
                    "actor/lr_effective(1e-4)",
                    "actor/lr_health_scale",
                    "actor/lr_health_scale_applied",
                    "actor/lr_health_scale_next",
                    "actor/lr_scheduler_skipped",
                ),
            ),
            (
                "19_critic_reward_advantage",
                (
                    "critic/rewards/mean",
                    "critic/rewards/max",
                    "critic/rewards/min",
                    "critic/advantages/mean",
                    "critic/returns/mean",
                ),
            ),
            (
                "20_timing",
                (
                    "timing/gen",
                    "timing/ref",
                    "timing/verify",
                    "timing/reward_model",
                    "timing/update_actor",
                    "timing/update_wm",
                    "timing/testing",
                ),
            ),
            (
                "21_io_cleanup",
                (
                    "wm/cleanup/freed_mb",
                    "wm/io/write_train_real",
                    "wm/io/write_eval_real",
                    "wm/io/persist_imag_shards",
                    "wm/io/warmup_active",
                ),
            ),
        ]
    ),
    "mbrl": OrderedDict(
        [
            ("00_learning_sr_auc", LEARNING_SR_AUC_KEYS),
            ("00b_learning_r2t", LEARNING_R2T_KEYS),
            (
                "01_policy_success_rate",
                (
                    "eval/success_rate/all",
                    "eval/success_rate/libero_10",
                    "rollout/real_success_rate",
                    "rollout/wm_proxy_success_rate",
                    "rollout/success_rate",
                ),
            ),
            (
                "01b_train_reward",
                (
                    "train_reward/main",
                    "train_reward/reward_all",
                    "train_reward/reward_model",
                    "train_reward/reward_model_wm_mean",
                    "train_reward/verifier",
                ),
            ),
            (
                "02_success_vs_real_samples",
                (
                    "eval/success_rate/all",
                    "eval/success_rate/libero_10",
                    "rollout/real_success_rate",
                ),
            ),
            (
                "03_delta_sample_efficiency",
                (
                    "sample_efficiency/delta_eval_success_rate_all_per_1k_real_samples",
                    "sample_efficiency/delta_eval_success_rate_libero_10_per_1k_real_samples",
                    "sample_efficiency/delta_rollout_real_success_rate_per_1k_real_samples",
                ),
            ),
            (
                "03b_wm_ratio_scheduler",
                (
                    "wm/ratio_wm",
                    "wm/ratio_wm_target",
                    "wm/ratio_real",
                    "wm/ratio_real_target",
                ),
            ),
            (
                "04_wm_train_update",
                (
                    "wm/loss",
                    "wm/loss_ema",
                    "wm/update/steps_done",
                    "wm/update/data_num_shards",
                    "wm/real_avail",
                    "wm/wm_avail",
                ),
            ),
            (
                "05_wm_eval_psnr",
                (
                    "wm/eval/psnr",
                    "wm/eval/full/psnr",
                ),
            ),
            (
                "06_wm_eval_ssim_lpips",
                (
                    "wm/eval/ssim",
                    "wm/eval/full/ssim",
                    "wm/eval/lpips",
                    "wm/eval/full/lpips",
                ),
            ),
            (
                "06b_wm_eval_video_metrics",
                (
                    "wm/eval/clips",
                    "wm/eval/full/clips",
                    "wm/eval/fid",
                    "wm/eval/full/fid",
                    "wm/eval/fvd",
                    "wm/eval/full/fvd",
                ),
            ),
            (
                "07_wm_eval_reward_done",
                (
                    "wm/eval/reward_MSE",
                    "wm/eval/full/reward_MSE",
                    "wm/eval/reward_Correlation",
                    "wm/eval/full/reward_Correlation",
                    "wm/eval/done_step_mae",
                    "wm/eval/full/done_step_mae",
                    "wm/eval/termination_event_accuracy",
                    "wm/eval/full/termination_event_accuracy",
                ),
            ),
            (
                "08_actor_critic_stability",
                (
                    "actor/pg_loss",
                    "actor/pg_clipfrac",
                    "actor/ppo_kl",
                    "actor/grad_norm",
                    "critic/kl",
                    "critic/rewards/mean",
                ),
            ),
            (
                "08b_actor_lr_health",
                (
                    "actor/lr(1e-4)",
                    "actor/lr_effective(1e-4)",
                    "actor/lr_health_scale",
                    "actor/lr_health_scale_applied",
                    "actor/lr_health_scale_next",
                    "actor/lr_scheduler_skipped",
                ),
            ),
            (
                "09_rollout_health",
                (
                    "rollout/success_rate",
                    "rollout/real_success_rate",
                    "rollout/wm_proxy_success_rate",
                    "rollout/finish_step_mean",
                    "rollout/full_horizon_ratio",
                    "rollout/dummy_step_ratio",
                    "rollout/represented_horizon",
                ),
            ),
            (
                "10_timing",
                (
                    "timing/gen",
                    "timing/ref",
                    "timing/verify",
                    "timing/reward_model",
                    "timing/update_actor",
                    "timing/update_wm",
                    "timing/testing",
                ),
            ),
        ]
    ),
    "mfrl": OrderedDict(
        [
            ("00_learning_sr_auc", LEARNING_SR_AUC_KEYS),
            ("00b_learning_r2t", LEARNING_R2T_KEYS),
            (
                "01_policy_success_rate",
                (
                    "eval/success_rate/all",
                    "eval/success_rate/libero_10",
                    "rollout/real_success_rate",
                    "rollout/wm_proxy_success_rate",
                    "rollout/success_rate",
                ),
            ),
            (
                "01b_train_reward",
                (
                    "train_reward/main",
                    "train_reward/reward_all",
                    "train_reward/reward_model",
                    "train_reward/reward_model_wm_mean",
                    "train_reward/verifier",
                ),
            ),
            (
                "02_success_vs_real_samples",
                (
                    "eval/success_rate/all",
                    "eval/success_rate/libero_10",
                    "rollout/real_success_rate",
                ),
            ),
            (
                "03_delta_sample_efficiency",
                (
                    "sample_efficiency/delta_eval_success_rate_all_per_1k_real_samples",
                    "sample_efficiency/delta_eval_success_rate_libero_10_per_1k_real_samples",
                    "sample_efficiency/delta_rollout_real_success_rate_per_1k_real_samples",
                ),
            ),
            (
                "04_actor_grpo_stability",
                (
                    "actor/pg_loss",
                    "actor/pg_clipfrac",
                    "actor/ppo_kl",
                    "actor/grad_norm",
                    "actor_after/entropy_loss_eval",
                    "critic/kl",
                ),
            ),
            (
                "04b_actor_lr_health",
                (
                    "actor/lr(1e-4)",
                    "actor/lr_effective(1e-4)",
                    "actor/lr_health_scale",
                    "actor/lr_health_scale_applied",
                    "actor/lr_health_scale_next",
                    "actor/lr_scheduler_skipped",
                ),
            ),
            (
                "05_critic_reward_advantage",
                (
                    "critic/rewards/mean",
                    "critic/rewards/max",
                    "critic/rewards/min",
                    "critic/advantages/mean",
                    "critic/returns/mean",
                ),
            ),
            (
                "06_rollout_health",
                (
                    "rollout/success_rate",
                    "rollout/real_success_rate",
                    "rollout/finish_step_mean",
                    "rollout/full_horizon_ratio",
                    "rollout/dummy_step_ratio",
                    "rollout/represented_horizon",
                ),
            ),
            (
                "07_timing",
                (
                    "timing/gen",
                    "timing/ref",
                    "timing/verify",
                    "timing/reward_model",
                    "timing/update_actor",
                    "timing/testing",
                ),
            ),
        ]
    ),
}

PRESET_X_KEYS: Mapping[str, str] = {
    "02_success_vs_real_samples": "sample_efficiency/cum_real_env_samples",
    "03_delta_sample_efficiency": "sample_efficiency/cum_real_env_samples",
}

PLOT_GUIDE = OrderedDict(
    [
        (
            "00_learning_sr_auc",
            "Learning-curve metrics: SR is the current success rate, AUC is trapezoidal area under SR over cumulative real samples, and R2T is the first real-sample budget reaching the target SR threshold.",
        ),
        (
            "00b_learning_r2t",
            "Learning-curve R2T metrics over cumulative real samples. A lower first-hit real-sample budget is better.",
        ),
        (
            "01_policy_success_rate",
            "True success-rate plot. Prefer val/success_rate/* for final comparison; rollout/real_success_rate is the training real-env signal, and rollout/wm_proxy_success_rate is only a WM proxy.",
        ),
        (
            "01b_train_reward",
            "Training-reward breakdown. train_reward/main is the reward actually used by PPO, reward_model is the WM proxy actually injected into PPO, and reward_model_raw is the raw WM proxy before any disable switch.",
        ),
        (
            "02_success_vs_real_samples",
            "Success-vs-budget plot. The x-axis is sample_efficiency/cum_real_env_samples, accumulated from actual real samples when available, so MERL/MBRL/MFRL can be compared on the same real-sample budget.",
        ),
        (
            "03_delta_sample_efficiency",
            "Incremental sample-efficiency plot: 1000 * delta(true success rate) / delta(real environment samples). Positive values mean new real samples improved SR.",
        ),
        (
            "04_merl_ratio_scheduler",
            "MERL mix-ratio scheduler. wm/ratio_wm rising means imagined samples are entering PPO; wm/ratio_wm_target is the scheduler target.",
        ),
        (
            "03b_wm_ratio_scheduler",
            "WM ratio plot for non-MERL presets. wm/ratio_wm and wm/ratio_real should reflect the actual PPO mix rather than reward quality.",
        ),
        (
            "05_merl_confidence_weight",
            "World-model confidence and imagined weights. Stable confidence/sample_weight indicates a more trustworthy imagined branch.",
        ),
        (
            "05b_merl_grpo_source_split",
            "GRPO source-split diagnostics. source_split_applied should be 1 once imagined samples enter PPO; cross_source_shared_count counts raw uid overlap before namespacing, not post-split contamination.",
        ),
        (
            "05c_merl_group_aware_actor_mix",
            "MERL group-aware actor mix diagnostics. selected_wm should be bounded by current_anchor_candidate_count, and target_zero_kept_current_real should be 1 on pure-real rounded steps.",
        ),
        (
            "06_merl_chunk_error",
            "Short-horizon world-model error. Lower chunk_obs_error and done_error are better; ratio_signal is the quality signal used by the scheduler.",
        ),
        (
            "08_merl_replay_mix",
            "Replay-pool composition. Use it to verify real samples, imagined samples, and pool availability.",
        ),
        (
            "09_wm_train_update",
            "World-model update plot. Falling loss/loss_ema suggests WM learning; steps_done confirms that inner updates are running.",
        ),
        (
            "10_wm_eval_psnr",
            "WM image-quality PSNR. Higher is better; mini/full correspond to lightweight and full evaluation intervals.",
        ),
        (
            "13_wm_eval_video_metrics",
            "WM video metrics. clips measures semantic similarity, while FID/FVD track frame/video distribution shift; mini/full correspond to lightweight and full evaluation intervals.",
        ),
        (
            "06b_wm_eval_video_metrics",
            "WM video metrics for the MBRL preset. clips/FID/FVD should all appear here when the log contains them.",
        ),
        (
            "13_wm_eval_reward",
            "WM reward-prediction quality. Lower reward_MSE/MAE and higher reward_Correlation are better.",
        ),
        (
            "14_wm_eval_done",
            "Termination/proxy completion diagnostics. Treat these as diagnostics rather than the sole scheduler signal.",
        ),
        (
            "16_imagined_ppo_contract",
            "Imagined PPO contract. When wm/ratio_wm > 0, actor/imag_token_count and actor/pg_loss_imag should become non-zero.",
        ),
        (
            "16b_imagined_reward_guard",
            "Imagined reward-routing diagnostics. reward_model_raw is the WM proxy before disabling, reward_model is the proxy actually injected into PPO, and merl_imagined_return_weight_mean shows the imagined-return downweight used by the critic path.",
        ),
        (
            "17_rollout_health",
            "Rollout health. Lower dummy rate is better; real_success_rate and wm_proxy_success_rate must be interpreted separately.",
        ),
        (
            "18_actor_grpo_stability",
            "Actor PPO stability. Watch KL, clipfrac, and grad_norm for spikes.",
        ),
        (
            "18b_actor_hard_guard",
            "Actor hard-guard plot. hard_skip_ratio > 0 means PPO chunks were actively skipped because local approximate KL exceeded the configured hard limit.",
        ),
        (
            "18c_actor_lr_health",
            "Actor LR health plot. lr_effective should drop below scheduled lr when hard KL skip ratio rises, then recover as PPO stabilizes.",
        ),
        (
            "20_timing",
            "Runtime-cost plot. gen/update_actor/update_wm/testing locate the main speed bottleneck.",
        ),
    ]
)


@dataclass
class ParseResult:
    steps: List[int]
    per_step: Dict[int, Dict[str, object]]
    files: List[Path]
    record_count: int
    bad_line_count: int
    duplicate_step_count: int
    step_sources: Dict[int, str]


def finite_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else math.nan
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null", "nan", "inf", "+inf", "-inf"}:
            return math.nan
        try:
            number = float(stripped)
        except ValueError:
            return math.nan
        return number if math.isfinite(number) else math.nan
    return math.nan


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "metrics"


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def split_log_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = strip_ansi(line)
    if "\t" in line:
        parts = line.split("\t", 2)
    else:
        parts = re.split(r"\s+", line, maxsplit=2)
    if len(parts) != 3:
        match = TRACKER_RE.search(line)
        if match is None:
            return None
        return match.group("timestamp"), match.group("step"), match.group("payload")
    if not parts[2].lstrip().startswith("{"):
        match = TRACKER_RE.search(line)
        if match is not None:
            return match.group("timestamp"), match.group("step"), match.group("payload")
    return parts[0], parts[1], parts[2]


def parse_console_step_line(line: str) -> Optional[Tuple[int, Dict[str, object]]]:
    line = strip_ansi(line)
    match = CONSOLE_STEP_RE.search(line)
    if match is None:
        return None
    metrics: Dict[str, object] = {}
    for segment in match.group("body").split(" - "):
        if ":" not in segment:
            continue
        key, value = segment.rsplit(":", 1)
        key = key.strip()
        if not key or key == "step":
            continue
        number = finite_float(value)
        if math.isfinite(number):
            metrics[key] = number
    if not metrics:
        return None
    return int(match.group("step")), metrics


def read_one_log(path: Path) -> Tuple[List[Tuple[int, Dict[str, object], str]], int]:
    records: List[Tuple[int, Dict[str, object], str]] = []
    bad_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = split_log_line(line)
            if parts is None:
                console_record = parse_console_step_line(line)
                if console_record is None:
                    bad_lines += 1
                    continue
                step, data = console_record
                records.append((step, data, f"{path}:{line_no}:console"))
                continue
            timestamp, step_text, payload = parts
            try:
                step = int(step_text)
                data = json.loads(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                console_record = parse_console_step_line(line)
                if console_record is None:
                    bad_lines += 1
                    continue
                step, data = console_record
            if isinstance(data, dict):
                records.append((step, data, f"{path}:{line_no}:{timestamp}"))
            else:
                console_record = parse_console_step_line(line)
                if console_record is None:
                    bad_lines += 1
                else:
                    step, data = console_record
                    records.append((step, data, f"{path}:{line_no}:console"))
    return records, bad_lines


def unique_existing_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def collect_log_files(args: argparse.Namespace) -> List[Path]:
    explicit_files: List[Path] = []
    dirs: List[Path] = []

    for item in args.paths or []:
        path = Path(item)
        if path.is_file():
            explicit_files.append(path)
        elif path.is_dir():
            dirs.append(path)

    explicit_files.extend(Path(item) for item in args.log_file or [])
    dirs.extend(Path(item) for item in args.log_dir or [])

    patterns = tuple(args.pattern or DEFAULT_PATTERNS)
    discovered: List[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            iterator = directory.rglob(pattern) if args.recursive else directory.glob(pattern)
            discovered.extend(path for path in iterator if path.is_file())

    files = unique_existing_paths([*explicit_files, *discovered])
    if args.file_sort == "name":
        files.sort(key=lambda path: str(path).lower())
    else:
        files.sort(key=lambda path: (path.stat().st_mtime, str(path).lower()))
    return files


def parse_logs(files: Sequence[Path]) -> ParseResult:
    per_step: MutableMapping[int, Dict[str, object]] = defaultdict(dict)
    step_sources: Dict[int, str] = {}
    record_count = 0
    bad_line_count = 0
    duplicate_step_count = 0
    seen_steps = set()

    for file_path in files:
        records, bad_lines = read_one_log(file_path)
        bad_line_count += bad_lines
        for step, data, source in records:
            if step in seen_steps:
                duplicate_step_count += 1
            seen_steps.add(step)
            per_step[step].update({key: value for key, value in data.items() if value is not None})
            step_sources[step] = source
            record_count += 1

    steps = sorted(per_step.keys())
    add_derived_metrics(per_step, steps)
    return ParseResult(
        steps=steps,
        per_step=dict(per_step),
        files=list(files),
        record_count=record_count,
        bad_line_count=bad_line_count,
        duplicate_step_count=duplicate_step_count,
        step_sources=step_sources,
    )


def first_finite_metric(metrics: Mapping[str, object], keys: Sequence[str]) -> float:
    for key in keys:
        value = finite_float(metrics.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def add_derived_metrics(
    per_step: MutableMapping[int, Dict[str, object]],
    steps: Sequence[int],
) -> None:
    cum_real_env_samples = 0.0
    cum_policy_samples = 0.0
    cum_wm_samples = 0.0
    score_specs = (
        (
            (
                "val/success_rate/all",
                "success_rate/all",
                "val/test_score/all",
                "test_score/all",
            ),
            "eval_success_rate_all",
        ),
        (
            (
                "val/success_rate/libero_10",
                "success_rate/libero_10",
                "val/test_score/libero_10",
                "test_score/libero_10",
            ),
            "eval_success_rate_libero_10",
        ),
        (
            (
                "rollout/real_success_rate",
                "train/real_success_rate",
                "rollout/success_rate",
            ),
            "rollout_real_success_rate",
        ),
    )
    prev_scores: Dict[str, float] = {}
    prev_score_real_samples: Dict[str, float] = {}
    learning_auc_state: Dict[str, Dict[str, object]] = {}
    r2t_thresholds = (0.3, 0.4, 0.5)
    for step in steps:
        metrics = per_step[step]
        train_reward_main = first_finite_metric(
            metrics,
            (
                "train_reward/reward_all",
                "train_reward/all",
                "train_reward/reward_model",
                "train_reward/verifier",
            ),
        )
        if math.isfinite(train_reward_main):
            metrics["train_reward/main"] = train_reward_main

        real_step = first_finite_metric(
            metrics,
            (
                "rollout/real_num_samples",
                "wm/num_real_sample",
                "sample_efficiency/real_env_samples_step",
                "env/real_sample_actual",
                "env/real_sample_target",
            ),
        )
        if not math.isfinite(real_step):
            wm_ratio = first_finite_metric(metrics, ("wm/ratio_wm",))
            wm_step_hint = first_finite_metric(
                metrics,
                (
                    "wm/num_wm_sample",
                    "sample_efficiency/wm_samples_step",
                ),
            )
            if (not math.isfinite(wm_ratio) or wm_ratio <= 0.0) and (
                not math.isfinite(wm_step_hint) or wm_step_hint <= 0.0
            ):
                real_step = first_finite_metric(metrics, ("rollout/num_samples",))
        policy_step = first_finite_metric(
            metrics,
            (
                "env/policy_sample_target",
                "env/total_sample_target",
                "wm/mixed_size",
            ),
        )
        wm_step = first_finite_metric(
            metrics,
            (
                "wm/num_wm_sample",
                "sample_efficiency/wm_samples_step",
            ),
        )
        if math.isfinite(real_step):
            cum_real_env_samples += max(0.0, real_step)
            metrics["sample_efficiency/real_env_samples_step"] = real_step
        if math.isfinite(policy_step):
            cum_policy_samples += max(0.0, policy_step)
            metrics["sample_efficiency/policy_samples_step"] = policy_step
        if math.isfinite(wm_step):
            cum_wm_samples += max(0.0, wm_step)
            metrics["sample_efficiency/wm_samples_step"] = wm_step

        metrics["sample_efficiency/cum_real_env_samples"] = cum_real_env_samples
        metrics["sample_efficiency/cum_policy_samples"] = cum_policy_samples
        metrics["sample_efficiency/cum_wm_samples"] = cum_wm_samples

        score = first_finite_metric(
            metrics,
            (
                "val/success_rate/all",
                "success_rate/all",
                "val/success_rate/libero_10",
                "success_rate/libero_10",
                "val/test_score/all",
                "test_score/all",
                "val/test_score/libero_10",
                "test_score/libero_10",
                "rollout/real_success_rate",
                "rollout/success_rate",
            ),
        )
        if math.isfinite(score) and cum_real_env_samples > 0:
            metrics["sample_efficiency/success_per_1k_real_samples"] = (
                score * 1000.0 / cum_real_env_samples
            )

        for score_keys, score_name in score_specs:
            score_value = first_finite_metric(metrics, score_keys)
            if not math.isfinite(score_value):
                continue
            training_state = learning_auc_state.setdefault(
                score_name,
                {
                    "first_real": math.nan,
                    "prev_real": math.nan,
                    "prev_score": math.nan,
                    "area": 0.0,
                    "r2t": {eta: math.nan for eta in r2t_thresholds},
                },
            )
            if cum_real_env_samples > 0:
                first_real = float(training_state["first_real"])
                prev_real = float(training_state["prev_real"])
                prev_score = float(training_state["prev_score"])
                if not math.isfinite(first_real):
                    training_state["first_real"] = cum_real_env_samples
                    training_state["prev_real"] = cum_real_env_samples
                    training_state["prev_score"] = score_value
                else:
                    delta_real_for_auc = cum_real_env_samples - prev_real
                    if delta_real_for_auc > 0 and math.isfinite(prev_score):
                        training_state["area"] = float(training_state["area"]) + (
                            0.5 * (prev_score + score_value) * delta_real_for_auc
                        )
                    training_state["prev_real"] = cum_real_env_samples
                    training_state["prev_score"] = score_value

                denom = cum_real_env_samples - float(training_state["first_real"])
                auc_value = (
                    float(training_state["area"]) / denom
                    if denom > 0
                    else score_value
                )
                metrics[f"learning_metrics/{score_name}_sr"] = score_value
                metrics[f"learning_metrics/{score_name}_sr_pct"] = score_value * 100.0
                metrics[f"learning_metrics/{score_name}_auc"] = auc_value
                metrics[f"learning_metrics/{score_name}_auc_pct"] = auc_value * 100.0

                r2t_state = training_state["r2t"]
                if not isinstance(r2t_state, dict):
                    r2t_state = {}
                    training_state["r2t"] = r2t_state
                for eta in r2t_thresholds:
                    r2t_state.setdefault(eta, math.nan)
                    if (
                        not math.isfinite(float(r2t_state[eta]))
                        and score_value >= eta
                    ):
                        r2t_state[eta] = cum_real_env_samples
                    if math.isfinite(float(r2t_state[eta])):
                        eta_label = str(eta).replace(".", "p")
                        metrics[
                            f"learning_metrics/{score_name}_r2t_{eta_label}"
                        ] = float(r2t_state[eta])
            if cum_real_env_samples > 0:
                metrics[
                    f"sample_efficiency/{score_name}_per_1k_real_samples"
                ] = score_value * 1000.0 / cum_real_env_samples
            if score_name in prev_scores:
                delta_real = (
                    cum_real_env_samples - prev_score_real_samples[score_name]
                )
                if delta_real > 0:
                    metrics[
                        f"sample_efficiency/delta_{score_name}_per_1k_real_samples"
                    ] = 1000.0 * (score_value - prev_scores[score_name]) / delta_real
                    metrics[f"sample_efficiency/delta_{score_name}"] = (
                        score_value - prev_scores[score_name]
                    )
                    metrics[
                        f"sample_efficiency/delta_{score_name}_real_samples"
                    ] = delta_real
            prev_scores[score_name] = score_value
            prev_score_real_samples[score_name] = cum_real_env_samples


def resolve_key(requested_key: str, result: ParseResult) -> str:
    candidates = ALIASES.get(requested_key, (requested_key,))
    for candidate in candidates:
        values = [finite_float(result.per_step[step].get(candidate)) for step in result.steps]
        if any(math.isfinite(value) for value in values):
            return candidate
    return requested_key


def series_for_key(requested_key: str, result: ParseResult) -> Tuple[str, List[float]]:
    resolved_key = resolve_key(requested_key, result)
    display_key = requested_key if requested_key == resolved_key else f"{requested_key}->{resolved_key}"
    values = [finite_float(result.per_step[step].get(resolved_key)) for step in result.steps]
    return display_key, values


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    for index in range(len(values)):
        segment = values[max(0, index - window + 1) : index + 1]
        finite = [value for value in segment if math.isfinite(value)]
        smoothed.append(sum(finite) / len(finite) if finite else math.nan)
    return smoothed


def rolling_spread(
    values: Sequence[float],
    window: int,
    stat: str = "std",
) -> List[float]:
    if window <= 1:
        return [math.nan] * len(values)
    spreads: List[float] = []
    for index in range(len(values)):
        segment = values[max(0, index - window + 1) : index + 1]
        finite = [value for value in segment if math.isfinite(value)]
        if len(finite) < 2:
            spreads.append(math.nan)
            continue
        if stat == "mad":
            center = statistics.median(finite)
            spread = 1.4826 * statistics.median(abs(value - center) for value in finite)
        else:
            spread = statistics.pstdev(finite)
        spreads.append(spread if math.isfinite(spread) else math.nan)
    return spreads


def x_series_for_key(result: ParseResult, x_key: Optional[str]) -> Tuple[str, List[float]]:
    if not x_key:
        return "Global Step", [float(step) for step in result.steps]
    values = [finite_float(result.per_step[step].get(x_key)) for step in result.steps]
    if any(math.isfinite(value) for value in values):
        return x_key, values
    return "Global Step", [float(step) for step in result.steps]


def numeric_keys(result: ParseResult) -> List[str]:
    counts: Dict[str, int] = defaultdict(int)
    for step in result.steps:
        for key, value in result.per_step[step].items():
            if math.isfinite(finite_float(value)):
                counts[key] += 1
    return sorted(key for key, count in counts.items() if count > 0)


def key_counts(result: ParseResult) -> List[Tuple[str, int, float, float]]:
    rows = []
    for key in numeric_keys(result):
        values = [finite_float(result.per_step[step].get(key)) for step in result.steps]
        finite_values = [value for value in values if math.isfinite(value)]
        if finite_values:
            rows.append((key, len(finite_values), finite_values[0], finite_values[-1]))
    return rows


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_group_matplotlib(
    result: ParseResult,
    keys: Sequence[str],
    title: str,
    save_path: Path,
    smooth: int = 1,
    band_window: int = 5,
    band_scale: float = 1.0,
    band_stat: str = "std",
    dpi: int = 220,
    show: bool = False,
    x_key: Optional[str] = None,
) -> bool:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = 0
    marker = "o" if len(result.steps) <= 120 else None
    x_label, x_values = x_series_for_key(result, x_key)

    for requested_key in keys:
        display_key, values = series_for_key(requested_key, result)
        paired = [
            (x, y)
            for x, y in zip(x_values, values)
            if math.isfinite(x) and math.isfinite(y)
        ]
        if not paired:
            continue
        xs = [item[0] for item in paired]
        finite_values = [item[1] for item in paired]
        values_to_plot = moving_average(finite_values, smooth)
        label = display_key if smooth <= 1 else f"{display_key} (ma{smooth})"
        line = ax.plot(
            xs,
            values_to_plot,
            label=label,
            marker=marker,
            linewidth=2,
            markersize=3,
        )
        if band_window > 1 and len(finite_values) > 1:
            spreads = rolling_spread(finite_values, band_window, stat=band_stat)
            lower: List[float] = []
            upper: List[float] = []
            for center, spread in zip(values_to_plot, spreads):
                if math.isfinite(center) and math.isfinite(spread):
                    lower.append(center - band_scale * spread)
                    upper.append(center + band_scale * spread)
                else:
                    lower.append(math.nan)
                    upper.append(math.nan)
            ax.fill_between(
                xs,
                lower,
                upper,
                color=line[0].get_color(),
                alpha=0.14,
                linewidth=0,
            )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_title(title.replace("_", " ").title())
    ax.set_xlabel(x_label)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return True


def svg_path_from_series(
    x_values: Sequence[float],
    values: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
) -> str:
    commands: List[str] = []
    in_segment = False
    x_span = max(1, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    for x_value, value in zip(x_values, values):
        if not math.isfinite(x_value) or not math.isfinite(value):
            continue
        x = left + (x_value - x_min) / x_span * plot_width
        y = top + (y_max - value) / y_span * plot_height
        command = "L" if in_segment else "M"
        commands.append(f"{command}{x:.2f},{y:.2f}")
        in_segment = True
    return " ".join(commands)


def svg_band_path_from_series(
    x_values: Sequence[float],
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
) -> str:
    upper_points: List[str] = []
    lower_points: List[str] = []
    x_span = max(1, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)

    for x_value, upper in zip(x_values, upper_values):
        if not math.isfinite(x_value) or not math.isfinite(upper):
            continue
        x = left + (x_value - x_min) / x_span * plot_width
        y = top + (y_max - upper) / y_span * plot_height
        upper_points.append(f"{x:.2f},{y:.2f}")

    for x_value, lower in zip(reversed(x_values), reversed(lower_values)):
        if not math.isfinite(x_value) or not math.isfinite(lower):
            continue
        x = left + (x_value - x_min) / x_span * plot_width
        y = top + (y_max - lower) / y_span * plot_height
        lower_points.append(f"{x:.2f},{y:.2f}")

    if len(upper_points) < 2 or len(lower_points) < 2:
        return ""
    return "M" + " L".join(upper_points + lower_points) + " Z"


def plot_group_svg(
    result: ParseResult,
    keys: Sequence[str],
    title: str,
    save_path: Path,
    smooth: int = 1,
    band_window: int = 5,
    band_scale: float = 1.0,
    band_stat: str = "std",
    x_key: Optional[str] = None,
) -> bool:
    x_label, raw_x_values = x_series_for_key(result, x_key)
    series: List[Tuple[str, List[float], List[float], List[float]]] = []
    for requested_key in keys:
        display_key, values = series_for_key(requested_key, result)
        paired = [
            (x, y)
            for x, y in zip(raw_x_values, values)
            if math.isfinite(x) and math.isfinite(y)
        ]
        if paired:
            dense_values = [math.nan] * len(raw_x_values)
            smoothed = moving_average([item[1] for item in paired], smooth)
            dense_lower = [math.nan] * len(raw_x_values)
            dense_upper = [math.nan] * len(raw_x_values)
            spreads = (
                rolling_spread([item[1] for item in paired], band_window, stat=band_stat)
                if band_window > 1 and len(paired) > 1
                else [math.nan] * len(paired)
            )
            pair_iter = iter(smoothed)
            spread_iter = iter(spreads)
            for index, (x, y) in enumerate(zip(raw_x_values, values)):
                if math.isfinite(x) and math.isfinite(y):
                    center = next(pair_iter)
                    spread = next(spread_iter)
                    dense_values[index] = center
                    if math.isfinite(center) and math.isfinite(spread):
                        dense_lower[index] = center - band_scale * spread
                        dense_upper[index] = center + band_scale * spread
            series.append((display_key, dense_values, dense_lower, dense_upper))
    if not series:
        return False

    width, height = 1280, 720
    left, right, top, bottom = 84, 300, 64, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite_x = [value for value in raw_x_values if math.isfinite(value)]
    x_min, x_max = min(finite_x), max(finite_x)
    finite_values = [
        value
        for _, values, lower_values, upper_values in series
        for value in (*values, *lower_values, *upper_values)
        if math.isfinite(value)
    ]
    y_min, y_max = min(finite_values), max(finite_values)
    if y_min == y_max:
        padding = abs(y_min) * 0.05 + 1.0
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.06
        y_min -= padding
        y_max += padding

    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{html.escape(title.replace("_", " ").title())}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#222" stroke-width="1"/>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.4g}</text>')

    for tick in range(6):
        ratio = tick / 5
        x = left + ratio * plot_width
        value = x_min + ratio * (x_max - x_min)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 46}" text-anchor="middle" font-family="Arial" font-size="12">{value:.0f}</text>')

    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(x_label)}</text>')
    parts.append(f'<text transform="translate(22,{top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">Value</text>')

    for index, (label, values, lower_values, upper_values) in enumerate(series):
        color = colors[index % len(colors)]
        band_path = svg_band_path_from_series(
            raw_x_values,
            lower_values,
            upper_values,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            top,
            plot_width,
            plot_height,
        )
        if band_path:
            parts.append(
                f'<path d="{band_path}" fill="{color}" fill-opacity="0.14" stroke="none"/>'
            )
        path_data = svg_path_from_series(
            raw_x_values,
            values,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            top,
            plot_width,
            plot_height,
        )
        parts.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        legend_y = top + 22 + index * 24
        parts.append(f'<line x1="{left + plot_width + 28}" y1="{legend_y}" x2="{left + plot_width + 58}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + plot_width + 66}" y="{legend_y + 4}" font-family="Arial" font-size="12">{html.escape(label)}</text>')

    parts.append("</svg>")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(parts), encoding="utf-8")
    return True


def plot_group(
    result: ParseResult,
    keys: Sequence[str],
    title: str,
    save_path: Path,
    smooth: int = 1,
    band_window: int = 5,
    band_scale: float = 1.0,
    band_stat: str = "std",
    dpi: int = 220,
    show: bool = False,
    x_key: Optional[str] = None,
) -> Optional[Path]:
    try:
        if plot_group_matplotlib(
            result,
            keys,
            title,
            save_path,
            smooth=smooth,
            band_window=band_window,
            band_scale=band_scale,
            band_stat=band_stat,
            dpi=dpi,
            show=show,
            x_key=x_key,
        ):
            return save_path
        return None
    except Exception as exc:  # pragma: no cover - exercised only when plotting backend is broken.
        fallback_path = save_path.with_suffix(".svg")
        if plot_group_svg(
            result,
            keys,
            title,
            fallback_path,
            smooth=smooth,
            band_window=band_window,
            band_scale=band_scale,
            band_stat=band_stat,
            x_key=x_key,
        ):
            print(f"matplotlib unavailable ({exc}); saved SVG fallback: {fallback_path}")
            return fallback_path
        return None


def write_csv(result: ParseResult, csv_path: Path) -> None:
    keys = numeric_keys(result)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *keys])
        for step in result.steps:
            row = [step]
            for key in keys:
                value = finite_float(result.per_step[step].get(key))
                row.append("" if not math.isfinite(value) else value)
            writer.writerow(row)


def write_manifest(result: ParseResult, out_dir: Path, generated: Sequence[Path]) -> None:
    manifest_path = out_dir / "plot_manifest.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(f"files: {len(result.files)}\n")
        handle.write(f"records: {result.record_count}\n")
        handle.write(f"steps: {len(result.steps)}\n")
        handle.write(f"duplicate_step_records: {result.duplicate_step_count}\n")
        handle.write(f"bad_lines: {result.bad_line_count}\n")
        if result.steps:
            handle.write(f"step_range: {result.steps[0]}..{result.steps[-1]}\n")
        handle.write("\ninput_files:\n")
        for file_path in result.files:
            handle.write(f"- {file_path}\n")
        handle.write("\ngenerated_files:\n")
        for file_path in generated:
            handle.write(f"- {file_path}\n")


def write_plot_guide(
    result: ParseResult,
    out_dir: Path,
    preset: Optional[str],
    generated: Sequence[Path],
) -> Path:
    guide_path = out_dir / "README_plots.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_names = {path.name for path in generated}
    with guide_path.open("w", encoding="utf-8") as handle:
        handle.write("# Plot Guide\n\n")
        handle.write("## Data Merge\n\n")
        handle.write(
            f"- input log files: {len(result.files)}\n"
            f"- parsed records: {result.record_count}\n"
            f"- unique global steps: {len(result.steps)}\n"
            f"- duplicate step records merged: {result.duplicate_step_count}\n"
        )
        if result.steps:
            handle.write(f"- global step range: {result.steps[0]}..{result.steps[-1]}\n")
        handle.write(
            "\nThe parser merges all selected resume logs by global step. "
            "If a metric only appears every N steps, finite points are connected directly. "
            "The shaded band is a single-run rolling variability envelope, not a multi-run standard deviation.\n\n"
        )

        handle.write("## Core Comparison Metrics\n\n")
        handle.write(
            "- Learning-curve metrics: `00_learning_sr_auc.*` and `00b_learning_r2t.*` report SR, AUC, and R2T over cumulative real samples. SR/AUC curves are kept in 0-1 units; `*_pct` columns in `merged_metrics.csv` are percentage points.\n"
            "- True success rate: `01_policy_success_rate.*` and auto `val/success_rate/*` plots. Old `val/test_score/*` is used only as a fallback for historical logs.\n"
            "- Training reward: `01b_train_reward.*`, where `train_reward/main` is the PPO reward actually optimized.\n"
            "- Success vs real-sample budget: `02_success_vs_real_samples.*`.\n"
            "- Delta sample efficiency: `03_delta_sample_efficiency.*`.\n"
            "- WM video metrics: `13_wm_eval_video_metrics.*` in MERL and `06b_wm_eval_video_metrics.*` in MBRL, covering clips/FID/FVD when present.\n"
            "- Runtime cost: timing plots / `90_auto_timing_runtime.*`.\n"
            "- Rollout validity: rollout-health plots / `20_auto_rollout_health.*`.\n\n"
        )

        if preset == "merl":
            handle.write("## MERL-Specific Diagnostics\n\n")
            handle.write(
                "- Ratio scheduler: `04_merl_ratio_scheduler.*`.\n"
                "- WM confidence and imagined weights: `05_merl_confidence_weight.*`.\n"
                "- GRPO source split: `05b_merl_grpo_source_split.*`.\n"
                "- WM chunk error: `06_merl_chunk_error.*`.\n"
                "- WM video metrics: `13_wm_eval_video_metrics.*`.\n"
                "- Replay mix: `08_merl_replay_mix.*`.\n"
                "- Imagined PPO contract: `16_imagined_ppo_contract.*`.\n"
                "- Imagined reward guard: `16b_imagined_reward_guard.*`.\n"
                "- Actor hard guard: `18b_actor_hard_guard.*`.\n\n"
            )

        handle.write("## Figure Notes\n\n")
        for stem, description in PLOT_GUIDE.items():
            matched = sorted(name for name in generated_names if name.startswith(stem))
            if matched:
                handle.write(f"- `{matched[0]}`: {description}\n")
        handle.write(
            "\n`missing_preset_metrics.tsv` lists preset metrics that were absent or all-NaN in the current logs; "
            "this is expected for low-frequency eval metrics before their first interval fires.\n"
        )
    return guide_path


def print_key_table(result: ParseResult) -> None:
    print("available numeric keys:")
    for key, count, first, last in key_counts(result):
        print(f"{key}\tcount={count}\tfirst={first:.6g}\tlast={last:.6g}")


def default_out_dir(args: argparse.Namespace, files: Sequence[Path]) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    base = files[0].parent if files else Path(".")
    suffix = args.preset or "custom"
    return base / f"plots_{suffix}"


def run_preset(result: ParseResult, preset: str, out_dir: Path, args: argparse.Namespace) -> List[Path]:
    generated: List[Path] = []
    for name, keys in PRESETS[preset].items():
        save_path = out_dir / f"{name}.png"
        x_key = PRESET_X_KEYS.get(name)
        generated_path = plot_group(
            result=result,
            keys=keys,
            title=name,
            save_path=save_path,
            smooth=max(1, args.smooth),
            band_window=max(0, args.band_window),
            band_scale=max(0.0, args.band_scale),
            band_stat=args.band_stat,
            dpi=args.dpi,
            show=args.show,
            x_key=x_key,
        )
        if generated_path is not None:
            generated.append(generated_path)
            print(f"saved: {generated_path}")
        else:
            print(f"skipped: {name} (no matching numeric metrics)")
    return generated


def metric_auto_group_name(key: str) -> str:
    parts = key.split("/")
    if key.startswith("actor/") or key.startswith("actor_after/"):
        return "80_auto_actor_policy_training"
    if key.startswith("rollout/"):
        return "20_auto_rollout_health"
    if key.startswith("timing/"):
        return "90_auto_timing_runtime"
    if key.startswith("env/"):
        return "70_auto_sampling_targets"
    if key.startswith("filter/"):
        return "75_auto_filtering"
    if key.startswith("train_reward/"):
        return "15_auto_train_reward"
    if key.startswith("train/"):
        return "95_auto_train_progress"
    if key.startswith("val/"):
        return f"10_auto_val_{sanitize_filename(parts[1])}" if len(parts) > 1 else "10_auto_val"
    if key.startswith("sample_efficiency/"):
        return "05_auto_sample_efficiency"
    if key.startswith("learning_metrics/"):
        return "00_auto_learning_metrics"
    if key.startswith("wm/eval/full/"):
        return "45_auto_wm_eval_full"
    if key.startswith("wm/eval/"):
        return "40_auto_wm_eval_mini"
    if key.startswith("wm/ratio"):
        return "30_auto_merl_ratio_scheduler"
    if key.startswith("wm/chunk"):
        return "35_auto_wm_chunk_quality"
    if key.startswith("wm/actor_input"):
        return "32_auto_merl_actor_input_contract"
    if key.startswith("wm/update"):
        return "50_auto_wm_update"
    if key.startswith("wm/io"):
        return "60_auto_wm_io"
    if key.startswith("wm/cleanup"):
        return "65_auto_wm_cleanup"
    if key.startswith("wm/"):
        return "55_auto_wm_misc"
    if len(parts) >= 2:
        return f"auto_{sanitize_filename(parts[0])}_{sanitize_filename(parts[1])}"
    return f"auto_{sanitize_filename(parts[0])}"


def run_auto_prefix_groups(
    result: ParseResult,
    out_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    generated: List[Path] = []
    grouped: Dict[str, List[str]] = defaultdict(list)
    for key in numeric_keys(result):
        grouped[metric_auto_group_name(key)].append(key)

    max_keys = max(1, int(args.auto_max_keys_per_plot))
    for group_name in sorted(grouped):
        keys = sorted(grouped[group_name])
        for chunk_index in range(0, len(keys), max_keys):
            chunk = keys[chunk_index : chunk_index + max_keys]
            suffix = "" if len(keys) <= max_keys else f"_{chunk_index // max_keys + 1:02d}"
            save_path = out_dir / f"{group_name}{suffix}.png"
            generated_path = plot_group(
                result=result,
                keys=chunk,
                title=f"{group_name}{suffix}",
                save_path=save_path,
                smooth=max(1, args.smooth),
                band_window=max(0, args.band_window),
                band_scale=max(0.0, args.band_scale),
                band_stat=args.band_stat,
                dpi=args.dpi,
                show=args.show,
            )
            if generated_path is not None:
                generated.append(generated_path)
                print(f"saved: {generated_path}")
    return generated


def write_missing_report(result: ParseResult, preset: Optional[str], out_dir: Path) -> Optional[Path]:
    if not preset:
        return None
    missing_rows: List[Tuple[str, str]] = []
    for group_name, keys in PRESETS[preset].items():
        for requested_key in keys:
            _, values = series_for_key(requested_key, result)
            if not any(math.isfinite(value) for value in values):
                missing_rows.append((group_name, requested_key))
    if not missing_rows:
        return None
    report_path = out_dir / "missing_preset_metrics.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["plot_group", "missing_metric"])
        writer.writerows(missing_rows)
    return report_path


def run_custom_plot(result: ParseResult, args: argparse.Namespace, out_dir: Path) -> Optional[Path]:
    if not args.keys:
        return None
    save_path = Path(args.save_path) if args.save_path else out_dir / "custom_metrics.png"
    title = args.title or "custom_metrics"
    generated_path = plot_group(
        result=result,
        keys=args.keys,
        title=title,
        save_path=save_path,
        smooth=max(1, args.smooth),
        band_window=max(0, args.band_window),
        band_scale=max(0.0, args.band_scale),
        band_stat=args.band_stat,
        dpi=args.dpi,
        show=args.show,
    )
    if generated_path is not None:
        print(f"saved: {generated_path}")
        return generated_path
    print("custom plot skipped: no matching numeric metrics")
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot current MERL/MBRL/MFRL metrics from resumed verl JSON tracker logs."
    )
    parser.add_argument("paths", nargs="*", help="Optional log files or directories.")
    parser.add_argument("--log_file", action="append", help="Log file path. Can be repeated.")
    parser.add_argument("--log_dir", action="append", help="Directory containing resumed run logs. Can be repeated.")
    parser.add_argument("--pattern", action="append", help="Glob pattern inside --log_dir. Default: run_*.log and log.txt.")
    parser.add_argument("--recursive", action="store_true", help="Search --log_dir recursively.")
    parser.add_argument("--file_sort", choices=("mtime", "name"), default="mtime", help="Resume log merge order.")
    parser.add_argument("--preset", choices=tuple(PRESETS.keys()), help="Plot a predefined metric dashboard.")
    parser.add_argument("--out_dir", help="Output directory for preset plots and CSV.")
    parser.add_argument("--keys", nargs="+", help="Ad-hoc metrics to plot in one figure.")
    parser.add_argument("--title", default="Training Metrics over Steps", help="Title for --keys plot.")
    parser.add_argument("--save_path", help="Output image path for --keys plot.")
    parser.add_argument("--smooth", type=int, default=1, help="Trailing moving average window. Default: 1.")
    parser.add_argument("--band_window", type=int, default=5, help="Rolling variability window for shaded bands. Use 0 to disable. Default: 5.")
    parser.add_argument("--band_scale", type=float, default=1.0, help="Scale factor for the shaded variability band. Default: 1.0.")
    parser.add_argument("--band_stat", choices=("std", "mad"), default="std", help="Statistic used for the shaded variability band. Default: std.")
    parser.add_argument("--dpi", type=int, default=220, help="Saved figure DPI. Default: 220.")
    parser.add_argument("--auto_prefix_groups", action="store_true", help="Also plot every available numeric metric grouped by prefix.")
    parser.add_argument("--auto_max_keys_per_plot", type=int, default=8, help="Max metrics per auto prefix plot. Default: 8.")
    parser.add_argument("--list_keys", action="store_true", help="Print available numeric metric keys and exit unless plotting is requested.")
    parser.add_argument("--no_csv", action="store_true", help="Do not write merged_metrics.csv.")
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    files = collect_log_files(args)
    if not files:
        print("no log files found", file=sys.stderr)
        return 2

    result = parse_logs(files)
    print(f"loaded_files={len(result.files)} records={result.record_count} steps={len(result.steps)}")
    if result.steps:
        print(f"step_range={result.steps[0]}..{result.steps[-1]}")
    if result.bad_line_count:
        print(f"bad_lines={result.bad_line_count}")
    if result.duplicate_step_count:
        print(f"duplicate_step_records={result.duplicate_step_count} (latest record wins per key)")

    if args.list_keys:
        print_key_table(result)
        if not args.preset and not args.keys:
            return 0

    out_dir = default_out_dir(args, files)
    generated: List[Path] = []

    if not args.no_csv:
        csv_path = out_dir / "merged_metrics.csv"
        write_csv(result, csv_path)
        generated.append(csv_path)
        print(f"saved: {csv_path}")

    if args.preset:
        generated.extend(run_preset(result, args.preset, out_dir, args))
        missing_report = write_missing_report(result, args.preset, out_dir)
        if missing_report is not None:
            generated.append(missing_report)
            print(f"saved: {missing_report}")

    if args.auto_prefix_groups:
        generated.extend(run_auto_prefix_groups(result, out_dir, args))

    custom_path = run_custom_plot(result, args, out_dir)
    if custom_path is not None:
        generated.append(custom_path)

    if args.preset or args.keys or args.auto_prefix_groups:
        guide_path = write_plot_guide(result, out_dir, args.preset, generated)
        generated.append(guide_path)
        print(f"saved: {guide_path}")
        write_manifest(result, out_dir, generated)
    elif not args.list_keys:
        print("no plot requested; use --preset merl, --keys ..., or --auto_prefix_groups")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
