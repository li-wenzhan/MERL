"""Explicit contracts for the online simulator baseline without trust mechanisms."""

MODES = ("MERL", "MFRL", "MBRL", "ONLINE_MBRL")
ONLINE_WM_MODES = ("MERL", "ONLINE_MBRL")


def validate_online_mbrl(wm):
    expected = dict(enable=True, fine_tune=True, use_wm_reward_proxy=True,
                    wm_real_anchor_reward=False, require_wm_anchor_reward=False,
                    zero_unanchored_wm_weight=False, weak_update_enable=False,
                    merl_imagined_reward_hard_constraint=False, imag_advantage_abs_clip=0,
                    wm_warmup_steps=0, wm_grpo_uid_mode="source")
    for key, value in expected.items():
        if wm.get(key, value) != value:
            raise ValueError(f"ONLINE_MBRL requires world_model.{key}={value!r}")


def online_mbrl_actor_contract(batch):
    """Unit loss weights on valid imagined samples; padding masks remain untouched."""
    import torch
    if not bool((batch["is_wm"] > 0.5).all()):
        raise RuntimeError("ONLINE_MBRL actor updates must contain only imagined samples")
    batch["is_weight"] = torch.ones_like(batch["is_weight"])


def require_online_wm_sync(results):
    rows = [results] if isinstance(results, dict) else results
    if not rows or any(not isinstance(row, dict) or not row.get("loaded") for row in rows):
        raise RuntimeError(f"ONLINE_MBRL failed to synchronize updated WM weights: {results}")


def require_online_real_batch(batch, expected_samples):
    actual = 0 if batch is None else len(batch)
    if actual < expected_samples:
        raise RuntimeError(f"ONLINE_MBRL real collection incomplete: {actual}/{expected_samples} valid samples. "
                           "Fix environment initialization/rollout errors before updating the actor or WM.")
