# PrioritizedPool and prioritized_mixed_sample implementation + demo
# Executable demonstration. This is a self-contained implementation that stores
# arbitrary sample objects (e.g., DataProto slices, dicts, custom objects).
#
# Usage:
# - pool.add(samples: List[Any], priorities: Optional[List[float]])
# - samples, indices, is_weights = pool.sample(n, alpha, beta, replace=False)
# - pool.update_priorities(indices, new_priorities)
# - prioritized_mixed_sample(...) returns (mixed_batch, metrics)
#
# If you want the output to be a DataProto, pass concat_fn that converts a list of
# samples to a DataProto (e.g., concat_fn = lambda samples: DataProto.concat(samples)).
#
# This implementation uses numpy probability sampling (O(N)) which is simple and
# easy to integrate. For production scale, replace sampling with a sum-tree (O(log N)).

import math
import numpy as np
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from verl import DataProto


# --------------------------
# PrioritizedPool class
# --------------------------
class PrioritizedPool:
    def __init__(self, capacity: int = 100000):
        """
        Simple prioritized pool that stores arbitrary samples and a float priority per sample.
        Capacity is enforced with circular overwrite when full.
        """
        assert capacity > 0
        self.capacity = int(capacity)
        self.buffer: List[Any] = []
        self.priorities: List[float] = []
        self.next_idx = 0  # for circular overwrite

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, samples: List[Any], priorities: Optional[List[float]] = None) -> None:
        """
        Add samples to the pool. If priorities is None, a small default priority will be used.
        samples: list of arbitrary Python objects (DataProto slices, dicts, etc.)
        priorities: list of floats same length as samples, or None
        """
        if priorities is not None:
            assert len(priorities) == len(samples), "priorities length mismatch"

        for i, s in enumerate(samples):
            p = None
            if priorities is not None:
                p = float(priorities[i])
            else:
                # default small priority
                p = 1e-6
            if len(self.buffer) < self.capacity:
                self.buffer.append(s)
                self.priorities.append(max(p, 1e-12))
            else:
                # overwrite oldest entry
                self.buffer[self.next_idx] = s
                self.priorities[self.next_idx] = max(p, 1e-12)
                self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(
        self, n: int, alpha: float = 0.6, beta: float = 0.4, replace: bool = False
    ) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample n items according to priority^alpha.
        Returns (samples, indices, is_weights)
        - is_weights: numpy array of shape (n,) containing normalized importance-sampling weights in (0, 1]
        """
        N = len(self.buffer)
        if N == 0:
            return [], [], np.array([])

        # Clip priorities for numerical stability
        ps = np.asarray(self.priorities, dtype=np.float64)
        ps = np.clip(ps, 1e-12, None)
        scaled = ps ** float(alpha)
        total = scaled.sum()
        if total <= 0 or np.isnan(total):
            # fallback to uniform
            P = np.ones_like(scaled) / scaled.size
        else:
            P = scaled / total

        # If without replacement but n >= N, force replace True to avoid error
        if not replace and n >= N:
            replace = True

        idxes = np.random.choice(np.arange(N), size=n, replace=replace, p=P)
        samples = [self.buffer[int(i)] for i in idxes]

        # importance sampling weights correction
        weights = (1.0 / (N * P[idxes])) ** float(beta)
        weights = weights / (weights.max() + 1e-12)  # normalize to (0,1]
        return samples, list(map(int, idxes)), weights

    def update_priorities(
        self, indices: List[int], new_priorities: List[float]
    ) -> None:
        """
        Update priorities at given indices. indices must be valid positions in the buffer.
        """
        assert len(indices) == len(new_priorities)
        for i, p in zip(indices, new_priorities):
            if 0 <= int(i) < len(self.priorities):
                self.priorities[int(i)] = float(max(p, 1e-12))

    def get_all_priorities(self) -> np.ndarray:
        return np.asarray(self.priorities, dtype=np.float64)


# --------------------------
# WM trust factor
# --------------------------
def wm_trust_factor(
    uncertainty: Optional[float], kappa: float = 1.0, mode: str = "exp"
) -> float:
    """
    Convert uncertainty -> trust factor tau in (0, 1].
    mode='exp' => tau = exp(-kappa * uncertainty)
    mode='inv' => tau = 1 / (1 + kappa * uncertainty)
    If uncertainty is None, returns 1.0
    """
    if uncertainty is None:
        return 1.0
    u = float(uncertainty)
    if u <= 0.0:
        return 1.0
    if mode == "exp":
        return float(math.exp(-kappa * u))
    elif mode == "inv":
        return float(1.0 / (1.0 + kappa * u))
    else:
        raise ValueError(f"unknown mode {mode} for wm_trust_factor")


def compute_priorities_from_dataproto_list(dp_list):
    prios = []
    for dp in dp_list:
        # 假设 compute_advantage 已在 later stage 执行，当前可用 advantages 字段或 token_level_scores
        if "advantages" in dp.batch:
            p = dp.batch["advantages"].abs().mean().item()
        elif "token_level_scores" in dp.batch:
            p = dp.batch["token_level_scores"].abs().mean().item()
        else:
            p = 1e-6
        prios.append(max(p, 1e-6))
    return prios


# --------------------------
# prioritized_mixed_sample implementation
# --------------------------
def prioritized_mixed_sample(
    prioritized_real_pool: PrioritizedPool,
    prioritized_wm_pool: PrioritizedPool,
    total_needed: int,
    r_wm: float,
    alpha: float = 0.6,
    beta: float = 0.4,
    wm_kappa: float = 1.0,
    wm_trust_mode: str = "exp",
    allow_replace: bool = False,
    strict_preferred_source: bool = False,
    ratio_rounding: str = "stochastic",
    ratio_carry: float = 0.0,
    concat_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Construct mixed batch from two prioritized pools.
    - prioritized_real_pool / prioritized_wm_pool: PrioritizedPool instances
    - total_needed: total samples required (int)
    - r_wm: target fraction for WM samples in [0,1]
        - ratio_rounding: "stochastic" preserves expected r_wm for small batches;
            "floor" keeps the old deterministic behavior; "ceil" is aggressive;
            "carry" accumulates fractional target WM counts across updates.
    - concat_fn: optional function(List[samples]) -> combined batch (e.g., DataProto.concat)
      If concat_fn is None, will return the list of samples as mixed_batch.
    Returns: (mixed_batch, metrics)
    """
    assert total_needed > 0
    metrics: Dict[str, Any] = {}

    # compute target counts. Small PPO batches can quantize the WM branch too
    # aggressively, so optionally carry the fractional WM target into later
    # updates instead of repeatedly losing it to per-step integer rounding.
    clipped_r_wm = float(max(0.0, min(1.0, r_wm)))
    raw_wm_to_take = total_needed * clipped_r_wm
    carry_in = float(max(0.0, ratio_carry)) if ratio_rounding == "carry" else 0.0
    if clipped_r_wm <= 0.0:
        n_wm_to_take = 0
    elif clipped_r_wm >= 1.0:
        n_wm_to_take = total_needed
    elif ratio_rounding == "stochastic":
        base = int(math.floor(raw_wm_to_take))
        frac = float(raw_wm_to_take - base)
        n_wm_to_take = base + int(np.random.random() < frac)
    elif ratio_rounding == "ceil":
        n_wm_to_take = int(math.ceil(raw_wm_to_take))
    elif ratio_rounding == "floor":
        n_wm_to_take = int(math.floor(raw_wm_to_take))
    elif ratio_rounding == "carry":
        effective_wm_to_take = raw_wm_to_take + carry_in
        n_wm_to_take = int(math.floor(effective_wm_to_take))
    else:
        raise ValueError(f"unknown ratio_rounding={ratio_rounding}")
    n_wm_to_take = int(max(0, min(total_needed, n_wm_to_take)))
    carry_out = 0.0
    if ratio_rounding == "carry":
        carry_out = float(max(0.0, raw_wm_to_take + carry_in - n_wm_to_take))
    n_real_to_take = total_needed - n_wm_to_take
    metrics["wm/target_wm_sample_float"] = float(raw_wm_to_take)
    metrics["wm/target_wm_sample_count"] = int(n_wm_to_take)
    metrics["wm/target_real_sample_count"] = int(n_real_to_take)
    metrics["wm/ratio_rounding_stochastic"] = (
        1.0 if ratio_rounding == "stochastic" else 0.0
    )
    metrics["wm/ratio_rounding_carry"] = 1.0 if ratio_rounding == "carry" else 0.0
    metrics["wm/ratio_carry_in"] = float(carry_in)
    metrics["wm/ratio_carry_out"] = float(carry_out)

    real_avail = len(prioritized_real_pool)
    wm_avail = len(prioritized_wm_pool)

    pure_real_mode = n_real_to_take == total_needed and n_wm_to_take == 0
    pure_wm_mode = n_wm_to_take == total_needed and n_real_to_take == 0

    # Clamp to available unless replacement is explicitly allowed.  MERL uses
    # replacement for anchored WM so a small but valid anchor pool can preserve
    # the scheduled ratio without letting unanchored WM occupy actor capacity.
    if allow_replace:
        take_real = n_real_to_take if real_avail > 0 else 0
        take_wm = n_wm_to_take if wm_avail > 0 else 0
    else:
        take_real = min(n_real_to_take, real_avail)
        take_wm = min(n_wm_to_take, wm_avail)

    if strict_preferred_source:
        if pure_real_mode and real_avail > 0:
            take_real = total_needed
            take_wm = 0
            allow_replace = allow_replace or (take_real > real_avail)
        elif pure_wm_mode and wm_avail > 0:
            take_real = 0
            take_wm = total_needed
            allow_replace = allow_replace or (take_wm > wm_avail)

    remaining = total_needed - (take_real + take_wm)
    # preferentially fill remaining from real pool
    if remaining > 0:
        real_room = remaining if allow_replace and real_avail > 0 else real_avail - take_real
        extra_from_real = min(remaining, max(0, real_room))
        take_real += extra_from_real
        remaining -= extra_from_real
    if remaining > 0:
        wm_room = remaining if allow_replace and wm_avail > 0 else wm_avail - take_wm
        extra_from_wm = min(remaining, max(0, wm_room))
        take_wm += extra_from_wm
        remaining -= extra_from_wm

    # if still remaining, allow replacement as last resort
    if remaining > 0:
        allow_replace = True

    mixed_samples: List[Any] = []
    mixed_indices_info: Dict[str, List[int]] = {"real": [], "wm": []}
    mixed_weights: List[float] = []

    # sample real
    if take_real > 0 and real_avail > 0:
        real_samples, real_idxes, real_weights = prioritized_real_pool.sample(
            take_real, alpha=alpha, beta=beta, replace=allow_replace
        )
        mixed_samples.extend(real_samples)
        mixed_indices_info["real"].extend(real_idxes)
        mixed_weights.extend(
            real_weights.tolist()
            if len(real_weights) > 0
            else [1.0] * len(real_samples)
        )

    # sample wm: adjust priorities by trust factor before sampling
    if take_wm > 0 and wm_avail > 0:
        # compute adjusted probabilities for wm manually to incorporate per-sample trust
        ps = np.asarray(prioritized_wm_pool.priorities, dtype=np.float64)
        ps = np.clip(ps, 1e-12, None)
        taus = np.ones_like(ps)
        # if samples store 'uncertainty' in a dict-like field, use it
        for idx, s in enumerate(prioritized_wm_pool.buffer):
            # try common attribute names; keep robust to arbitrary sample type
            u = None
            if isinstance(s, dict):
                u = s.get("uncertainty", None)
            else:
                # try attribute access
                u = (
                    getattr(s, "uncertainty", None)
                    if hasattr(s, "uncertainty")
                    else None
                )
                # also allow nested .data dict (e.g., earlier SimpleSample)
                if (
                    u is None
                    and hasattr(s, "data")
                    and isinstance(getattr(s, "data"), dict)
                ):
                    u = s.data.get("uncertainty", None)
            taus[idx] = wm_trust_factor(u, kappa=wm_kappa, mode=wm_trust_mode)

        adjusted = (ps * taus) ** float(alpha)
        total_adj = adjusted.sum()
        if total_adj <= 0 or np.isnan(total_adj):
            P = np.ones_like(adjusted) / adjusted.size
        else:
            P = adjusted / total_adj

        # if not replace and take_wm >= wm_avail then force replace True
        replace = allow_replace or (not allow_replace and take_wm >= wm_avail)

        idxes = np.random.choice(np.arange(len(P)), size=take_wm, replace=replace, p=P)
        wm_idxes = list(map(int, idxes))
        wm_samples = [prioritized_wm_pool.buffer[i] for i in wm_idxes]
        # importance sampling weights for wm (using P used for sampling)
        wm_weights = (1.0 / (len(P) * P[idxes])) ** float(beta)
        wm_weights = wm_weights / (wm_weights.max() + 1e-12)

        mixed_samples.extend(wm_samples)
        mixed_indices_info["wm"].extend(wm_idxes)
        mixed_weights.extend(wm_weights.tolist())

    # If still lacking, pad from whichever pool has items (with replacement)
    if len(mixed_samples) < total_needed:
        need = total_needed - len(mixed_samples)
        if real_avail > 0:
            s, idxs, w = prioritized_real_pool.sample(
                need, alpha=alpha, beta=beta, replace=True
            )
            mixed_samples.extend(s)
            mixed_indices_info["real"].extend(idxs)
            mixed_weights.extend(w.tolist())
        elif wm_avail > 0:
            s, idxs, w = prioritized_wm_pool.sample(
                need, alpha=alpha, beta=beta, replace=True
            )
            mixed_samples.extend(s)
            mixed_indices_info["wm"].extend(idxs)
            mixed_weights.extend(w.tolist())

    # Trim to exact total_needed
    mixed_samples = mixed_samples[:total_needed]
    mixed_weights = mixed_weights[:total_needed]

    metrics["wm/num_wm_sample"] = int(
        sum(
            1
            for s in mixed_samples
            if getattr(s, "is_wm", False)
            or (isinstance(s, dict) and s.get("is_wm", False))
        )
    )
    metrics["wm/num_real_sample"] = total_needed - metrics["wm/num_wm_sample"]
    metrics["wm/real_avail"] = real_avail
    metrics["wm/wm_avail"] = wm_avail
    metrics["wm/mixed_size"] = len(mixed_samples)
    metrics["wm/is_weights_mean"] = float(
        np.mean(mixed_weights) if len(mixed_weights) > 0 else 0.0
    )

    # attach final loss weight into samples by combining replay importance weight
    # with any pre-computed sample_weight carried by the sample itself.
    for s, w in zip(mixed_samples, mixed_weights):
        base_weight = 1.0
        if isinstance(s, dict):
            base_weight = float(s.get("sample_weight", s.get("is_weight", 1.0)))
            s["is_weight"] = float(base_weight) * float(w)
        else:
            if hasattr(s, "data") and isinstance(getattr(s, "data"), dict):
                base_weight = float(
                    s.data.get("sample_weight", s.data.get("is_weight", 1.0))
                )
                s.data["is_weight"] = float(base_weight) * float(w)
            else:
                # best-effort: set attribute
                try:
                    base_weight = float(getattr(s, "sample_weight", 1.0))
                    setattr(s, "is_weight", float(base_weight) * float(w))
                except Exception:
                    pass

    # if concat_fn provided, return concat_fn(mixed_samples), else return mixed_samples list
    mixed_batch = concat_fn(mixed_samples) if concat_fn is not None else mixed_samples
    return mixed_batch, metrics


def push_dataproto_to_prioritized_pool(
    pool: PrioritizedPool,
    dataproto: DataProto,
    is_wm: bool,
):
    """
    每一个 trajectory / rollout 作为一个 sample
    priority 初始值用 |adv| 或 |reward|
    """
    samples = []
    priorities = []

    batch = dataproto.batch
    B = len(dataproto)

    for i in range(B):
        sample = {
            "dataproto": dataproto.slice(i, i + 1),
            "is_wm": is_wm,
        }

        if "advantages" in batch:
            p = batch["advantages"][i].abs().mean().item()
        elif "reward" in batch:
            p = batch["reward"][i].abs().mean().item()
        else:
            p = 1.0

        samples.append(sample)
        priorities.append(p + 1e-6)

    pool.add(samples, priorities)


# --------------------------
# Demo to show expected behavior
# --------------------------
def _make_dummy_sample(
    traj_id: str, adv: float, is_wm: bool = False, uncertainty: Optional[float] = None
):
    # simple dict-like sample; in your system this could be a DataProto slice
    return {
        "traj_id": traj_id,
        "adv": float(adv),
        "is_wm": bool(is_wm),
        "uncertainty": (None if uncertainty is None else float(uncertainty)),
    }


def demo():
    random.seed(1)
    np.random.seed(1)
    real_pool = PrioritizedPool(capacity=1000)
    wm_pool = PrioritizedPool(capacity=1000)

    # add 30 real samples with priorities proportional to |adv|
    real_samples = []
    real_priorities = []
    for i in range(30):
        adv = abs(np.random.randn()) + 0.5
        s = _make_dummy_sample(f"real_{i}", adv, is_wm=False)
        real_samples.append(s)
        real_priorities.append(float(abs(adv)))
    real_pool.add(real_samples, priorities=real_priorities)

    # add 120 wm samples with uncertainties
    wm_samples = []
    wm_priorities = []
    for i in range(120):
        adv = abs(np.random.randn()) * 0.8 + 0.2
        uncertainty = abs(np.random.randn()) * 1.5  # higher means less trusted
        s = _make_dummy_sample(f"wm_{i}", adv, is_wm=True, uncertainty=uncertainty)
        wm_samples.append(s)
        wm_priorities.append(float(abs(adv)))
    wm_pool.add(wm_samples, priorities=wm_priorities)

    total_needed = 16
    r_wm = 0.6
    mixed_batch, metrics = prioritized_mixed_sample(
        prioritized_real_pool=real_pool,
        prioritized_wm_pool=wm_pool,
        total_needed=total_needed,
        r_wm=r_wm,
        alpha=0.6,
        beta=0.4,
        wm_kappa=1.2,
        wm_trust_mode="exp",
        allow_replace=False,
        concat_fn=None,  # return Python list; replace with DataProto.concat if available
    )

    print("metrics:", metrics)
    print("mixed traj ids:", [s["traj_id"] for s in mixed_batch])
    print("is_weights:", [s.get("is_weight") for s in mixed_batch])


if __name__ == "__main__":
    # run demo
    demo()
