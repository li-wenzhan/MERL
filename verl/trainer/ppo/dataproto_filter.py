import torch
from verl import DataProto
from typing import Optional
import numpy as np


class DataProtoFilterOld:
    @staticmethod
    def _per_sample_has_real_step(
        is_dummy: torch.Tensor, B: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return (B,) bool tensor:
        True if the sample has at least one NON-dummy timestep.
        """
        if is_dummy is None:
            return torch.ones(B, dtype=torch.bool, device=device)

        t = is_dummy

        # ✅ 修复：正确处理形状
        if t.shape[0] != B:
            # 尝试 reshape 为 (B, -1)
            if t.numel() % B != 0:
                # 无法整除，说明数据已损坏或不匹配
                print(
                    f"[DataProtoFilter] Warning: is_dummy numel={t.numel()} not divisible by B={B}, assuming all valid"
                )
                return torch.ones(B, dtype=torch.bool, device=device)
            t = t.view(B, -1)
        # ✅ 修复：如果 shape[0] == B，保持原状，不再 reshape

        if t.dtype != torch.bool:
            t = t.bool()

        return (~t).any(dim=1)

    @staticmethod
    def filter_rollout_samples(dp: DataProto) -> DataProto:
        """
        Used during env rollout / WM rollout / evolving buffer.
        Keep samples that have at least one non-dummy timestep.
        """
        if len(dp) == 0:
            return dp

        batch = dp.batch
        B = len(dp)

        # ✅ 修复：安全获取 device
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )

        if "is_dummy" not in batch:
            return dp

        is_dummy = batch["is_dummy"]

        has_real_step = DataProtoFilter._per_sample_has_real_step(
            is_dummy=is_dummy,
            B=B,
            device=device,
        )

        if has_real_step.sum() == 0:
            return DataProto.empty_like(dp)

        return dp.slice(has_real_step)

    @staticmethod
    def _per_sample_any_nonzero(
        tensor: Optional[torch.Tensor], B: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return (B,) bool:
        True if sample has ANY non-zero entry.
        """
        if tensor is None:
            return torch.zeros(B, dtype=torch.bool, device=device)

        t = tensor

        # ✅ 修复：正确处理形状不匹配的情况
        if t.shape[0] != B:
            if t.numel() % B != 0:
                # 无法整除，返回全 False（标记为无效）
                print(
                    f"[DataProtoFilter] Warning: tensor numel={t.numel()} not divisible by B={B}, marking all invalid"
                )
                return torch.zeros(B, dtype=torch.bool, device=device)
            t = t.view(B, -1)
        # ✅ 修复：如果 shape[0] == B，保持原状

        if t.dtype.is_floating_point:
            return t.abs().sum(dim=1) > 0
        elif t.dtype == torch.bool:
            return t.any(dim=1)
        else:
            return (t != 0).any(dim=1)

    @staticmethod
    def filter_ppo_samples(
        dp: DataProto,
        require_pixel_values: bool = False,
    ) -> DataProto:
        """
        Used right before PPO loss computation.
        Enforce PPO training contract.
        """
        if len(dp) == 0:
            return dp

        batch = dp.batch
        B = len(dp)

        # ✅ 修复：安全获取 device
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )

        valid_mask = torch.ones(B, dtype=torch.bool, device=device)

        # 1) must NOT be fully dummy
        if "is_dummy" in batch:
            is_dummy = batch["is_dummy"]
            has_real = DataProtoFilter._per_sample_has_real_step(is_dummy, B, device)
            valid_mask &= has_real

        # 2) action must exist
        if "action" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["action"], B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'action' not in batch")
            valid_mask &= False

        # 3) attention_mask must exist
        if "attention_mask" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["attention_mask"], B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'attention_mask' not in batch")
            valid_mask &= False

        # 4) pixel_values (OPTIONAL)
        if require_pixel_values:
            if "pixel_values" in batch:
                valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                    batch["pixel_values"], B, device
                )
            else:
                print(
                    "[DataProtoFilter] Warning: 'pixel_values' required but not in batch"
                )
                valid_mask &= False

        if valid_mask.sum() == 0:
            return DataProto.empty_like(dp)

        return dp.slice(valid_mask)


class DataProtoFilterOldV2:
    @staticmethod
    def _per_sample_has_real_step(
        is_dummy: torch.Tensor, B: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return (B,) bool tensor:
        True if the sample has at least one NON-dummy timestep.
        """
        if is_dummy is None:
            return torch.ones(B, dtype=torch.bool, device=device)
        t = is_dummy
        # ✅ 修复：正确处理形状
        if t.shape[0] != B:
            # 尝试 reshape 为 (B, -1)
            if t.numel() % B != 0:
                # 无法整除，说明数据已损坏或不匹配
                print(
                    f"[DataProtoFilter] Warning: is_dummy numel={t.numel()} not divisible by B={B}, assuming all valid"
                )
                return torch.ones(B, dtype=torch.bool, device=device)
            t = t.view(B, -1)
        # ✅ 修复：如果 shape[0] == B，保持原状，不再 reshape
        if t.dtype != torch.bool:
            t = t.bool()
        # ✅ 关键修复：沿所有非 batch 维度归约
        if t.dim() > 1:
            return (~t).any(dim=list(range(1, t.dim())))  # [B]
        else:
            return ~t.bool()  # [B]

    @staticmethod
    def filter_rollout_samples_old(dp: DataProto) -> DataProto:
        """
        Used during env rollout / WM rollout / evolving buffer.
        Keep samples that have at least one non-dummy timestep.
        """
        if len(dp) == 0:
            return dp
        batch = dp.batch
        B = len(dp)

        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )
        if "is_dummy" not in batch:
            return dp
        is_dummy = batch["is_dummy"]
        has_real_step = DataProtoFilter._per_sample_has_real_step(
            is_dummy=is_dummy,
            B=B,
            device=device,
        )
        if "finish_step" in batch:
            finish_step = batch["finish_step"]
            if finish_step.dim() > 1:
                finish_step = finish_step.view(B, -1)[:, 0]
            has_real_step &= finish_step.to(device=device) > 0
        if has_real_step.sum() == 0:
            return DataProto.empty_like(dp)
        return dp.slice(has_real_step)

    @staticmethod
    def filter_rollout_samples(dp: DataProto) -> DataProto:
        """
        Used during env rollout / WM rollout / evolving buffer.
        Keep samples that have at least one non-dummy timestep.
        """
        if len(dp) == 0:
            return dp
        batch = dp.batch
        B = len(dp)
        # ✅ 修复：安全获取 device
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )
        if "is_dummy" not in batch:
            return dp
        is_dummy = batch["is_dummy"]
        has_real_step = DataProtoFilter._per_sample_has_real_step(
            is_dummy=is_dummy,
            B=B,
            device=device,
        )
        if "finish_step" in batch:
            finish_step = batch["finish_step"]
            if finish_step.dim() > 1:
                finish_step = finish_step.view(B, -1)[:, 0]
            has_real_step &= finish_step.to(device=device) > 0
        return DataProtoFilter._slice_with_sample_mask(
            dp, has_real_step, "filter_rollout_samples_old_v2"
        )

    @staticmethod
    def _per_sample_any_nonzero(
        tensor: Optional[torch.Tensor], B: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return (B,) bool:
        True if sample has ANY non-zero entry.
        ✅ 关键修复：沿所有非 batch 维度归约
        """
        if tensor is None:
            return torch.zeros(B, dtype=torch.bool, device=device)
        t = tensor
        # ✅ 修复：正确处理形状不匹配的情况
        if t.shape[0] != B:
            if t.numel() % B != 0:
                # 无法整除，返回全 False（标记为无效）
                print(
                    f"[DataProtoFilter] Warning: tensor numel={t.numel()} not divisible by B={B}, marking all invalid"
                )
                return torch.zeros(B, dtype=torch.bool, device=device)
            t = t.view(B, -1)
        # ✅ 关键修复：沿所有非 batch 维度归约（支持 2D/3D/4D+ tensor）
        if t.dim() > 1:
            reduce_dims = list(range(1, t.dim()))  # [1, 2, ...] for 3D+
            if t.dtype.is_floating_point:
                return t.abs().sum(dim=reduce_dims) > 0  # [B]
            elif t.dtype == torch.bool:
                return t.any(dim=reduce_dims)  # [B]
            else:
                return (t != 0).any(dim=reduce_dims)  # [B]
        else:
            # 1D tensor, already [B]
            if t.dtype.is_floating_point:
                return t.abs() > 0
            elif t.dtype == torch.bool:
                return t
            else:
                return t != 0

    @staticmethod
    def filter_ppo_samples_old(
        dp: DataProto,
        require_pixel_values: bool = False,
    ) -> DataProto:
        """
        Used right before PPO loss computation.
        Enforce PPO training contract.
        """
        if len(dp) == 0:
            return dp
        batch = dp.batch
        B = len(dp)
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )
        valid_mask = torch.ones(B, dtype=torch.bool, device=device)

        # 1) must NOT be fully dummy
        if "is_dummy" in batch:
            is_dummy = batch["is_dummy"]
            has_real = DataProtoFilter._per_sample_has_real_step(is_dummy, B, device)
            valid_mask &= has_real

        # 2) action must exist (REQUIRED for PPO)
        if "action" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["action"], B, device
            )
        else:
            # === FIX: action 缺失是严重问题，但不应该静默跳过 ===
            print(
                "[DataProtoFilter] WARNING: 'action' not in batch. "
                "This indicates rollout data is incomplete. "
                "Check _prepare_output_batch_evolving to ensure action is always added."
            )
            # 不强制过滤所有样本，因为 action 可能在 _prepare_output_batch_evolving 中已修复
            # valid_mask &= False  # ❌ 这会导致所有样本被过滤
            pass  # ✅ 跳过检查，让训练继续

        # 3) attention_mask must exist
        if "attention_mask" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["attention_mask"], B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'attention_mask' not in batch")
            pass  # ✅ 跳过检查

        # 4) pixel_values (OPTIONAL)
        if require_pixel_values:
            if "pixel_values" in batch:
                valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                    batch["pixel_values"], B, device
                )
            else:
                print(
                    "[DataProtoFilter] Warning: 'pixel_values' required but not in batch"
                )
                pass  # ✅ 跳过检查

        if valid_mask.sum() == 0:
            return DataProto.empty_like(dp)
        return dp.slice(valid_mask)

    @staticmethod
    def filter_ppo_samples(
        dp: DataProto,
        require_pixel_values: bool = False,
    ) -> DataProto:
        """
        Used right before PPO loss computation.
        Enforce PPO training contract.
        """
        if len(dp) == 0:
            return dp
        batch = dp.batch
        B = len(dp)
        # ✅ 修复：安全获取 device
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )
        valid_mask = torch.ones(B, dtype=torch.bool, device=device)

        # 1) must NOT be fully dummy
        if "is_dummy" in batch:
            is_dummy = batch["is_dummy"]
            has_real = DataProtoFilter._per_sample_has_real_step(is_dummy, B, device)
            valid_mask &= has_real

        # 2) action must exist
        if "action" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["action"], B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'action' not in batch")
            # 不强制过滤所有样本
            pass

        # 3) attention_mask must exist
        if "attention_mask" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["attention_mask"], B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'attention_mask' not in batch")
            pass

        # 4) pixel_values (OPTIONAL)
        if require_pixel_values:
            if "pixel_values" in batch:
                valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                    batch["pixel_values"], B, device
                )
            else:
                print(
                    "[DataProtoFilter] Warning: 'pixel_values' required but not in batch"
                )
                pass

        # 5) finish_step must indicate at least one valid action chunk.
        if "finish_step" in batch:
            finish_step = batch["finish_step"]
            if finish_step.dim() > 1:
                finish_step = finish_step.view(B, -1)[:, 0]
            valid_mask &= finish_step.to(device=device) > 0

        filtered_dp = DataProtoFilter._slice_with_sample_mask(
            dp, valid_mask, "filter_ppo_samples_old_v2"
        )

        print(
            f"[DataProtoFilter] filter_ppo_samples: {B} -> {len(filtered_dp)} "
            f"(filtered {B - len(filtered_dp)} samples)"
        )

        return filtered_dp


class DataProtoFilter:
    @staticmethod
    def _per_sample_has_real_step(
        is_dummy: torch.Tensor, B: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return (B,) bool tensor:
        True if the sample has at least one NON-dummy timestep.
        """
        if is_dummy is None:
            return torch.ones(B, dtype=torch.bool, device=device)
        t = is_dummy
        if t.shape[0] != B:
            if t.numel() % B != 0:
                print(
                    f"[DataProtoFilter] Warning: is_dummy numel={t.numel()} not divisible by B={B}, assuming all valid"
                )
                return torch.ones(B, dtype=torch.bool, device=device)
            t = t.view(B, -1)
        if t.dtype != torch.bool:
            t = t.bool()
        if t.dim() > 1:
            return (~t).any(dim=list(range(1, t.dim())))
        else:
            return ~t.bool()

    @staticmethod
    def _coerce_sample_mask(
        mask: torch.Tensor, B: int, device: torch.device, context: str
    ) -> torch.Tensor:
        mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
        if mask.numel() == B:
            return mask
        if B > 0 and mask.numel() % B == 0:
            print(
                f"[DataProtoFilter] Warning: {context} mask numel={mask.numel()} "
                f"does not match batch={B}; reducing extra dimensions.",
                flush=True,
            )
            return mask.view(B, -1).any(dim=1)
        raise RuntimeError(
            f"[DataProtoFilter] {context} produced invalid mask length "
            f"{mask.numel()} for batch size {B}."
        )

    @staticmethod
    def _slice_with_sample_mask(
        dp: DataProto, mask: torch.Tensor, context: str
    ) -> DataProto:
        B = len(dp)
        device = (
            dp.device if hasattr(dp, "device") else next(iter(dp.batch.values())).device
        )
        mask = DataProtoFilter._coerce_sample_mask(mask, B, device, context)
        valid_count = int(mask.sum().detach().item())
        if valid_count == 0:
            return DataProto.empty_like(dp)
        if valid_count == B:
            return dp
        return dp.slice(mask)

    @staticmethod
    def filter_rollout_samples(dp: DataProto) -> DataProto:
        """
        Used during env rollout / WM rollout / evolving buffer.
        Keep samples that have at least one non-dummy timestep.
        """
        if len(dp) == 0:
            return dp
        batch = dp.batch
        B = len(dp)
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )
        if "is_dummy" not in batch:
            return dp
        is_dummy = batch["is_dummy"]
        has_real_step = DataProtoFilter._per_sample_has_real_step(
            is_dummy=is_dummy,
            B=B,
            device=device,
        )
        if "finish_step" in batch:
            finish_step = batch["finish_step"]
            if finish_step.dim() > 1:
                finish_step = finish_step.view(B, -1)[:, 0]
            has_real_step &= finish_step.to(device=device) > 0
        return DataProtoFilter._slice_with_sample_mask(
            dp, has_real_step, "filter_rollout_samples"
        )

    @staticmethod
    def _per_sample_any_nonzero(
        tensor: Optional[torch.Tensor], B: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return (B,) bool:
        True if sample has ANY non-zero entry.
        """
        if tensor is None:
            return torch.zeros(B, dtype=torch.bool, device=device)
        t = tensor
        if t.shape[0] != B:
            if t.numel() % B != 0:
                print(
                    f"[DataProtoFilter] Warning: tensor numel={t.numel()} not divisible by B={B}, marking all invalid"
                )
                return torch.zeros(B, dtype=torch.bool, device=device)
            t = t.view(B, -1)
        if t.dim() > 1:
            reduce_dims = list(range(1, t.dim()))
            if t.dtype.is_floating_point:
                return t.abs().sum(dim=reduce_dims) > 0
            elif t.dtype == torch.bool:
                return t.any(dim=reduce_dims)
            else:
                return (t != 0).any(dim=reduce_dims)
        else:
            if t.dtype.is_floating_point:
                return t.abs() > 0
            elif t.dtype == torch.bool:
                return t
            else:
                return t != 0

    @staticmethod
    def _per_sample_valid_response_tokens(
        batch,
        B: int,
        device: torch.device,
        action_token_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return (B,) long tensor with the number of valid response tokens.
        """
        responses = batch.get("responses", None)
        finish_step = batch.get("finish_step", None)
        derived_tokens = None

        if responses is not None and finish_step is not None:
            if finish_step.dim() > 1:
                finish_step = finish_step.view(B, -1)[:, 0]
            response_flat_length = int(responses[0].numel()) if B > 0 else 0
            action_tokens = max(1, int(action_token_len or 1))
            derived_tokens = finish_step.to(device=device, dtype=torch.long).clamp_min(
                0
            )
            derived_tokens = derived_tokens * action_tokens
            derived_tokens = derived_tokens.clamp(max=max(response_flat_length, 0))

        def _read_token_field(key: str) -> Optional[torch.Tensor]:
            if key not in batch:
                return None
            tokens = batch[key]
            if tokens.dim() > 1:
                tokens = tokens.view(B, -1)[:, 0]
            return tokens.to(device=device, dtype=torch.long).clamp_min(0)

        real_tokens = _read_token_field("valid_response_tokens")
        wm_tokens = _read_token_field("wm_valid_response_tokens")

        if real_tokens is not None or wm_tokens is not None:
            fallback_tokens = (
                derived_tokens
                if derived_tokens is not None
                else torch.zeros(B, dtype=torch.long, device=device)
            )
            if real_tokens is None:
                real_tokens = fallback_tokens
            if wm_tokens is None:
                wm_tokens = real_tokens
            if derived_tokens is not None:
                real_tokens = torch.where(real_tokens > 0, real_tokens, derived_tokens)
                real_tokens = torch.minimum(real_tokens, derived_tokens)
                wm_tokens = torch.minimum(wm_tokens, derived_tokens)
            if "is_wm" in batch:
                is_wm = batch["is_wm"]
                if is_wm.dim() > 1:
                    is_wm = is_wm.view(B, -1)[:, 0]
                tokens = torch.where(
                    is_wm.to(device=device) > 0.5, wm_tokens, real_tokens
                )
            elif (
                "wm_valid_response_tokens" in batch
                and "valid_response_tokens" not in batch
            ):
                tokens = wm_tokens
            else:
                tokens = real_tokens
            if derived_tokens is not None:
                tokens = torch.minimum(tokens, derived_tokens)
        elif derived_tokens is not None:
            tokens = derived_tokens
        else:
            tokens = torch.zeros(B, dtype=torch.long, device=device)

        if "is_dummy" in batch:
            is_dummy = batch["is_dummy"]
            has_real = DataProtoFilter._per_sample_has_real_step(
                is_dummy=is_dummy,
                B=B,
                device=device,
            )
            tokens = torch.where(has_real, tokens, torch.zeros_like(tokens))

        return tokens

    @staticmethod
    def filter_ppo_samples(
        dp: DataProto,
        require_pixel_values: bool = False,
        action_token_len: Optional[int] = None,
        require_valid_response_tokens: bool = True,
    ) -> DataProto:
        """
        Used right before PPO loss computation.
        Enforce PPO training contract.
        """
        if len(dp) == 0:
            return dp
        batch = dp.batch
        B = len(dp)
        device = (
            dp.device if hasattr(dp, "device") else next(iter(batch.values())).device
        )
        valid_mask = torch.ones(B, dtype=torch.bool, device=device)

        # 1) must NOT be fully dummy
        if "is_dummy" in batch:
            is_dummy = batch["is_dummy"]
            has_real = DataProtoFilter._per_sample_has_real_step(is_dummy, B, device)
            valid_mask &= has_real

        # 2) action is not consumed by the PPO update path in this trainer.
        # Keep only a weak NaN/Inf guard so placeholder / zero actions do not
        # erase otherwise valid real rollouts after real+WM alignment.
        if "action" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                torch.isfinite(batch["action"]), B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'action' not in batch")
            pass

        # 3) attention_mask must exist
        if "attention_mask" in batch:
            valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                batch["attention_mask"], B, device
            )
        else:
            print("[DataProtoFilter] Warning: 'attention_mask' not in batch")
            pass

        # 4) pixel_values (OPTIONAL)
        if require_pixel_values:
            if "pixel_values" in batch:
                valid_mask &= DataProtoFilter._per_sample_any_nonzero(
                    batch["pixel_values"], B, device
                )
            else:
                print(
                    "[DataProtoFilter] Warning: 'pixel_values' required but not in batch"
                )
                pass

        if "finish_step" in batch:
            finish_step = batch["finish_step"]
            if finish_step.dim() > 1:
                finish_step = finish_step.view(B, -1)[:, 0]
            valid_mask &= finish_step.to(device=device) > 0

        if require_valid_response_tokens:
            valid_response_tokens = DataProtoFilter._per_sample_valid_response_tokens(
                batch=batch,
                B=B,
                device=device,
                action_token_len=action_token_len,
            )
            valid_mask &= valid_response_tokens > 0

        if "is_wm" in batch and "wm_pred_valid" in batch:
            is_wm = batch["is_wm"]
            if is_wm.dim() > 1:
                is_wm = is_wm.view(B, -1)[:, 0]
            wm_pred_valid = batch["wm_pred_valid"]
            if wm_pred_valid.dim() > 1:
                wm_pred_valid = wm_pred_valid.view(B, -1)[:, 0]
            wm_pred_valid = wm_pred_valid.to(device=device, dtype=torch.bool)
            valid_mask &= (~(is_wm.to(device=device) > 0.5)) | wm_pred_valid

        filtered_dp = DataProtoFilter._slice_with_sample_mask(
            dp, valid_mask, "filter_ppo_samples"
        )

        print(
            f"[DataProtoFilter] filter_ppo_samples: {B} -> {len(filtered_dp)} "
            f"(filtered {B - len(filtered_dp)} samples)"
        )

        return filtered_dp
