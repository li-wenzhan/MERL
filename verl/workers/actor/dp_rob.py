# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import (
    log_probs_from_logits_all_rmpad,
    logprobs_from_logits,
)
from verl.workers.actor import BasePPOActor

__all__ = ["RobDataParallelPPOActor"]


class RobDataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        self.use_proprio = bool(self.config.get("use_proprio", False))
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        print(f"Actor use_proprio={self.use_proprio}")
        print(f'PRM use dynamic bsz={self.config.get("use_dynamic_bsz", False)}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False  # self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(
            verl_F.entropy_from_logits, dynamic=True
        )

    @staticmethod
    def _slice_contract_field(
        raw_field,
        target: torch.Tensor,
        slice_id: int,
        next_slice_id: int,
        *,
        dtype: torch.dtype,
        name: str,
        allow_sample_broadcast: bool,
    ) -> torch.Tensor:
        batch_size = target.shape[0]
        field = raw_field.to(device=target.device, dtype=dtype)
        if field.dim() == 0:
            field = field.view(1, 1)
        elif field.dim() == 1:
            field = field.view(batch_size, -1)
        else:
            field = field.reshape(batch_size, -1)

        if field.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch mismatch in actor contract: "
                f"{field.shape[0]} vs {batch_size}"
            )
        if field.shape[-1] == target.shape[-1]:
            return field
        if field.shape[-1] >= next_slice_id:
            return field[:, slice_id:next_slice_id]
        if allow_sample_broadcast and field.shape[-1] == 1:
            return field.expand(-1, target.shape[-1])
        raise ValueError(
            f"{name} token length mismatch in actor contract: "
            f"got {field.shape[-1]}, need slice [{slice_id}, {next_slice_id}) "
            f"or chunk length {target.shape[-1]}"
        )

    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len

    def process_tensor_unpad_columns(self, tensor: torch.Tensor, pad_id: int):
        """
        Unpad by removing columns that are all pad across the batch dimension.
        tensor: shape (B, S, ...) or (B, S)
        Returns: tensor[:, col_mask, ...], valid_len (num kept columns)
        """
        # assume tensor shape (B, S, ...) or (B, S)
        # first get boolean mask per element along sequence dim
        mask = tensor != pad_id
        # if tensor has extra dims beyond seq dim, reduce to (B, S)
        if mask.ndim > 2:
            # keep non-pad iff any non-pad across the trailing dims (rare)
            # but typical case: token ids -> (B, S)
            mask = mask.any(dim=tuple(range(2, mask.ndim)))
        # col_mask: shape (S,) True for columns where any batch element is non-pad
        col_mask = mask.any(dim=0)
        valid_len = int(col_mask.sum().item())
        if valid_len == 0:
            raise ValueError("All tokens are padding (valid_len == 0).")
        # apply col_mask to keep only selected columns
        # handle tensors with extra dims: use advanced indexing on dim=1
        return tensor[:, col_mask, ...], valid_len

    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,),
            traj_len:
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask

    def _build_response_mask(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        responses = data["responses"]
        batch_size = responses.size(0)
        response_length = responses.size(1) * responses.size(2)

        valid_tokens = data.get("valid_response_tokens", None)
        if valid_tokens is None:
            valid_tokens = data.get("wm_valid_response_tokens", None)

        if valid_tokens is not None:
            if valid_tokens.dim() > 1:
                valid_tokens = valid_tokens.reshape(batch_size, -1)[:, 0]
            valid_tokens = valid_tokens.to(
                device=responses.device, dtype=torch.long
            )
        else:
            finish_step = data["finish_step"]
            if finish_step.dim() > 1:
                finish_step = finish_step.reshape(batch_size, -1)[:, 0]
            valid_tokens = finish_step.to(
                device=responses.device, dtype=torch.long
            ).clamp_min(0) * int(self.config.action_token_len)

        valid_tokens = valid_tokens.clamp(min=0, max=response_length)
        steps = torch.arange(response_length, device=responses.device).unsqueeze(0)
        response_mask = steps < valid_tokens.unsqueeze(1)
        return response_mask, valid_tokens

    @staticmethod
    def _is_cuda_oom_error(error: BaseException) -> bool:
        return (
            isinstance(error, torch.OutOfMemoryError)
            or "out of memory" in str(error).lower()
        )

    def _get_initial_traj_chunk_size(self, traj_len: int) -> int:
        chunk_size = int(self.config.get("traj_mini_batch_size", traj_len) or traj_len)
        return max(1, min(traj_len, chunk_size))

    def _reduce_traj_chunk_size(
        self, start: int, end: int, chunk_size: int, mode: str
    ) -> int:
        new_chunk_size = max(1, chunk_size // 2)
        torch.cuda.empty_cache()
        print(
            f"[dp_rob] CUDA OOM during {mode} traj chunk {start}:{end}; "
            f"reducing traj chunk size from {chunk_size} to {new_chunk_size} and retrying."
        )
        return new_chunk_size

    @staticmethod
    def _dist_activity_flags(local_has_activity: bool) -> Tuple[bool, bool, bool]:
        dist_ready = dist.is_available() and dist.is_initialized()
        if not dist_ready:
            local_flag = bool(local_has_activity)
            return local_flag, local_flag, local_flag

        device = torch.device("cuda", torch.cuda.current_device())
        local_flag_t = torch.tensor(int(local_has_activity), device=device)
        min_flag_t = local_flag_t.clone()
        max_flag_t = local_flag_t.clone()
        dist.all_reduce(min_flag_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_flag_t, op=dist.ReduceOp.MAX)
        return (
            bool(local_flag_t.item()),
            bool(max_flag_t.item()),
            bool(min_flag_t.item()),
        )

    @staticmethod
    def _dist_max_float(local_value: float) -> float:
        dist_ready = dist.is_available() and dist.is_initialized()
        if not dist_ready:
            return float(local_value)

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        value_t = torch.tensor(float(local_value), device=device, dtype=torch.float32)
        dist.all_reduce(value_t, op=dist.ReduceOp.MAX)
        return float(value_t.item())

    def _get_dummy_input_token_id(self) -> int:
        pad_token_id = int(getattr(self, "pad_token_id", 0) or 0)
        return 1 if pad_token_id != 1 else 2

    def _get_dummy_response_token_id(self) -> int:
        if self.config.vla == "openvla-oft":
            vocab_size = int(getattr(self.actor_module, "vocab_size", 32000))
            return vocab_size - 256
        return self._get_dummy_input_token_id()

    def _inject_dummy_active_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        responses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_ids.clone()
        attention_mask = attention_mask.clone()
        responses = responses.clone()

        dummy_input_token_id = self._get_dummy_input_token_id()
        dummy_response_token_id = self._get_dummy_response_token_id()

        input_ids[:, 0] = dummy_input_token_id
        attention_mask[:, 0] = 1
        responses.fill_(dummy_response_token_id)
        return input_ids, attention_mask, responses

    def _traj_token_slice_bounds(self, start: int, end: int) -> Tuple[int, int]:
        token_span = self.config.action_token_len * self.config.action_chunks_len
        return start * token_span, end * token_span

    def _slice_micro_batch_by_traj(self, micro_batch, traj_start: int, traj_end: int):
        traj_len = micro_batch["responses"].size(1)
        sliced_batch = {}
        for key, value in micro_batch.items():
            if not torch.is_tensor(value):
                sliced_batch[key] = value
                continue
            if value.ndim >= 2 and value.shape[1] == traj_len:
                sliced_batch[key] = value[:, traj_start:traj_end, ...]
            else:
                sliced_batch[key] = value
        return sliced_batch

    @staticmethod
    def _get_entropy_metric_key(batch_data: DataProto) -> str:
        is_filtered = bool(batch_data.meta_info["is_filtered"])
        train_mode = bool(batch_data.meta_info["train_mode"])
        if is_filtered and train_mode:
            return "actor_after/entropy_loss_train"
        if is_filtered and not train_mode:
            return "actor_after/entropy_loss_eval"
        if (not is_filtered) and train_mode:
            return "actor_before/entropy_loss_train"
        return "actor_before/entropy_loss_eval"

    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked:
            entropy_masked:
        """
        mask_expanded = mask.unsqueeze(-1)

        log_probs_masked = torch.where(
            mask_expanded, log_probs, torch.zeros_like(log_probs, requires_grad=False)
        )

        entropy_masked = torch.where(
            mask_expanded, entropy, torch.zeros_like(entropy, requires_grad=False)
        )

        return log_probs_masked, entropy_masked

    def _forward_micro_batch(
        self, micro_batch, temperature
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:

        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        batch_size = micro_batch["responses"].size(0)
        traj_len = micro_batch["responses"].size(1)
        tot_pad_len = micro_batch["input_ids"].size(2)

        assert all(
            micro_batch[key].size(0) == batch_size
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(1) == traj_len
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(2) == tot_pad_len
            for key in ["input_ids", "attention_mask"]
        )
        if self.use_proprio:
            assert (
                micro_batch["proprio"].size(0) == batch_size
                and micro_batch["proprio"].size(1) == traj_len
                and micro_batch["proprio"].size(2) == self.config.action_token_len
            )

        response_length = micro_batch["responses"].size(-1)  # 7*8

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]

            input_ids = input_ids.reshape(
                (batch_size * traj_len,) + input_ids.shape[2:]
            )
            attention_mask = attention_mask.reshape(
                (batch_size * traj_len,) + attention_mask.shape[2:]
            )
            pixel_values = pixel_values.reshape(
                (batch_size * traj_len,) + pixel_values.shape[2:]
            )
            responses = responses.reshape(
                (batch_size * traj_len,) + responses.shape[2:]
            )

            if self.use_proprio:
                proprio = micro_batch["proprio"]
                proprio = proprio.reshape((batch_size * traj_len,) + proprio.shape[2:])
            else:
                proprio = None

            # input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            # attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
            input_ids_unpad, _ = self.process_tensor_unpad_columns(
                input_ids, self.pad_token_id
            )
            attention_mask_unpad, _ = self.process_tensor_unpad_columns(
                attention_mask, 0
            )

            if self.config.vla == "openvla-oft":
                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    proprio=proprio,
                )  # prevent model thinks we are generating

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[
                    ..., -256 - 64 : -64
                ]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                # assert (0<=responses<=255).all()

                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(log_probs.shape) == 2 and len(entropy.shape) == 2
                log_probs = log_probs.reshape(
                    (
                        batch_size,
                        traj_len * self.config.action_chunks_len,
                        self.config.action_token_len,
                    )
                )  # *
                entropy = entropy.reshape(
                    (
                        batch_size,
                        traj_len * self.config.action_chunks_len,
                        self.config.action_token_len,
                    )
                )

                mask = self.generate_traj_mask(
                    micro_batch["finish_step"], traj_len * self.config.action_chunks_len
                )  # , self.config.action_token_len
                log_probs, entropy = self.apply_mask_with_grad_control(
                    log_probs, entropy, mask
                )

                log_probs = log_probs.reshape((batch_size, traj_len * response_length))
                entropy = entropy.reshape((batch_size, traj_len * response_length))

            elif self.config.vla == "openvla":
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits

                logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                # ADD

                log_probs = log_probs.reshape(
                    (
                        batch_size,
                        traj_len,
                    )
                    + log_probs.shape[1:]
                )
                entropy = entropy.reshape(
                    (
                        batch_size,
                        traj_len,
                    )
                    + entropy.shape[1:]
                )

                mask = self.generate_traj_mask(micro_batch["finish_step"], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(
                    log_probs, entropy, mask
                )

                log_probs = log_probs.reshape((batch_size, traj_len * response_length))
                entropy = entropy.reshape((batch_size, traj_len * response_length))

            return entropy, log_probs

    def _forward_micro_batch_update(
        self, input_ids, attention_mask, pixel_values, responses, temperature, proprio
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.vla == "openvla-oft":

                input_ids_unpad, _ = self.process_tensor_unpad_columns(
                    input_ids, self.pad_token_id
                )
                attention_mask_unpad, _ = self.process_tensor_unpad_columns(
                    attention_mask, 0
                )

                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    proprio=proprio,
                )

                assert logits.requires_grad

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[
                    ..., -256 - 64 : -64
                ]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index

                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                return entropy, log_probs

            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, _ = self.process_tensor_unpad_columns(
                    input_ids, self.pad_token_id
                )
                attention_mask_unpad, _ = self.process_tensor_unpad_columns(
                    attention_mask, 0
                )
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                #

                logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                return entropy, log_probs

    def _forward_micro_batch_entropy(
        self, micro_batch, temperature
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch["responses"].size(0)
        traj_len = micro_batch["responses"].size(1)
        tot_pad_len = micro_batch["input_ids"].size(2)

        assert all(
            micro_batch[key].size(0) == batch_size
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(1) == traj_len
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(2) == tot_pad_len
            for key in ["input_ids", "attention_mask"]
        )

        if self.use_proprio:
            assert (
                micro_batch["proprio"].size(0) == batch_size
                and micro_batch["proprio"].size(1) == traj_len
                and micro_batch["proprio"].size(2) == self.config.action_token_len
            )

        response_length = micro_batch["responses"].size(-1)
        # assert response_length == 7*8

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            # batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            pixel_values = micro_batch["pixel_values"]

            input_ids = input_ids.reshape(
                (batch_size * traj_len,) + input_ids.shape[2:]
            )
            attention_mask = attention_mask.reshape(
                (batch_size * traj_len,) + attention_mask.shape[2:]
            )
            pixel_values = pixel_values.reshape(
                (batch_size * traj_len,) + pixel_values.shape[2:]
            )

            if self.use_proprio:
                proprio = micro_batch["proprio"]
                proprio = proprio.reshape((batch_size * traj_len,) + proprio.shape[2:])
            else:
                proprio = None

            input_ids_unpad, _ = self.process_tensor_unpad_columns(
                input_ids, self.pad_token_id
            )
            attention_mask_unpad, _ = self.process_tensor_unpad_columns(
                attention_mask, 0
            )

            if self.config.vla == "openvla-oft":

                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    proprio=proprio,
                )

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[
                    ..., -256 - 64 : -64
                ]  # Shape: [batch_size, seq_len, 256]

                logits = logits.div(temperature)

                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape) == 2
                entropy = entropy.reshape(
                    (
                        batch_size,
                        traj_len * self.config.action_chunks_len,
                        self.config.action_token_len,
                    )
                )
                mask = self.generate_traj_mask(
                    micro_batch["finish_step"], traj_len * self.config.action_chunks_len
                )
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len * response_length))
                return entropy

            elif self.config.vla == "openvla":
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                #

                logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
                logits = logits.div(temperature)

                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                # ADD

                entropy = entropy.reshape(
                    (
                        batch_size,
                        traj_len,
                    )
                    + entropy.shape[1:]
                )
                mask = self.generate_traj_mask(micro_batch["finish_step"], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len * response_length))
                return entropy

    def _optimizer_step(self, skip_step: bool = False):
        grad_clip = getattr(self.config, "grad_clip", None)
        if grad_clip is None:
            grad_clip = 1.0
            print(
                "[dp_rob] WARNING: actor grad_clip is None; using fallback grad_clip=1.0.",
                flush=True,
            )

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=grad_clip
            )

        if skip_step:
            self.actor_optimizer.zero_grad()
            return grad_norm

        if not torch.isfinite(grad_norm).all():
            raise FloatingPointError("Non-finite actor gradient norm; optimizer update aborted")
        self._force_actor_optimizer_single_tensor_step()
        self._repair_actor_optimizer_tensor_contract(context="before_step")
        # AdamW can update some parameters before failing. Retrying the whole step
        # would update those parameters twice; propagate failures instead.
        self.actor_optimizer.step()
        return grad_norm

    def _force_actor_optimizer_single_tensor_step(self) -> None:
        if self.actor_optimizer is None:
            return
        for group in self.actor_optimizer.param_groups:
            group["foreach"] = False
            if "fused" in group:
                group["fused"] = False
        if hasattr(self.actor_optimizer, "defaults"):
            self.actor_optimizer.defaults["foreach"] = False
            if "fused" in self.actor_optimizer.defaults:
                self.actor_optimizer.defaults["fused"] = False

    def _repair_actor_optimizer_tensor_contract(self, context: str = "") -> None:
        """Keep AdamW state tensors compatible with optimizer offload.

        FSDP owns the gradient tensor layout.  Do not replace ``param.grad`` here:
        flat/sharded FSDP params can legally have grad tensors whose logical shape
        differs from ``param.shape`` during optimizer offload/load transitions.
        """
        if self.actor_optimizer is None:
            return

        repaired = 0
        skipped_shape = 0
        for group in self.actor_optimizer.param_groups:
            capturable = bool(group.get("capturable", False))
            fused = bool(group.get("fused", False))
            for param in group.get("params", []):
                if param is None or not isinstance(param, torch.Tensor):
                    continue
                target_device = param.device
                target_dtype = (
                    param.dtype if param.is_floating_point() else torch.float32
                )

                state = self.actor_optimizer.state.get(param, None)
                if not state:
                    continue
                for key, value in list(state.items()):
                    if not isinstance(value, torch.Tensor):
                        continue
                    if key == "step":
                        # AdamW allows non-capturable step tensors to stay on CPU.
                        # Keeping them CPU avoids mixed CPU/GPU optimizer-offload
                        # residue from tripping foreach grouping.
                        desired_device = (
                            target_device
                            if capturable or fused
                            else torch.device("cpu")
                        )
                        desired_dtype = (
                            value.dtype
                            if value.dtype in (torch.float32, torch.float64)
                            else torch.float32
                        )
                    elif value.is_floating_point():
                        desired_device = target_device
                        desired_dtype = target_dtype
                        if value.numel() not in (0, param.numel()):
                            skipped_shape += 1
                            continue
                    else:
                        desired_device = target_device
                        desired_dtype = value.dtype
                        if value.numel() not in (0, param.numel()):
                            skipped_shape += 1
                            continue

                    if value.device != desired_device or value.dtype != desired_dtype:
                        state[key] = value.to(
                            device=desired_device,
                            dtype=desired_dtype,
                            non_blocking=True,
                        )
                        repaired += 1

        if repaired > 0:
            if not hasattr(self, "_optimizer_contract_repair_count"):
                self._optimizer_contract_repair_count = 0
            self._optimizer_contract_repair_count += repaired
            print(
                "[dp_rob] repaired actor optimizer tensor contract "
                f"({context}): moved/cast {repaired} tensors; "
                f"total={self._optimizer_contract_repair_count}",
                flush=True,
            )
        if skipped_shape > 0:
            if not hasattr(self, "_optimizer_contract_shape_skip_count"):
                self._optimizer_contract_shape_skip_count = 0
            self._optimizer_contract_shape_skip_count += skipped_shape
            print(
                "[dp_rob] WARNING: skipped actor optimizer state repair for "
                f"{skipped_shape} tensors with non-param shape ({context}); "
                "leaving them untouched to preserve FSDP/offload layout. "
                f"total={self._optimizer_contract_shape_skip_count}",
                flush=True,
            )

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]  # 256
        print(f"[dp_rob] micro_batch_size: {micro_batch_size}")
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]  # trues
        self.pad_token_id = data.meta_info["pad_token_id"]

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "finish_step",
        ]
        if self.use_proprio:
            select_keys.append("proprio")
        select_keys = [key for key in select_keys if key in data.batch]
        batch = data.select(batch_keys=select_keys).batch

        # def iter_micro_batches(batch: dict, micro_batch_size: int):
        #     batch_size = next(iter(batch.values())).size(0)
        #     print(f"batch_size: {batch_size}")
        #     for start in range(0, batch_size, micro_batch_size):
        #         micro = {k: v[start:start+micro_batch_size] for k, v in batch.items()}
        #         yield micro

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = (
                data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            )
            micro_batches, indices = rearrange_micro_batches(
                batch=batch, max_token_len=max_token_len
            )
        else:
            micro_batches = batch.split(micro_batch_size)
            # micro_batches = iter_micro_batches(batch, micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature
                )
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(
                0
            ), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy_old(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = (
            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        )
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "old_log_probs",
            "advantages",
            "token_level_rewards",
            "finish_step",
            "is_weight",
            "is_wm",
            "is_weight_token",
            "is_wm_token_mask",
            "has_anchor_reward",
            "valid_response_tokens",
            "wm_valid_response_tokens",
        ]
        if self.use_proprio:
            select_keys.append("proprio")
        select_keys = [key for key in select_keys if key in data.batch]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, _ = rearrange_micro_batches(
                    batch=mini_batch, max_token_len=max_token_len
                )
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()
            mini_has_optimizer_signal = False
            mini_skip_optimizer_for_kl = False

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                response_mask, _ = self._build_response_mask(data)

                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]
                response_length = response_mask.shape[-1]

                raw_weight_token_full = data.get("is_weight_token", None)
                if raw_weight_token_full is not None:
                    full_weight = self._slice_contract_field(
                        raw_weight_token_full,
                        response_mask,
                        0,
                        response_length,
                        dtype=torch.float32,
                        name="is_weight_token",
                        allow_sample_broadcast=False,
                    )
                else:
                    raw_weight_full = data.get("is_weight", None)
                    if raw_weight_full is None:
                        full_weight = torch.ones_like(response_mask, dtype=torch.float32)
                    else:
                        full_weight = self._slice_contract_field(
                            raw_weight_full,
                            response_mask,
                            0,
                            response_length,
                            dtype=torch.float32,
                            name="is_weight",
                            allow_sample_broadcast=True,
                        )

                raw_is_wm_token_full = data.get("is_wm_token_mask", None)
                if raw_is_wm_token_full is not None:
                    full_is_wm_mask = (
                        self._slice_contract_field(
                            raw_is_wm_token_full,
                            response_mask,
                            0,
                            response_length,
                            dtype=torch.float32,
                            name="is_wm_token_mask",
                            allow_sample_broadcast=False,
                        )
                        > 0.5
                    )
                else:
                    raw_is_wm_full = data.get("is_wm", None)
                    if raw_is_wm_full is None:
                        full_is_wm_mask = torch.zeros_like(
                            response_mask, dtype=torch.bool
                        )
                    else:
                        full_is_wm_mask = (
                            self._slice_contract_field(
                                raw_is_wm_full,
                                response_mask,
                                0,
                                response_length,
                                dtype=torch.float32,
                                name="is_wm",
                                allow_sample_broadcast=True,
                            )
                            > 0.5
                        )

                raw_has_anchor_full = data.get("has_anchor_reward", None)
                if raw_has_anchor_full is not None:
                    full_has_anchor_mask = (
                        self._slice_contract_field(
                            raw_has_anchor_full,
                            response_mask,
                            0,
                            response_length,
                            dtype=torch.float32,
                            name="has_anchor_reward",
                            allow_sample_broadcast=True,
                        )
                        > 0.5
                    )
                    full_weight = torch.where(
                        full_is_wm_mask & (~full_has_anchor_mask),
                        torch.zeros_like(full_weight),
                        full_weight,
                    )

                effective_response_mask = response_mask & (
                    full_weight.detach().abs() > 0
                )
                loss_denominator = effective_response_mask.sum(axis=None).clamp_min(1)

                # clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data["responses"].size(0)
                traj_len = data["responses"].size(1)

                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    "actor/pg_loss": 0,
                    "actor/pg_loss_real": 0,
                    "actor/pg_loss_imag": 0,
                    "actor/pg_clipfrac": 0,
                    "actor/ppo_kl": 0,
                    "actor/imag_weight_mean": 0,
                }

                print(
                    f"[dp_rob] traj len: {traj_len}, traj_mini_batch_size: {self.config.traj_mini_batch_size}"
                )
                assert traj_len % self.config.traj_mini_batch_size == 0
                traj_split_num = int(traj_len / self.config.traj_mini_batch_size)

                for i in range(0, traj_len, int(traj_len / traj_split_num)):
                    slice_id = (
                        i * self.config.action_token_len * self.config.action_chunks_len
                    )
                    next_slice_id = (
                        (i + int(traj_len / traj_split_num))
                        * self.config.action_token_len
                        * self.config.action_chunks_len
                    )
                    response_mask_tmp = response_mask[:, slice_id:next_slice_id]
                    local_chunk_has_tokens, chunk_has_any_tokens, _ = (
                        self._dist_activity_flags(
                            bool(response_mask_tmp.sum().item() > 0)
                        )
                    )
                    if not chunk_has_any_tokens:
                        continue

                    traj_batch = self._slice_micro_batch_by_traj(
                        data, i, i + int(traj_len / traj_split_num)
                    )
                    traj_input_ids = traj_batch["input_ids"].reshape(
                        (-1,) + traj_batch["input_ids"].shape[2:]
                    )
                    traj_attention_mask = traj_batch["attention_mask"].reshape(
                        (-1,) + traj_batch["attention_mask"].shape[2:]
                    )
                    if local_chunk_has_tokens and (not traj_attention_mask.ne(0).any()):
                        raise ValueError(
                            "attention_mask is fully padded in update_policy_old despite non-empty response_mask: "
                            f"chunk={i}:{i + int(traj_len / traj_split_num)}, finish_step={data['finish_step'].detach().cpu().tolist()}"
                        )
                    traj_pixel_values = traj_batch["pixel_values"].reshape(
                        (-1,) + traj_batch["pixel_values"].shape[2:]
                    )
                    traj_responses = traj_batch["responses"].reshape(
                        (-1,) + traj_batch["responses"].shape[2:]
                    )
                    if self.use_proprio:
                        traj_proprio = traj_batch["proprio"].reshape(
                            (-1,) + traj_batch["proprio"].shape[2:]
                        )
                    else:
                        traj_proprio = None

                    if not local_chunk_has_tokens:
                        (
                            traj_input_ids,
                            traj_attention_mask,
                            traj_responses,
                        ) = self._inject_dummy_active_tokens(
                            traj_input_ids,
                            traj_attention_mask,
                            traj_responses,
                        )

                    entropy, log_prob = self._forward_micro_batch_update(
                        input_ids=traj_input_ids,
                        attention_mask=traj_attention_mask,
                        pixel_values=traj_pixel_values,
                        responses=traj_responses,
                        temperature=temperature,
                        proprio=traj_proprio,
                    )
                    entropy = entropy.reshape(batch_size, -1)
                    log_prob = log_prob.reshape(batch_size, -1)

                    if not local_chunk_has_tokens:
                        dummy_loss = (
                            entropy.float().sum() * 0.0 + log_prob.float().sum() * 0.0
                        ) / self.gradient_accumulation
                        dummy_loss.backward()
                        continue

                    old_log_prob_tmp = old_log_prob[:, slice_id:next_slice_id]
                    advantages_tmp = advantages[:, slice_id:next_slice_id]

                    if log_prob.shape != old_log_prob_tmp.shape:
                        raise ValueError(
                            "log_prob shape mismatch in update_policy_old: "
                            f"got {tuple(log_prob.shape)}, expected {tuple(old_log_prob_tmp.shape)}"
                        )
                    if log_prob.shape != advantages_tmp.shape:
                        raise ValueError(
                            "advantages shape mismatch in update_policy_old: "
                            f"got {tuple(log_prob.shape)} vs {tuple(advantages_tmp.shape)}"
                        )

                    raw_weight_token = data.get("is_weight_token", None)
                    if raw_weight_token is not None:
                        weight = self._slice_contract_field(
                            raw_weight_token,
                            advantages_tmp,
                            slice_id,
                            next_slice_id,
                            dtype=advantages_tmp.dtype,
                            name="is_weight_token",
                            allow_sample_broadcast=False,
                        )
                    else:
                        raw_weight = data.get("is_weight", None)
                        if raw_weight is None:
                            weight = torch.ones_like(advantages_tmp)
                        else:
                            weight = self._slice_contract_field(
                                raw_weight,
                                advantages_tmp,
                                slice_id,
                                next_slice_id,
                                dtype=advantages_tmp.dtype,
                                name="is_weight",
                                allow_sample_broadcast=True,
                            )

                    raw_is_wm_token = data.get("is_wm_token_mask", None)
                    if raw_is_wm_token is not None:
                        is_wm_mask = (
                            self._slice_contract_field(
                                raw_is_wm_token,
                                advantages_tmp,
                                slice_id,
                                next_slice_id,
                                dtype=torch.float32,
                                name="is_wm_token_mask",
                                allow_sample_broadcast=False,
                            )
                            > 0.5
                        )
                    else:
                        raw_is_wm = data.get("is_wm", None)
                        if raw_is_wm is None:
                            is_wm_mask = torch.zeros_like(
                                advantages_tmp, dtype=torch.bool
                            )
                        else:
                            is_wm_mask = (
                                self._slice_contract_field(
                                    raw_is_wm,
                                    advantages_tmp,
                                    slice_id,
                                    next_slice_id,
                                    dtype=torch.float32,
                                    name="is_wm",
                                    allow_sample_broadcast=True,
                                )
                                > 0.5
                            )

                    raw_has_anchor = data.get("has_anchor_reward", None)
                    if raw_has_anchor is not None:
                        has_anchor_mask = (
                            self._slice_contract_field(
                                raw_has_anchor,
                                advantages_tmp,
                                slice_id,
                                next_slice_id,
                                dtype=torch.float32,
                                name="has_anchor_reward",
                                allow_sample_broadcast=True,
                            )
                            > 0.5
                        )
                        weight = torch.where(
                            is_wm_mask & (~has_anchor_mask),
                            torch.zeros_like(weight),
                            weight,
                        )

                    real_mask = response_mask_tmp & (~is_wm_mask)
                    imag_mask = response_mask_tmp & is_wm_mask
                    weighted_advantages = advantages_tmp * weight

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob_tmp,
                        log_prob=log_prob,
                        advantages=weighted_advantages,
                        eos_mask=response_mask_tmp,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_low=clip_ratio_low,
                    )

                    if real_mask.any():
                        pg_loss_real, _, _ = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob_tmp,
                            log_prob=log_prob,
                            advantages=weighted_advantages,
                            eos_mask=real_mask,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_low=clip_ratio_low,
                        )
                    else:
                        pg_loss_real = torch.zeros_like(pg_loss)

                    if imag_mask.any():
                        pg_loss_imag, _, _ = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob_tmp,
                            log_prob=log_prob,
                            advantages=weighted_advantages,
                            eos_mask=imag_mask,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_low=clip_ratio_low,
                        )
                    else:
                        pg_loss_imag = torch.zeros_like(pg_loss)

                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss * response_mask_tmp_sum
                    pg_clipfrac = (
                        pg_clipfrac * response_mask_tmp_sum / loss_denominator
                    )
                    ppo_kl = ppo_kl * response_mask_tmp_sum / loss_denominator

                    policy_loss = pg_loss / loss_denominator

                    loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    loss_info["actor/pg_loss"] = (
                        loss_info["actor/pg_loss"] + policy_loss.detach().item()
                    )
                    loss_info["actor/pg_loss_real"] = (
                        loss_info["actor/pg_loss_real"] + pg_loss_real.detach().item()
                    )
                    loss_info["actor/pg_loss_imag"] = (
                        loss_info["actor/pg_loss_imag"] + pg_loss_imag.detach().item()
                    )
                    loss_info["actor/pg_clipfrac"] = (
                        loss_info["actor/pg_clipfrac"] + pg_clipfrac.detach().item()
                    )
                    loss_info["actor/ppo_kl"] = (
                        loss_info["actor/ppo_kl"] + ppo_kl.detach().item()
                    )
                    if imag_mask.any():
                        loss_info["actor/imag_weight_mean"] = (
                            loss_info["actor/imag_weight_mean"]
                            + weight[imag_mask].float().mean().detach().item()
                        )

                append_to_dict(metrics, loss_info)

            grad_norm = self._optimizer_step()
            data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    def update_policy(self, data: DataProto):
        if data.meta_info.get("chunk_objective", False):
            return self.update_chunk_policy(data)
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = (
            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        )
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error

        required_select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "old_log_probs",
            "advantages",
            "token_level_rewards",
            "finish_step",
        ]
        optional_select_keys = [
            "is_weight",
            "is_wm",
            "is_weight_token",
            "is_wm_token_mask",
            "has_anchor_reward",
            "valid_response_tokens",
            "wm_valid_response_tokens",
        ]
        if self.use_proprio:
            required_select_keys.append("proprio")
        select_keys = required_select_keys + [
            key for key in optional_select_keys if key in data.batch
        ]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        #! ---------- BEGIN FSDP-SAFE EMPTY-BATCH HANDLING ----------
        # helper: check dist ready (should be true in FSDP worker)
        dist_ready = dist.is_available() and dist.is_initialized()
        print("[dp_rob] update_policy batch_length:", len(batch))
        print(
            "[dp_rob] self.config.ppo_mini_batch_size:", self.config.ppo_mini_batch_size
        )
        base_ppo_mini_batch_size = max(1, int(self.config.ppo_mini_batch_size))
        ppo_kl_hard_limit = max(
            0.0, float(getattr(self.config, "ppo_kl_hard_limit", 0.0) or 0.0)
        )

        if not dist_ready:
            # fallback: single-proc, treat as normal
            ppo_mini_batch_size = min(max(len(batch), 1), base_ppo_mini_batch_size)
            num_minibatch = len(batch) // ppo_mini_batch_size
            dataloader = batch.split(ppo_mini_batch_size)[:num_minibatch]
        else:
            # per-rank indicator: has any sample?
            local_has = torch.tensor(int(len(batch) > 0), device="cuda")
            min_has = local_has.clone()
            max_has = local_has.clone()
            dist.all_reduce(min_has, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_has, op=dist.ReduceOp.MAX)

            # Case 0: everyone empty -> safe global skip, but return consistent metrics
            if max_has.item() == 0:
                # return metrics with same keys so outer aggregation won't drop fields
                return {
                    "actor/pg_loss": 0.0,
                    "actor/pg_loss_real": 0.0,
                    "actor/pg_loss_imag": 0.0,
                    "actor/pg_clipfrac": 0.0,
                    "actor/ppo_kl": 0.0,
                    "actor/ppo_kl_hard_limit": ppo_kl_hard_limit,
                    "actor/ppo_kl_hard_skip_count": 0.0,
                    "actor/ppo_kl_hard_total_count": 0.0,
                    "actor/ppo_kl_hard_skip_ratio": 0.0,
                    "actor/ppo_kl_hard_max": 0.0,
                    "actor/grad_norm": 0.0,
                    "actor/imag_weight_mean": 0.0,
                    "actor/imag_token_count": 0.0,
                    "actor/real_token_count": 0.0,
                    "actor/imag_adv_abs_mean": 0.0,
                    "actor/imag_reward_mean": 0.0,
                    "actor/skipped_all": 1,
                }

            # Case 1: some ranks empty, some non-empty -> construct dummy sample on empty ranks
            if min_has.item() == 0 and max_has.item() == 1:
                if len(batch) == 0:
                    # gather shape templates from ranks that have a sample
                    # prepare local shape info (None if empty)
                    local_shape_info = None
                    # batch might not be a plain dict; try to extract map of key->shape if possible
                    try:
                        # attempt to build shape info from batch (works for tensor-mapped batch)
                        local_shape_info = {
                            k: tuple(v.shape)
                            for k, v in ({} if batch is None else batch.items())
                        }
                    except Exception:
                        # fallback: try to use data.select to get a single element sample if possible
                        local_shape_info = None

                    # collect shape infos
                    gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(gathered, local_shape_info)

                    # pick first non-None template
                    template = None
                    for s in gathered:
                        if s is not None:
                            template = s
                            break
                    assert (
                        template is not None
                    ), "No template shapes gathered for dummy batch (but some ranks have data)."

                    # build a single-sample dict matching template shapes (tensors on cuda)
                    sample = {}
                    for k, shape in template.items():
                        # Heuristic for dtype choice — adapt if you have explicit schema
                        if "mask" in k or "id" in k or "finish" in k:
                            dtype = torch.long
                        else:
                            dtype = torch.float
                        # create zeros tensor on cuda
                        sample[k] = torch.zeros(shape, device="cuda", dtype=dtype)

                    # create a DataProto single-sample (use the same factory you used in fit_wm_v1)
                    # Note: DataProto.from_single_dict was used elsewhere; adapt if your API differs.
                    dummy_dp = DataProto.from_single_dict(sample)
                    # set batch to the dummy's batch representation (so subsequent .split works)
                    batch = dummy_dp.batch

            # Now align number of mini-batches across ranks
            local_mini_batch_size = min(max(len(batch), 1), base_ppo_mini_batch_size)
            mini_batch_size_t = torch.tensor(local_mini_batch_size, device="cuda")
            dist.all_reduce(mini_batch_size_t, op=dist.ReduceOp.MIN)
            ppo_mini_batch_size = max(1, int(mini_batch_size_t.item()))
            num_minibatch = len(batch) // ppo_mini_batch_size
            num_minibatch_t = torch.tensor(num_minibatch, device="cuda")
            dist.all_reduce(num_minibatch_t, op=dist.ReduceOp.MIN)
            num_minibatch = num_minibatch_t.item()
            dataloader = batch.split(ppo_mini_batch_size)[:num_minibatch]
        print("[dp_rob] update_policy dataloader_length:", len(dataloader))
        #! ---------- END FSDP-SAFE EMPTY-BATCH HANDLING ----------

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # dataloader = batch.split(self.config.ppo_mini_batch_size)  #! commented now
        metrics = {}
        global_imag_weight_sum = 0.0
        global_imag_weight_count = 0
        global_imag_token_count = 0
        global_real_token_count = 0
        global_imag_adv_abs_sum = 0.0
        global_imag_reward_sum = 0.0
        global_imag_reward_count = 0
        global_pg_loss_imag_sum = 0.0
        global_pg_loss_imag_count = 0
        global_pg_loss_real_sum = 0.0
        global_pg_loss_real_count = 0
        global_ppo_kl_hard_skip_count = 0
        global_ppo_kl_hard_total_count = 0
        global_ppo_kl_hard_max = 0.0
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, _ = rearrange_micro_batches(
                    batch=mini_batch, max_token_len=max_token_len
                )
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()
            mini_has_optimizer_signal = False
            mini_skip_optimizer_for_kl = False

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data["responses"]
                response_mask, valid_tokens = self._build_response_mask(data)

                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]

                response_length = response_mask.shape[-1]
                raw_weight_token_full = data.get("is_weight_token", None)
                if raw_weight_token_full is not None:
                    full_weight = self._slice_contract_field(
                        raw_weight_token_full,
                        response_mask,
                        0,
                        response_length,
                        dtype=torch.float32,
                        name="is_weight_token",
                        allow_sample_broadcast=False,
                    )
                else:
                    raw_weight_full = data.get("is_weight", None)
                    if raw_weight_full is None:
                        full_weight = torch.ones_like(response_mask, dtype=torch.float32)
                    else:
                        full_weight = self._slice_contract_field(
                            raw_weight_full,
                            response_mask,
                            0,
                            response_length,
                            dtype=torch.float32,
                            name="is_weight",
                            allow_sample_broadcast=True,
                        )

                raw_is_wm_token_full = data.get("is_wm_token_mask", None)
                if raw_is_wm_token_full is not None:
                    full_is_wm_mask = (
                        self._slice_contract_field(
                            raw_is_wm_token_full,
                            response_mask,
                            0,
                            response_length,
                            dtype=torch.float32,
                            name="is_wm_token_mask",
                            allow_sample_broadcast=False,
                        )
                        > 0.5
                    )
                else:
                    raw_is_wm_full = data.get("is_wm", None)
                    if raw_is_wm_full is None:
                        full_is_wm_mask = torch.zeros_like(
                            response_mask, dtype=torch.bool
                        )
                    else:
                        full_is_wm_mask = (
                            self._slice_contract_field(
                                raw_is_wm_full,
                                response_mask,
                                0,
                                response_length,
                                dtype=torch.float32,
                                name="is_wm",
                                allow_sample_broadcast=True,
                            )
                            > 0.5
                        )

                raw_has_anchor_full = data.get("has_anchor_reward", None)
                if raw_has_anchor_full is not None:
                    full_has_anchor_mask = (
                        self._slice_contract_field(
                            raw_has_anchor_full,
                            response_mask,
                            0,
                            response_length,
                            dtype=torch.float32,
                            name="has_anchor_reward",
                            allow_sample_broadcast=True,
                        )
                        > 0.5
                    )
                    full_weight = torch.where(
                        full_is_wm_mask & (~full_has_anchor_mask),
                        torch.zeros_like(full_weight),
                        full_weight,
                    )

                effective_response_mask = response_mask & (
                    full_weight.detach().abs() > 0
                )
                loss_denominator = effective_response_mask.sum(axis=None).clamp_min(1)

                # clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data["responses"].size(0)
                traj_len = data["responses"].size(1)

                print(
                    f"[dp_rob] responses_shape={tuple(data['responses'].shape)}, "
                    f"finish_step={data['finish_step'].detach().cpu().tolist()}, "
                    f"valid_response_tokens={valid_tokens.detach().cpu().tolist()}"
                )

                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    "actor/pg_loss": 0,
                    "actor/pg_loss_real": 0,
                    "actor/pg_loss_imag": 0,
                    "actor/pg_clipfrac": 0,
                    "actor/ppo_kl": 0,
                    "actor/imag_weight_mean": 0,
                }

                # ==========================================================
                #! FSDP PATCH：空轨迹也必须 backward
                # ==========================================================
                _, batch_has_any_tokens, batch_has_all_tokens = (
                    self._dist_activity_flags(bool(response_mask_sum.item() > 0))
                )
                if not batch_has_any_tokens:
                    dummy_loss = torch.zeros(
                        (), device=data["responses"].device, requires_grad=True
                    )
                    (dummy_loss / self.gradient_accumulation).backward()
                    append_to_dict(metrics, loss_info)
                    continue

                traj_chunk_size = self._get_initial_traj_chunk_size(traj_len)
                print(
                    f"[dp_rob] traj len: {traj_len}, initial_traj_chunk_size: {traj_chunk_size}"
                )

                traj_start = 0
                while traj_start < traj_len:
                    traj_end = min(traj_start + traj_chunk_size, traj_len)
                    slice_id, next_slice_id = self._traj_token_slice_bounds(
                        traj_start, traj_end
                    )
                    response_mask_tmp = response_mask[:, slice_id:next_slice_id]
                    local_chunk_has_tokens, chunk_has_any_tokens, _ = (
                        self._dist_activity_flags(
                            bool(response_mask_tmp.sum().item() > 0)
                        )
                    )
                    if not chunk_has_any_tokens:
                        traj_start = traj_end
                        continue

                    traj_batch = self._slice_micro_batch_by_traj(
                        data, traj_start, traj_end
                    )
                    traj_input_ids = traj_batch["input_ids"].reshape(
                        (-1,) + traj_batch["input_ids"].shape[2:]
                    )
                    traj_attention_mask = traj_batch["attention_mask"].reshape(
                        (-1,) + traj_batch["attention_mask"].shape[2:]
                    )
                    if local_chunk_has_tokens and (not traj_attention_mask.ne(0).any()):
                        raise ValueError(
                            "attention_mask is fully padded in update_policy despite non-empty response_mask: "
                            f"chunk={traj_start}:{traj_end}, finish_step={data['finish_step'].detach().cpu().tolist()}"
                        )
                    traj_pixel_values = traj_batch["pixel_values"].reshape(
                        (-1,) + traj_batch["pixel_values"].shape[2:]
                    )
                    traj_responses = traj_batch["responses"].reshape(
                        (-1,) + traj_batch["responses"].shape[2:]
                    )
                    if self.use_proprio:
                        traj_proprio = traj_batch["proprio"].reshape(
                            (-1,) + traj_batch["proprio"].shape[2:]
                        )
                    else:
                        traj_proprio = None
                    if not local_chunk_has_tokens:
                        (
                            traj_input_ids,
                            traj_attention_mask,
                            traj_responses,
                        ) = self._inject_dummy_active_tokens(
                            traj_input_ids,
                            traj_attention_mask,
                            traj_responses,
                        )
                    entropy, log_prob = self._forward_micro_batch_update(
                        input_ids=traj_input_ids,
                        attention_mask=traj_attention_mask,
                        pixel_values=traj_pixel_values,
                        responses=traj_responses,
                        temperature=temperature,
                        proprio=traj_proprio,
                    )
                    entropy = entropy.reshape(batch_size, -1)
                    log_prob = log_prob.reshape(batch_size, -1)

                    old_log_prob_tmp = old_log_prob[:, slice_id:next_slice_id]
                    advantages_tmp = advantages[:, slice_id:next_slice_id]

                    if log_prob.shape != old_log_prob_tmp.shape:
                        raise ValueError(
                            "log_prob shape mismatch in update_policy: "
                            f"got {tuple(log_prob.shape)}, expected {tuple(old_log_prob_tmp.shape)}"
                        )
                    if log_prob.shape != advantages_tmp.shape:
                        raise ValueError(
                            "advantages shape mismatch in update_policy: "
                            f"got {tuple(log_prob.shape)} vs {tuple(advantages_tmp.shape)}"
                        )

                    raw_weight_token = data.get("is_weight_token", None)
                    if raw_weight_token is not None:
                        weight = self._slice_contract_field(
                            raw_weight_token,
                            advantages_tmp,
                            slice_id,
                            next_slice_id,
                            dtype=advantages_tmp.dtype,
                            name="is_weight_token",
                            allow_sample_broadcast=False,
                        )
                    else:
                        raw_weight = data.get("is_weight", None)
                        if raw_weight is None:
                            weight = torch.ones_like(advantages_tmp)
                        else:
                            weight = self._slice_contract_field(
                                raw_weight,
                                advantages_tmp,
                                slice_id,
                                next_slice_id,
                                dtype=advantages_tmp.dtype,
                                name="is_weight",
                                allow_sample_broadcast=True,
                            )

                    raw_is_wm_token = data.get("is_wm_token_mask", None)
                    if raw_is_wm_token is not None:
                        is_wm_mask = (
                            self._slice_contract_field(
                                raw_is_wm_token,
                                advantages_tmp,
                                slice_id,
                                next_slice_id,
                                dtype=torch.float32,
                                name="is_wm_token_mask",
                                allow_sample_broadcast=False,
                            )
                            > 0.5
                        )
                    else:
                        raw_is_wm = data.get("is_wm", None)
                        if raw_is_wm is None:
                            is_wm_mask = torch.zeros_like(
                                advantages_tmp, dtype=torch.bool
                            )
                        else:
                            is_wm_mask = (
                                self._slice_contract_field(
                                    raw_is_wm,
                                    advantages_tmp,
                                    slice_id,
                                    next_slice_id,
                                    dtype=torch.float32,
                                    name="is_wm",
                                    allow_sample_broadcast=True,
                                )
                                > 0.5
                            )

                    raw_has_anchor = data.get("has_anchor_reward", None)
                    if raw_has_anchor is not None:
                        has_anchor_mask = (
                            self._slice_contract_field(
                                raw_has_anchor,
                                advantages_tmp,
                                slice_id,
                                next_slice_id,
                                dtype=torch.float32,
                                name="has_anchor_reward",
                                allow_sample_broadcast=True,
                            )
                            > 0.5
                        )
                        weight = torch.where(
                            is_wm_mask & (~has_anchor_mask),
                            torch.zeros_like(weight),
                            weight,
                        )

                    real_mask = response_mask_tmp & (~is_wm_mask)
                    imag_mask = response_mask_tmp & is_wm_mask
                    effective_loss_mask = response_mask_tmp & (
                        weight.detach().abs() > 0
                    )
                    real_loss_mask = effective_loss_mask & (~is_wm_mask)
                    imag_loss_mask = effective_loss_mask & is_wm_mask
                    weighted_advantages = advantages_tmp * weight
                    local_chunk_has_loss_signal = bool(
                        (
                            effective_loss_mask
                            & (weighted_advantages.detach().abs() > 0)
                        )
                        .sum()
                        .item()
                        > 0
                    )
                    _, chunk_has_any_loss_signal, _ = self._dist_activity_flags(
                        local_chunk_has_loss_signal
                    )
                    imag_token_count = int(imag_mask.sum().detach().item())
                    real_token_count = int(real_mask.sum().detach().item())
                    global_imag_token_count += imag_token_count
                    global_real_token_count += real_token_count

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob_tmp,
                        log_prob=log_prob,
                        advantages=weighted_advantages,
                        eos_mask=effective_loss_mask,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_low=clip_ratio_low,
                    )

                    if real_loss_mask.any():
                        pg_loss_real, _, _ = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob_tmp,
                            log_prob=log_prob,
                            advantages=weighted_advantages,
                            eos_mask=real_loss_mask,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_low=clip_ratio_low,
                        )
                    else:
                        pg_loss_real = torch.zeros_like(pg_loss)

                    if imag_loss_mask.any():
                        pg_loss_imag, _, _ = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob_tmp,
                            log_prob=log_prob,
                            advantages=weighted_advantages,
                            eos_mask=imag_loss_mask,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_low=clip_ratio_low,
                        )
                    else:
                        pg_loss_imag = torch.zeros_like(pg_loss)

                    response_mask_tmp_sum = effective_loss_mask.sum(axis=None)
                    skip_chunk_for_kl = False
                    if ppo_kl_hard_limit > 0.0:
                        global_ppo_kl_hard_total_count += 1
                        chunk_ppo_kl = max(0.0, float(ppo_kl.detach().item()))
                        dist_chunk_ppo_kl = self._dist_max_float(chunk_ppo_kl)
                        global_ppo_kl_hard_max = max(
                            global_ppo_kl_hard_max, dist_chunk_ppo_kl
                        )
                        if dist_chunk_ppo_kl > ppo_kl_hard_limit:
                            global_ppo_kl_hard_skip_count += 1
                            mini_skip_optimizer_for_kl = True
                            loss_info["actor/ppo_kl"] = (
                                loss_info["actor/ppo_kl"]
                                + dist_chunk_ppo_kl
                                * response_mask_tmp_sum.detach().item()
                                / max(loss_denominator.detach().item(), 1.0)
                            )
                            skip_chunk_for_kl = True

                    pg_loss = pg_loss * response_mask_tmp_sum
                    pg_clipfrac = (
                        pg_clipfrac * response_mask_tmp_sum / loss_denominator
                    )
                    ppo_kl = ppo_kl * response_mask_tmp_sum / loss_denominator

                    policy_loss = pg_loss / loss_denominator
                    if skip_chunk_for_kl:
                        policy_loss = (
                            entropy.float().sum() * 0.0
                            + log_prob.float().sum() * 0.0
                        )
                        pg_clipfrac = torch.zeros_like(policy_loss)
                        ppo_kl = torch.zeros_like(policy_loss)
                        pg_loss_real = torch.zeros_like(policy_loss)
                        pg_loss_imag = torch.zeros_like(policy_loss)
                    else:
                        if chunk_has_any_loss_signal:
                            mini_has_optimizer_signal = True
                        if real_loss_mask.any():
                            global_pg_loss_real_sum += float(
                                pg_loss_real.detach().item()
                            )
                            global_pg_loss_real_count += 1
                        if imag_loss_mask.any():
                            global_pg_loss_imag_sum += float(
                                pg_loss_imag.detach().item()
                            )
                            global_pg_loss_imag_count += 1

                    loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    loss_info["actor/pg_loss"] = (
                        loss_info["actor/pg_loss"] + policy_loss.detach().item()
                    )
                    loss_info["actor/pg_loss_real"] = (
                        loss_info["actor/pg_loss_real"] + pg_loss_real.detach().item()
                    )
                    loss_info["actor/pg_loss_imag"] = (
                        loss_info["actor/pg_loss_imag"] + pg_loss_imag.detach().item()
                    )
                    loss_info["actor/pg_clipfrac"] = (
                        loss_info["actor/pg_clipfrac"] + pg_clipfrac.detach().item()
                    )
                    loss_info["actor/ppo_kl"] = (
                        loss_info["actor/ppo_kl"] + ppo_kl.detach().item()
                    )
                    if imag_mask.any():
                        imag_weight_values = weight[imag_mask].float()
                        global_imag_weight_sum += float(
                            imag_weight_values.detach().sum().item()
                        )
                        global_imag_weight_count += int(imag_weight_values.numel())
                        global_imag_adv_abs_sum += float(
                            advantages_tmp[imag_mask]
                            .abs()
                            .float()
                            .detach()
                            .sum()
                            .item()
                        )
                        token_level_rewards = data.get("token_level_rewards", None)
                        if token_level_rewards is not None:
                            reward_tmp = token_level_rewards[:, slice_id:next_slice_id]
                            if reward_tmp.shape == imag_mask.shape:
                                global_imag_reward_sum += float(
                                    reward_tmp[imag_mask].float().detach().sum().item()
                                )
                                global_imag_reward_count += int(imag_mask.sum().item())

                    traj_start = traj_end

                append_to_dict(metrics, loss_info)

            skip_optimizer_step = (
                (not mini_has_optimizer_signal) or mini_skip_optimizer_for_kl
            )
            grad_norm = self._optimizer_step(skip_step=skip_optimizer_step)
            data = {"actor/grad_norm": grad_norm.detach().item()}
            data["actor/optimizer_step_count"] = (
                0.0 if skip_optimizer_step else 1.0
            )
            data["actor/optimizer_step_skipped"] = (
                1.0 if skip_optimizer_step else 0.0
            )
            data["actor/optimizer_step_skipped_by_kl"] = (
                1.0 if mini_skip_optimizer_for_kl else 0.0
            )
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()

        aggregate = torch.tensor(
            [
                float(global_imag_weight_sum),
                float(global_imag_weight_count),
                float(global_imag_token_count),
                float(global_real_token_count),
                float(global_imag_adv_abs_sum),
                float(global_imag_reward_sum),
                float(global_imag_reward_count),
                float(global_pg_loss_imag_sum),
                float(global_pg_loss_imag_count),
                float(global_pg_loss_real_sum),
                float(global_pg_loss_real_count),
            ],
            dtype=torch.float64,
            device=(
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
        )
        if dist_ready:
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        (
            global_imag_weight_sum,
            global_imag_weight_count,
            global_imag_token_count,
            global_real_token_count,
            global_imag_adv_abs_sum,
            global_imag_reward_sum,
            global_imag_reward_count,
            global_pg_loss_imag_sum,
            global_pg_loss_imag_count,
            global_pg_loss_real_sum,
            global_pg_loss_real_count,
        ) = (
            aggregate.detach().cpu().tolist()
        )

        kl_gate_aggregate = torch.tensor(
            [
                float(global_ppo_kl_hard_skip_count),
                float(global_ppo_kl_hard_total_count),
                float(global_ppo_kl_hard_max),
            ],
            dtype=torch.float64,
            device=(
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
        )
        if dist_ready:
            dist.all_reduce(kl_gate_aggregate, op=dist.ReduceOp.MAX)
        (
            global_ppo_kl_hard_skip_count,
            global_ppo_kl_hard_total_count,
            global_ppo_kl_hard_max,
        ) = kl_gate_aggregate.detach().cpu().tolist()

        if global_pg_loss_imag_count > 0:
            metrics["actor/pg_loss_imag"] = [
                float(global_pg_loss_imag_sum / global_pg_loss_imag_count)
            ]
        if global_pg_loss_real_count > 0:
            metrics["actor/pg_loss_real"] = [
                float(global_pg_loss_real_sum / global_pg_loss_real_count)
            ]
        metrics["actor/imag_weight_mean"] = [
            (
                global_imag_weight_sum / global_imag_weight_count
                if global_imag_weight_count > 0
                else 0.0
            )
        ]
        metrics["actor/imag_token_count"] = [float(global_imag_token_count)]
        metrics["actor/real_token_count"] = [float(global_real_token_count)]
        metrics["actor/imag_adv_abs_mean"] = [
            (
                global_imag_adv_abs_sum / global_imag_token_count
                if global_imag_token_count > 0
                else 0.0
            )
        ]
        metrics["actor/imag_reward_mean"] = [
            (
                global_imag_reward_sum / global_imag_reward_count
                if global_imag_reward_count > 0
                else 0.0
            )
        ]
        metrics["actor/ppo_kl_hard_limit"] = [float(ppo_kl_hard_limit)]
        metrics["actor/ppo_kl_hard_skip_count"] = [
            float(global_ppo_kl_hard_skip_count)
        ]
        metrics["actor/ppo_kl_hard_total_count"] = [
            float(global_ppo_kl_hard_total_count)
        ]
        metrics["actor/ppo_kl_hard_skip_ratio"] = [
            (
                float(global_ppo_kl_hard_skip_count)
                / float(global_ppo_kl_hard_total_count)
                if global_ppo_kl_hard_total_count > 0
                else 0.0
            )
        ]
        metrics["actor/ppo_kl_hard_max"] = [float(global_ppo_kl_hard_max)]
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    def update_chunk_policy(self, data: DataProto):
        """One outer-stage update with global CHUNK branch normalization."""
        from merl.algorithm import clipped_chunk_loss
        if data.batch["responses"].shape[1] != 1:
            raise ValueError("MERL actor updates require one chunk per sample")
        self.pad_token_id = data.meta_info["pad_token_id"]
        self.actor_module.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        world = dist.get_world_size() if dist.is_initialized() else 1
        total = torch.zeros((), device="cuda")
        for row in data.batch.split(1):
            row = row.cuda()
            _, logp = self._forward_micro_batch_update(
                input_ids=row["input_ids"][:, 0], attention_mask=row["attention_mask"][:, 0],
                pixel_values=row["pixel_values"][:, 0], responses=row["responses"][:, 0],
                temperature=data.meta_info["temperature"], proprio=None)
            mask = torch.arange(logp.shape[-1], device=logp.device)[None] < row["valid_response_tokens"][:, None]
            terms = clipped_chunk_loss(logp, row["old_log_probs"], row["advantages"], mask,
                                       self.config.clip_ratio_low, self.config.clip_ratio_high)
            loss = (terms * row["branch_coefficient"]).sum()
            # FSDP averages gradients over ranks; coefficients already use the
            # global branch counts, so undo this averaging exactly once.
            (world * loss).backward()
            total += loss.detach()
        norm = self._optimizer_step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        if dist.is_initialized():
            dist.all_reduce(total)
        return {"actor/pg_loss": [float(total)], "actor/grad_norm": [float(norm)],
                "actor/valid_chunks": [len(data)], "actor/optimizer_updates": [1]}

    def compute_entropy(self, bacth_data: DataProto):

        if bacth_data.meta_info["train_mode"] == True:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        assert self.config.ppo_micro_batch_size == 1
        self.gradient_accumulation = (
            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        )
        temperature = bacth_data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error
        entropy_metric_key = self._get_entropy_metric_key(bacth_data)

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "finish_step",
        ]
        select_keys += [
            key
            for key in ["valid_response_tokens", "wm_valid_response_tokens"]
            if key in bacth_data.batch
        ]
        if self.use_proprio:
            select_keys.append("proprio")
        batch = bacth_data.select(batch_keys=select_keys).batch

        #! ---------- BEGIN FSDP-SAFE EMPTY-BATCH HANDLING ----------
        # helper: check dist ready (should be true in FSDP worker)
        dist_ready = dist.is_available() and dist.is_initialized()
        if not dist_ready:
            # fallback: single-proc, treat as normal
            num_minibatch = len(batch) // self.config.ppo_mini_batch_size
            dataloader = batch.split(self.config.ppo_mini_batch_size)[:num_minibatch]
        else:
            # per-rank indicator: has any sample?
            local_has = torch.tensor(int(len(batch) > 0), device="cuda")
            min_has = local_has.clone()
            max_has = local_has.clone()
            dist.all_reduce(min_has, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_has, op=dist.ReduceOp.MAX)

            # Case 0: everyone empty -> safe global skip, but return consistent metrics
            if max_has.item() == 0:
                return {
                    entropy_metric_key: 0.0,
                    "actor/skipped_all": 1,
                }

            # Case 1: some ranks empty, some non-empty -> construct dummy sample on empty ranks
            if min_has.item() == 0 and max_has.item() == 1:
                if len(batch) == 0:
                    # gather shape templates from ranks that have a sample
                    # prepare local shape info (None if empty)
                    local_shape_info = None
                    # batch might not be a plain dict; try to extract map of key->shape if possible
                    try:
                        # attempt to build shape info from batch (works for tensor-mapped batch)
                        local_shape_info = {
                            k: tuple(v.shape)
                            for k, v in ({} if batch is None else batch.items())
                        }
                    except Exception:
                        # fallback: try to use data.select to get a single element sample if possible
                        local_shape_info = None

                    # collect shape infos
                    gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(gathered, local_shape_info)

                    # pick first non-None template
                    template = None
                    for s in gathered:
                        if s is not None:
                            template = s
                            break
                    assert (
                        template is not None
                    ), "No template shapes gathered for dummy batch (but some ranks have data)."

                    # build a single-sample dict matching template shapes (tensors on cuda)
                    sample = {}
                    for k, shape in template.items():
                        # Heuristic for dtype choice — adapt if you have explicit schema
                        if "mask" in k or "id" in k or "finish" in k:
                            dtype = torch.long
                        else:
                            dtype = torch.float
                        # create zeros tensor on cuda
                        sample[k] = torch.zeros(shape, device="cuda", dtype=dtype)

                    # create a DataProto single-sample (use the same factory you used in fit_wm_v1)
                    # Note: DataProto.from_single_dict was used elsewhere; adapt if your API differs.
                    dummy_dp = DataProto.from_single_dict(sample)
                    # set batch to the dummy's batch representation (so subsequent .split works)
                    batch = dummy_dp.batch

            # Now align number of mini-batches across ranks
            num_minibatch = len(batch) // self.config.ppo_mini_batch_size
            num_minibatch_t = torch.tensor(num_minibatch, device="cuda")
            dist.all_reduce(num_minibatch_t, op=dist.ReduceOp.MIN)
            num_minibatch = num_minibatch_t.item()
            dataloader = batch.split(self.config.ppo_mini_batch_size)[:num_minibatch]
        print("[dp_rob] compute_entory dataloader_length:", len(dataloader))
        #! ---------- END FSDP-SAFE EMPTY-BATCH HANDLING ----------

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # dataloader = batch.split(self.config.ppo_mini_batch_size)  #! commented now

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, _ = rearrange_micro_batches(
                    batch=mini_batch, max_token_len=max_token_len
                )
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                response_mask, _ = self._build_response_mask(data)

                response_mask_sum = response_mask.sum()
                _, batch_has_any_tokens, _ = self._dist_activity_flags(
                    bool(response_mask_sum.item() > 0)
                )
                if not batch_has_any_tokens:
                    entropy_loss = torch.zeros(
                        (), device=data["responses"].device, dtype=torch.float32
                    )
                else:
                    traj_len = data["responses"].size(1)
                    traj_chunk_size = self._get_initial_traj_chunk_size(traj_len)
                    entropy_sum = torch.zeros(
                        (), device=data["responses"].device, dtype=torch.float32
                    )
                    traj_start = 0

                    while traj_start < traj_len:
                        traj_end = min(traj_start + traj_chunk_size, traj_len)
                        slice_id, next_slice_id = self._traj_token_slice_bounds(
                            traj_start, traj_end
                        )
                        response_mask_tmp = response_mask[:, slice_id:next_slice_id]
                        local_chunk_has_tokens, chunk_has_any_tokens, _ = (
                            self._dist_activity_flags(
                                bool(response_mask_tmp.sum().item() > 0)
                            )
                        )
                        if not chunk_has_any_tokens:
                            traj_start = traj_end
                            continue

                        entropy_batch = self._slice_micro_batch_by_traj(
                            data, traj_start, traj_end
                        )
                        local_attention_mask = entropy_batch["attention_mask"]
                        if local_chunk_has_tokens and (
                            not local_attention_mask.ne(0).any()
                        ):
                            raise ValueError(
                                "attention_mask is fully padded in compute_entropy despite non-empty response_mask: "
                                f"chunk={traj_start}:{traj_end}, finish_step={data['finish_step'].detach().cpu().tolist()}"
                            )
                        if not local_chunk_has_tokens:
                            traj_input_ids = entropy_batch["input_ids"].reshape(
                                (-1,) + entropy_batch["input_ids"].shape[2:]
                            )
                            traj_attention_mask = local_attention_mask.reshape(
                                (-1,) + local_attention_mask.shape[2:]
                            )
                            traj_responses = entropy_batch["responses"].reshape(
                                (-1,) + entropy_batch["responses"].shape[2:]
                            )
                            (
                                traj_input_ids,
                                traj_attention_mask,
                                traj_responses,
                            ) = self._inject_dummy_active_tokens(
                                traj_input_ids,
                                traj_attention_mask,
                                traj_responses,
                            )
                            entropy_batch = dict(entropy_batch)
                            entropy_batch["input_ids"] = traj_input_ids.reshape(
                                entropy_batch["input_ids"].shape
                            )
                            entropy_batch["attention_mask"] = (
                                traj_attention_mask.reshape(
                                    entropy_batch["attention_mask"].shape
                                )
                            )
                            entropy_batch["responses"] = traj_responses.reshape(
                                entropy_batch["responses"].shape
                            )
                        with torch.no_grad():
                            entropy = self._forward_micro_batch_entropy(
                                micro_batch=entropy_batch,
                                temperature=temperature,
                            )

                        if local_chunk_has_tokens:
                            entropy_sum = (
                                entropy_sum + (entropy * response_mask_tmp).sum()
                            )
                        traj_start = traj_end

                    entropy_loss = entropy_sum / response_mask_sum

                data = {entropy_metric_key: entropy_loss.detach().item()}
                append_to_dict(metrics, data)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
