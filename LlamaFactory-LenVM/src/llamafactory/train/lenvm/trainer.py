# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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
import os
import json
from collections import OrderedDict
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from torch.utils.data import Dataset
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)

EPS = 1e-12

class LengthValueTrainer(Trainer):
    r"""Trainer for length value regression."""
    @staticmethod
    def _format_pos_key(pos: float) -> str:
        percent = pos * 100.0
        if abs(percent - round(percent)) < 1e-6:
            return str(int(round(percent)))
        return f"{percent:.2f}".replace(".", "p")

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()


        super().__init__(**kwargs)
        # Only `token-mean` consumes `num_items_in_batch` directly. The other
        # aggregation modes still rely on Trainer's default GA normalization.
        self.model_accepts_loss_kwargs = (finetuning_args.lenvm_agg_method == "token-mean")
        self.finetuning_args = finetuning_args
        self.can_return_loss = True
        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))
        
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Diagnostics accumulators for gradient accumulation-safe logging
        self._diag_ga_sum_lam0 = 0.0
        self._diag_ga_sum_lam1 = 0.0
        # Relative loss diagnostics (|y - ret| / |ret|)
        self._diag_ga_sum_rel_lam0 = 0.0
        self._diag_ga_sum_rel_lam1 = 0.0
        self._diag_ga_sum_abs_len_lam1 = 0.0
        self._diag_ga_sum_rel_len_lam1 = 0.0
        self._diag_ga_count = 0
        self._diag_ga_step = 0
        self._diag_ga_pos_fracs = list(finetuning_args.lenvm_pos_metrics)
        self._diag_ga_pos_keys = [self._format_pos_key(pos) for pos in self._diag_ga_pos_fracs]
        self._diag_ga_pos_abs = {key: 0.0 for key in self._diag_ga_pos_keys}
        self._diag_ga_pos_rel = {key: 0.0 for key in self._diag_ga_pos_keys}
        self._diag_ga_pos_val_abs_lam1 = {key: 0.0 for key in self._diag_ga_pos_keys}
        self._diag_ga_pos_val_rel_lam1 = {key: 0.0 for key in self._diag_ga_pos_keys}
        self._diag_ga_pos_count = {key: 0.0 for key in self._diag_ga_pos_keys}

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def _get_num_items_in_batch(
        self, batch_samples: list[dict[str, torch.Tensor]], device: torch.device
    ) -> Optional[Union[int, torch.Tensor]]:
        if len(batch_samples) == 0 or "value_mask" not in batch_samples[0]:
            return super()._get_num_items_in_batch(batch_samples, device)

        num_items_in_batch = None
        try:
            num_items_in_batch = sum(batch["value_mask"].sum() for batch in batch_samples)
        except (TypeError, AttributeError, KeyError):
            return super()._get_num_items_in_batch(batch_samples, device)

        if self.args.average_tokens_across_devices:
            if self.args.world_size > 1:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch.to(device)).sum()
        elif self.args.n_gpu > 1:
            num_items_in_batch = num_items_in_batch // self.args.n_gpu

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(device)

            if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                num_items_in_batch = num_items_in_batch.unsqueeze(0).expand(self.args.n_gpu, -1)
            if pc := getattr(self.accelerator, "parallelism_config", None):
                num_items_in_batch = num_items_in_batch // pc.non_data_parallel_size

        return num_items_in_batch

    def _forward_value(self, model: "PreTrainedModel", inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            value_preds = outputs.get("value", outputs.get("values"))
        elif isinstance(outputs, tuple):
            if len(outputs) >= 3:
                value_preds = outputs[2]
            else:
                raise ValueError("Model outputs do not contain value predictions.")
        else:
            raise ValueError("Unsupported model output type for length value trainer.")

        if value_preds.dim() == 3:
            value_preds = value_preds.squeeze(-1)

        return value_preds

    @staticmethod
    def _value_to_length(v: torch.Tensor, gamma: float, eps: float = EPS) -> torch.Tensor:
        r"""Convert discounted return value v in (-1, 0) to remaining length.

        Paper: l = ln(1 + v) / ln(gamma).
        """
        gamma_t = torch.tensor(gamma, dtype=v.dtype, device=v.device)
        denom = torch.log(gamma_t).clamp_max(-eps)  # ln(gamma) < 0, avoid divide-by-0 when gamma≈1
        v_clamped = v.clamp(min=-1.0 + eps, max=0.0 - eps)  # keep 1+v in (0,1)
        return torch.log1p(v_clamped) / denom

    @staticmethod
    def _length_to_value(length: torch.Tensor, gamma: float) -> torch.Tensor:
        r"""Convert remaining length to discounted return value."""
        if gamma == 1.0:
            return torch.zeros_like(length)

        gamma_t = torch.tensor(gamma, dtype=length.dtype, device=length.device)
        return -(1.0 - torch.pow(gamma_t, length.clamp_min(0.0)))

    def _get_target_type(self) -> str:
        return self.finetuning_args.lenvm_target_type

    def _get_length_normalizer(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.finetuning_args.lenvm_length_normalizer, dtype=dtype, device=device)

    def _raw_to_length_prediction(self, raw_preds: torch.Tensor) -> torch.Tensor:
        activation = self.finetuning_args.lenvm_length_activation
        if activation == "softplus":
            return F.softplus(raw_preds)
        elif activation == "sigmoid":
            return torch.sigmoid(raw_preds)
        else:
            raise ValueError(f"Unsupported length activation: {activation}")

    def _raw_to_target_prediction(self, raw_preds: torch.Tensor) -> torch.Tensor:
        target_type = self._get_target_type()
        if target_type == "discounted_return":
            return -torch.sigmoid(raw_preds)
        elif target_type == "length":
            return self._raw_to_length_prediction(raw_preds)
        elif target_type == "log_length":
            return F.softplus(raw_preds)
        else:
            raise ValueError(f"Unsupported LenVM target type: {target_type}")

    def _build_regression_target(
        self,
        pred_target: torch.Tensor,
        value_mask: torch.Tensor,
        value_labels: torch.Tensor,
        gamma: float,
        lam: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        target_type = self._get_target_type()
        if target_type == "discounted_return":
            _, returns = self._compute_gae_advantage_return(pred_target, value_mask, value_labels, gamma, lam)
            return returns
        elif target_type == "length":
            length_scale = self._get_length_normalizer(pred_target.dtype, pred_target.device)
            return value_labels.to(pred_target.dtype) / length_scale
        elif target_type == "log_length":
            return torch.log1p(value_labels.to(pred_target.dtype))
        else:
            raise ValueError(f"Unsupported LenVM target type: {target_type}")

    def _prediction_to_length(self, pred_target: torch.Tensor, gamma: float) -> torch.Tensor:
        target_type = self._get_target_type()
        if target_type == "discounted_return":
            return self._value_to_length(pred_target, gamma)
        elif target_type == "length":
            length_scale = self._get_length_normalizer(pred_target.dtype, pred_target.device)
            return pred_target * length_scale
        elif target_type == "log_length":
            return torch.expm1(pred_target)
        else:
            raise ValueError(f"Unsupported LenVM target type: {target_type}")

    def _prediction_to_discounted_value(self, pred_target: torch.Tensor, gamma: float) -> torch.Tensor:
        target_type = self._get_target_type()
        if target_type == "discounted_return":
            return pred_target

        len_pred = self._prediction_to_length(pred_target, gamma)
        return self._length_to_value(len_pred, gamma)

    @staticmethod
    def _fixed_step_reward(
        gamma: float, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        r"""Compute per-step reward = -(1 - gamma) as a tensor.

        Validation that reward == -(1-gamma) is done once in __init__; callers
        no longer need to pass a `reward` argument.
        """
        gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
        return -(1.0 - gamma_t)

    # def _apply_decay_factor(self, value_labels: torch.Tensor, value_mask: torch.Tensor) -> torch.Tensor:
    #     r"""Apply decay factor to future token values.
        
    #     Args:
    #         value_labels: Tensor of shape (batch_size, seq_len) containing the number of future tokens
    #         value_mask: Tensor of shape (batch_size, seq_len) indicating valid positions
            
    #     Returns:
    #         Modified value_labels with decay applied
    #     """
    #     if self.finetuning_args.lenvm_decay_factor == 1.0:
    #         return value_labels
            
    #     # Create decay weights: for each position, apply decay_factor^(n-1) where n is the distance
    #     batch_size, seq_len = value_labels.shape
    #     device = value_labels.device
        
    #     # Create position indices for each sequence
    #     positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
    #     # Calculate decay weights: decay_factor^(n-1) where n is the distance from current token
    #     # For value_labels[i, j], the distance is value_labels[i, j], so decay is decay_factor^(value_labels[i, j]-1)
    #     decay_weights = torch.pow(self.finetuning_args.lenvm_decay_factor, value_labels)
        
    #     # Apply decay weights to value_labels
    #     decayed_labels = 1 - decay_weights
        
    #     # Only apply decay where mask is valid
    #     decayed_labels = decayed_labels * value_mask
        
    #     return decayed_labels

    def _compute_gae_advantage_return(
        self,
        y_hat: torch.Tensor,
        value_mask: torch.Tensor,
        value_labels: torch.Tensor,
        gamma: float,
        lam: Union[float, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute token-level GAE advantages and λ-return targets.

        This implementation follows the paper:
          δ_t = r_t + γ v_{t+1}^{old} - v_t^{old}
          A_t = δ_t + γλ A_{t+1}
          v_t^{tgt} = v_t^{old} + A_t

        Key detail for cutoff truncation:
        - When the trajectory continues beyond the observed window (no token t+1 in the batch),
          we bootstrap v_{t+1}^{old} using the deterministic Monte-Carlo return derived from
          the ground-truth remaining length `value_labels` (see `value_regression.py`).

        Notes:
        - `value_mask` selects valid regression positions (prompt last token + response tokens, including EOS if enabled).
        - `value_labels` is the remaining number of tokens to go until EOS for each position.
        - Per-step reward is -(1-γ), computed internally by `_fixed_step_reward`.
        """
        dtype = y_hat.dtype
        device = y_hat.device

        # We only regress on mask=1 positions; everything else is ignored by the loss.
        value_mask_f = value_mask.to(dtype)
        remaining = value_labels.to(dtype)

        # stopgrad baseline
        v_old = y_hat.detach()

        lam_t = torch.as_tensor(lam, dtype=dtype, device=device)
        reward_t = self._fixed_step_reward(gamma, dtype, device)
        gamma_t = torch.tensor(gamma, dtype=dtype, device=device)

        # Deterministic MC return from remaining length:
        # reward = -(1-γ) => G_t = -(1 - γ^n)
        # where n = remaining steps until EOS (i.e., `remaining`).
        def mc_value_from_remaining(n: torch.Tensor) -> torch.Tensor:
            if gamma == 1.0:
                # Degenerate case; in practice gamma∈(0,1).
                return torch.zeros_like(n)
            return -(1.0 - torch.pow(gamma_t, n))

        B, T = v_old.shape
        advantages = torch.zeros((B, T), dtype=dtype, device=device)
        lastgaelam = torch.zeros((B,), dtype=dtype, device=device)

        for t in reversed(range(T)):
            mask_t = value_mask_f[:, t]
            is_eos = remaining[:, t] <= 0.0  # EOS token has remaining length 0
            if t < T - 1:
                v_next_model = v_old[:, t + 1]
                # If the next token is outside the observed window for this sequence
                # (e.g., cutoff truncation or padding), bootstrap from deterministic MC value.
                need_det = (remaining[:, t] > 1.0) & (value_mask_f[:, t + 1] == 0.0)
                v_next_det = mc_value_from_remaining(torch.clamp(remaining[:, t] - 1.0, min=0.0))
                v_next = torch.where(need_det, v_next_det, v_next_model)
                # If the true next step is EOS (remaining==1), its value is deterministic 0.
                v_next = torch.where(remaining[:, t] <= 1.0, torch.zeros_like(v_next), v_next)
            else: # t == T - 1
                # No t+1 token in the observed window. Bootstrap from the deterministic return
                # of the next state using ground-truth remaining length.
                # Next state's remaining length is (remaining-1).
                v_next = torch.where(
                    remaining[:, t] <= 1.0,
                    torch.zeros((B,), dtype=dtype, device=device),
                    mc_value_from_remaining(torch.clamp(remaining[:, t] - 1.0, min=0.0)),
                )

            delta = reward_t + gamma * v_next - v_old[:, t]
            gae_t = delta + gamma * lam_t * lastgaelam

            # EOS token (remaining==0) is terminal: target return is exactly 0.
            # We set its advantage to -v_old so that return=v_old+adv=0, and we
            # reset the recursion so EOS does not leak into earlier steps.
            eos_adv = -v_old[:, t]
            adv_to_use = torch.where(is_eos, eos_adv, gae_t)

            # Only update recursion on valid positions.
            new_lastgaelam = adv_to_use * mask_t + lastgaelam * (1.0 - mask_t)
            # Reset recursion at EOS so A_{t-1} does not include A_eos.
            new_lastgaelam = torch.where(is_eos & (mask_t > 0), torch.zeros_like(new_lastgaelam), new_lastgaelam)

            # Store: keep EOS advantage (so EOS trains to 0), otherwise store the recursion value.
            advantages[:, t] = torch.where(is_eos, eos_adv * mask_t, new_lastgaelam)
            lastgaelam = new_lastgaelam

        returns = v_old + advantages
        return advantages, returns

    def aggregate_loss(self, loss: torch.Tensor, value_mask: torch.Tensor, method: str = "token-mean") -> torch.Tensor:
        if method == "token-mean":
            value_mask = value_mask.to(loss.dtype)
            total = (loss * value_mask).sum()
            denom = value_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return total / denom
        elif method == "seq-mean-token-sum":
            value_mask = value_mask.to(loss.dtype)
            per_seq_sum = (loss * value_mask).sum(dim=-1)
            valid_seq_mask = (value_mask.sum(dim=-1) > 0).to(loss.dtype)
            denom = valid_seq_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return (per_seq_sum * valid_seq_mask).sum() / denom
        elif method == "seq-mean-token-mean":
            value_mask_f = value_mask.to(loss.dtype)
            per_seq_token_count = value_mask_f.sum(dim=-1).clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            per_seq_mean = (loss * value_mask_f).sum(dim=-1) / per_seq_token_count
            valid_seq_mask = (value_mask.sum(dim=1) > 0).to(loss.dtype)
            denom = valid_seq_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return (per_seq_mean * valid_seq_mask).sum() / denom
        elif method == "seq-sum-token-sum":
            value_mask = value_mask.to(loss.dtype)
            # Sum over tokens within each sequence, then sum over all sequences.
            # This is an unnormalized total loss over all valid tokens.
            return (loss * value_mask).sum()
        elif method == "seq-mean-token-mean-max":
            # Divide per-sequence sum by a fixed max-length denominator instead of actual
            # token count to prevent overflow when sequences are very long.
            value_mask = value_mask.to(loss.dtype)
            per_seq_sum = (loss * value_mask).sum(dim=-1) / 5000.0
            valid_seq_mask = (value_mask.sum(dim=-1) > 0).to(loss.dtype)
            denom = valid_seq_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return (per_seq_sum * valid_seq_mask).sum() / denom
        else:
            raise ValueError(f"Invalid method: {method}")

    def _compute_value_diagnostics(
        self,
        y_hat: torch.Tensor,
        value_mask: torch.Tensor,
        value_labels: torch.Tensor,
        gamma: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        OrderedDict[str, tuple[float, float, float, float, float]],
    ]:
        r"""Compute diagnostics used for both train and predict.

        Vectorized — no Python loop over T. Uses closed-form expressions:
          - returns_lam1 = -(1 - γ^n)              (MC return, exact)
          - returns_lam0 = r + γ · v_next           (1-step TD, vectorized shift)

        Returns:
            - sum_l0, sum_l1: summed squared errors for lam0/lam1 over valid tokens
            - sum_rel_l0, sum_rel_l1: summed relative errors for lam0/lam1 over valid tokens
            - sum_abs_len_l1: summed length-space absolute error for lam1 over valid tokens
            - sum_rel_len_l1: summed length-space relative error for lam1 over valid tokens
            - pos_metrics: OrderedDict keyed by position key (e.g., "0", "25", "50"...")
                mapping to (abs_sum, rel_sum, val_abs_sum, val_rel_sum, count) where:
                - abs_sum, rel_sum are in *length space* (len_pred vs true remaining length)
                - val_abs_sum is absolute error in *value space* for lam1 (|y_hat - returns_lam1|)
                - val_rel_sum is relative error in *value space* for lam1 (y_hat vs returns_lam1)
        """
        dtype = y_hat.dtype
        device = y_hat.device
        v_old = y_hat.detach()
        B, T = v_old.shape

        value_mask_f = value_mask.to(dtype)
        remaining = value_labels.to(dtype)
        gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
        reward_t = self._fixed_step_reward(gamma, dtype, device)

        # ── returns_lam1: MC return = -(1 - γ^n), no loop needed ──
        returns_lam1 = -(1.0 - torch.pow(gamma_t, remaining))
        # EOS token (remaining==0) is terminal and should have target return 0.
        returns_lam1 = torch.where(remaining <= 0.0, torch.zeros_like(returns_lam1), returns_lam1)

        # ── returns_lam0: 1-step TD target = r + γ · v_next, vectorized ──
        # MC value for next state (remaining - 1 steps)
        mc_next = -(1.0 - torch.pow(gamma_t, torch.clamp(remaining - 1.0, min=0.0)))

        v_next = torch.zeros((B, T), dtype=dtype, device=device)
        if T > 1:
            # Default: use model's own prediction at t+1
            v_next[:, :-1] = v_old[:, 1:]
            # Bootstrap from MC where next position is outside valid window
            need_det = (remaining[:, :-1] > 1.0) & (value_mask_f[:, 1:] == 0.0)
            v_next[:, :-1] = torch.where(need_det, mc_next[:, :-1], v_next[:, :-1])
            # If current is last before EOS (remaining<=1), v_next = 0
            v_next[:, :-1] = torch.where(
                remaining[:, :-1] <= 1.0, torch.zeros_like(v_next[:, :-1]), v_next[:, :-1]
            )
        # t == T-1: no t+1 in window, always bootstrap
        v_next[:, -1] = torch.where(
            remaining[:, -1] <= 1.0,
            torch.zeros((B,), dtype=dtype, device=device),
            mc_next[:, -1],
        )
        returns_lam0 = reward_t + gamma * v_next
        # EOS token (remaining==0) is terminal and should have target return 0.
        returns_lam0 = torch.where(remaining <= 0.0, torch.zeros_like(returns_lam0), returns_lam0)

        loss_lam0 = 0.5 * (y_hat - returns_lam0) ** 2
        loss_lam1 = 0.5 * (y_hat - returns_lam1) ** 2
        _abs_ret_lam0 = torch.abs(returns_lam0)
        _abs_ret_lam1 = torch.abs(returns_lam1)
        rel_loss_lam0 = torch.where(_abs_ret_lam0 > 0, torch.abs(y_hat - returns_lam0) / _abs_ret_lam0, torch.zeros_like(y_hat))
        rel_loss_lam1 = torch.where(_abs_ret_lam1 > 0, torch.abs(y_hat - returns_lam1) / _abs_ret_lam1, torch.zeros_like(y_hat))

        len_pred = self._value_to_length(y_hat, gamma)
        len_tgt = remaining  # true remaining length (no round-trip needed)
        _abs_len_tgt = torch.abs(len_tgt)
        abs_len_loss = torch.abs(len_pred - len_tgt)
        rel_len_loss = torch.where(_abs_len_tgt > 0, abs_len_loss / _abs_len_tgt, torch.zeros_like(len_pred))

        sum_l0 = self.aggregate_loss(loss_lam0, value_mask, method="seq-sum-token-sum")
        sum_l1 = self.aggregate_loss(loss_lam1, value_mask, method="seq-sum-token-sum")
        sum_rel_l0 = self.aggregate_loss(rel_loss_lam0, value_mask, method="seq-sum-token-sum")
        sum_rel_l1 = self.aggregate_loss(rel_loss_lam1, value_mask, method="seq-sum-token-sum")
        sum_abs_len_l1 = self.aggregate_loss(abs_len_loss, value_mask, method="seq-sum-token-sum")
        sum_rel_len_l1 = self.aggregate_loss(rel_len_loss, value_mask, method="seq-sum-token-sum")

        pos_metrics: OrderedDict[str, tuple[float, float, float, float, float]] = OrderedDict()
        for key in self._diag_ga_pos_keys:
            pos_metrics[key] = (0.0, 0.0, 0.0, 0.0, 0.0)

        if self._diag_ga_pos_keys:
            value_mask_bool = value_mask > 0
            if value_mask_bool.any():
                rank = torch.cumsum(value_mask_bool.to(torch.long), dim=1) - 1
                valid_counts = value_mask_bool.sum(dim=1)
                abs_err = torch.abs(len_pred - len_tgt)
                rel_err = torch.where(_abs_len_tgt > 0, abs_err / _abs_len_tgt, torch.zeros_like(abs_err))
                val_abs_err = torch.abs(y_hat - returns_lam1)
                val_rel_err = rel_loss_lam1
                for pos, key in zip(self._diag_ga_pos_fracs, self._diag_ga_pos_keys):
                    pos_rank = torch.floor((valid_counts - 1).clamp_min(0) * pos).to(torch.long)
                    select = (rank == pos_rank[:, None]) & value_mask_bool
                    count = float(select.sum().item())
                    if count > 0:
                        select_f = select.to(abs_err.dtype)
                        abs_sum = float((abs_err * select_f).sum().item())
                        rel_sum = float((rel_err * select_f).sum().item())
                        val_abs_sum = float((val_abs_err * select_f).sum().item())
                        val_rel_sum = float((val_rel_err * select_f).sum().item())
                        pos_metrics[key] = (abs_sum, rel_sum, val_abs_sum, val_rel_sum, count)

        return sum_l0, sum_l1, sum_rel_l0, sum_rel_l1, sum_abs_len_l1, sum_rel_len_l1, pos_metrics

    def _init_nontrain_diag(self, prefix: str) -> None:
        if hasattr(self, f"_diag_{prefix}_sum_lam0"):
            return

        setattr(self, f"_diag_{prefix}_sum_lam0", 0.0)
        setattr(self, f"_diag_{prefix}_sum_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_sum_rel_lam0", 0.0)
        setattr(self, f"_diag_{prefix}_sum_rel_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_sum_abs_len_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_sum_rel_len_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_count", 0.0)
        setattr(self, f"_diag_{prefix}_step", 0)

        setattr(self, f"_diag_{prefix}_pos_abs", {key: 0.0 for key in self._diag_ga_pos_keys})
        setattr(self, f"_diag_{prefix}_pos_rel", {key: 0.0 for key in self._diag_ga_pos_keys})
        setattr(self, f"_diag_{prefix}_pos_val_abs_lam1", {key: 0.0 for key in self._diag_ga_pos_keys})
        setattr(self, f"_diag_{prefix}_pos_val_rel_lam1", {key: 0.0 for key in self._diag_ga_pos_keys})
        setattr(self, f"_diag_{prefix}_pos_count", {key: 0.0 for key in self._diag_ga_pos_keys})

    def _accumulate_nontrain_diag(
        self,
        prefix: str,
        sum_l0: torch.Tensor,
        sum_l1: torch.Tensor,
        sum_rel_l0: torch.Tensor,
        sum_rel_l1: torch.Tensor,
        sum_abs_len_l1: torch.Tensor,
        sum_rel_len_l1: torch.Tensor,
        pos_metrics: OrderedDict[str, tuple[float, float, float, float, float]],
        cnt: torch.Tensor,
    ) -> None:
        self._init_nontrain_diag(prefix)
        setattr(self, f"_diag_{prefix}_sum_lam0", getattr(self, f"_diag_{prefix}_sum_lam0") + float(sum_l0.detach()))
        setattr(self, f"_diag_{prefix}_sum_lam1", getattr(self, f"_diag_{prefix}_sum_lam1") + float(sum_l1.detach()))
        setattr(
            self,
            f"_diag_{prefix}_sum_rel_lam0",
            getattr(self, f"_diag_{prefix}_sum_rel_lam0") + float(sum_rel_l0.detach()),
        )
        setattr(
            self,
            f"_diag_{prefix}_sum_rel_lam1",
            getattr(self, f"_diag_{prefix}_sum_rel_lam1") + float(sum_rel_l1.detach()),
        )
        setattr(
            self,
            f"_diag_{prefix}_sum_abs_len_lam1",
            getattr(self, f"_diag_{prefix}_sum_abs_len_lam1") + float(sum_abs_len_l1.detach()),
        )
        setattr(
            self,
            f"_diag_{prefix}_sum_rel_len_lam1",
            getattr(self, f"_diag_{prefix}_sum_rel_len_lam1") + float(sum_rel_len_l1.detach()),
        )
        setattr(self, f"_diag_{prefix}_count", getattr(self, f"_diag_{prefix}_count") + float(cnt.detach()))
        setattr(self, f"_diag_{prefix}_step", getattr(self, f"_diag_{prefix}_step") + 1)

        pos_abs = getattr(self, f"_diag_{prefix}_pos_abs")
        pos_rel = getattr(self, f"_diag_{prefix}_pos_rel")
        pos_val_abs_lam1 = getattr(self, f"_diag_{prefix}_pos_val_abs_lam1")
        pos_val_rel_lam1 = getattr(self, f"_diag_{prefix}_pos_val_rel_lam1")
        pos_cnt = getattr(self, f"_diag_{prefix}_pos_count")
        for key, (abs_sum, rel_sum, val_abs_sum, val_rel_sum, count) in pos_metrics.items():
            pos_abs[key] += abs_sum
            pos_rel[key] += rel_sum
            pos_val_abs_lam1[key] += val_abs_sum
            pos_val_rel_lam1[key] += val_rel_sum
            pos_cnt[key] += count

    def _flush_nontrain_logs(self, prefix: str) -> None:
        if not hasattr(self, f"_diag_{prefix}_sum_lam0"):
            return

        steps = getattr(self, f"_diag_{prefix}_step")
        count = getattr(self, f"_diag_{prefix}_count")
        if steps == 0 or count == 0.0:
            return

        # `Trainer` may not expose `self.device` in some transformers versions.
        if getattr(self, "args", None) is not None and getattr(self.args, "device", None) is not None:
            device = self.args.device
        elif getattr(self, "accelerator", None) is not None:
            device = self.accelerator.device
        else:
            device = next(self.model.parameters()).device
        sum_l0_t = torch.tensor(getattr(self, f"_diag_{prefix}_sum_lam0"), device=device, dtype=torch.float64)
        sum_l1_t = torch.tensor(getattr(self, f"_diag_{prefix}_sum_lam1"), device=device, dtype=torch.float64)
        sum_rel_l0_t = torch.tensor(getattr(self, f"_diag_{prefix}_sum_rel_lam0"), device=device, dtype=torch.float64)
        sum_rel_l1_t = torch.tensor(getattr(self, f"_diag_{prefix}_sum_rel_lam1"), device=device, dtype=torch.float64)
        sum_abs_len_l1_t = torch.tensor(
            getattr(self, f"_diag_{prefix}_sum_abs_len_lam1"), device=device, dtype=torch.float64
        )
        sum_rel_len_l1_t = torch.tensor(
            getattr(self, f"_diag_{prefix}_sum_rel_len_lam1"), device=device, dtype=torch.float64
        )

        pos_abs = {
            k: torch.tensor(v, device=device, dtype=torch.float64)
            for k, v in getattr(self, f"_diag_{prefix}_pos_abs").items()
        }
        pos_rel = {
            k: torch.tensor(v, device=device, dtype=torch.float64)
            for k, v in getattr(self, f"_diag_{prefix}_pos_rel").items()
        }
        pos_val_abs_lam1 = {
            k: torch.tensor(v, device=device, dtype=torch.float64)
            for k, v in getattr(self, f"_diag_{prefix}_pos_val_abs_lam1").items()
        }
        pos_val_rel_lam1 = {
            k: torch.tensor(v, device=device, dtype=torch.float64)
            for k, v in getattr(self, f"_diag_{prefix}_pos_val_rel_lam1").items()
        }
        pos_cnt = {
            k: torch.tensor(v, device=device, dtype=torch.float64)
            for k, v in getattr(self, f"_diag_{prefix}_pos_count").items()
        }
        cnt_t = torch.tensor(getattr(self, f"_diag_{prefix}_count"), device=device, dtype=torch.float64)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_l0_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_l1_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_rel_l0_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_rel_l1_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_abs_len_l1_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_rel_len_l1_t, op=torch.distributed.ReduceOp.SUM)
            for key in pos_abs:
                torch.distributed.all_reduce(pos_abs[key], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(pos_rel[key], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(pos_val_abs_lam1[key], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(pos_val_rel_lam1[key], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(pos_cnt[key], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)

        if self.accelerator.is_main_process:
            log_payload = {
                # Use `{prefix}_...` so HF integrations (e.g., WandbCallback) can recognize
                # eval/test-like metrics and avoid turning `eval/...` into `train/eval/...`.
                f"{prefix}_value_loss_lam0": (sum_l0_t / cnt_t.clamp_min(1.0)).item(),
                f"{prefix}_value_loss_lam1": (sum_l1_t / cnt_t.clamp_min(1.0)).item(),
                f"{prefix}_value_rel_loss_lam0": (sum_rel_l0_t / cnt_t.clamp_min(1.0)).item(),
                f"{prefix}_value_rel_loss_lam1": (sum_rel_l1_t / cnt_t.clamp_min(1.0)).item(),
                f"{prefix}_len_abs_loss_lam1": (sum_abs_len_l1_t / cnt_t.clamp_min(1.0)).item(),
                f"{prefix}_len_rel_loss_lam1": (sum_rel_len_l1_t / cnt_t.clamp_min(1.0)).item(),
            }
            for key in pos_abs:
                denom = pos_cnt[key].clamp_min(1.0)
                log_payload[f"{prefix}_len_abs_p{key}"] = (pos_abs[key] / denom).item()
                log_payload[f"{prefix}_len_rel_p{key}"] = (pos_rel[key] / denom).item()
                log_payload[f"{prefix}_value_abs_lam1_p{key}"] = (pos_val_abs_lam1[key] / denom).item()
                log_payload[f"{prefix}_value_rel_lam1_p{key}"] = (pos_val_rel_lam1[key] / denom).item()
            self.log(log_payload)

        setattr(self, f"_diag_{prefix}_sum_lam0", 0.0)
        setattr(self, f"_diag_{prefix}_sum_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_sum_rel_lam0", 0.0)
        setattr(self, f"_diag_{prefix}_sum_rel_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_sum_abs_len_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_sum_rel_len_lam1", 0.0)
        setattr(self, f"_diag_{prefix}_count", 0.0)
        setattr(self, f"_diag_{prefix}_step", 0)
        for key in self._diag_ga_pos_keys:
            getattr(self, f"_diag_{prefix}_pos_abs")[key] = 0.0
            getattr(self, f"_diag_{prefix}_pos_rel")[key] = 0.0
            getattr(self, f"_diag_{prefix}_pos_val_abs_lam1")[key] = 0.0
            getattr(self, f"_diag_{prefix}_pos_val_rel_lam1")[key] = 0.0
            getattr(self, f"_diag_{prefix}_pos_count")[key] = 0.0

    @override
    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ):
        value_labels = inputs.pop("value_labels").to(torch.float64)  # remaining number of tokens
        value_mask = inputs.pop("value_mask").to(torch.float64)  # mask of response tokens

        value_preds = self._forward_value(model, inputs).to(torch.float64)

        gamma = self.finetuning_args.lenvm_gamma
        lam = self.finetuning_args.lenvm_lam
        target_type = self._get_target_type()

        delta = self.finetuning_args.lenvm_huber_loss_delta
        agg_method = self.finetuning_args.lenvm_agg_method

        if self.finetuning_args.lenvm_loss_type == "bce":
            # BCE loss: target = 1 - γ^l ∈ [0,1], applied directly to raw logits.
            # No GAE needed — uses the MC return directly.
            gamma_t = torch.tensor(gamma, dtype=value_preds.dtype, device=value_preds.device)
            remaining = value_labels.to(value_preds.dtype)
            bce_target = 1.0 - torch.pow(gamma_t, remaining.clamp_min(0.0))
            # EOS (remaining==0) => target = 0
            bce_target = torch.where(remaining <= 0, torch.zeros_like(bce_target), bce_target)
            per_token_loss = F.binary_cross_entropy_with_logits(
                value_preds, bce_target, reduction="none"
            )
            # For diagnostics: pred in discounted-return space = -sigmoid(raw)
            pred_target = -torch.sigmoid(value_preds)
        else:
            pred_target = self._raw_to_target_prediction(value_preds)

            with torch.no_grad():
                target = self._build_regression_target(pred_target, value_mask, value_labels, gamma, lam)

            # Optionally convert to relative loss so that positions with
            # larger target values do not dominate the overall loss.
            if self.finetuning_args.lenvm_relative_loss:
                if target_type == "discounted_return":
                    # If EOS is included in `value_mask`, its target return is exactly 0.
                    # Using a tiny epsilon (1e-12) makes relative loss explode (~1e11-1e12) on EOS.
                    # Use a principled floor based on the per-step reward magnitude (1-gamma).
                    denom_floor = torch.tensor(1.0 - gamma, dtype=target.dtype, device=target.device)
                    denom = target.abs().clamp_min(denom_floor)
                    # Do not amplify EOS positions; keep them on absolute loss scale.
                    denom = torch.where(value_labels <= 0.0, torch.ones_like(denom), denom)
                else:
                    # For direct length-space baselines, keep zero-length targets on absolute scale.
                    denom = target.abs().clamp_min(torch.tensor(1.0, dtype=target.dtype, device=target.device))
                    denom = torch.where(value_labels <= 0.0, torch.ones_like(denom), denom)
                # Relative error before Huber
                rel_diff = (pred_target - target) / denom
                per_token_loss = torch.nn.functional.huber_loss(
                    rel_diff, torch.zeros_like(rel_diff), reduction="none", delta=delta
                )
            else:
                per_token_loss = torch.nn.functional.huber_loss(
                    pred_target, target, reduction="none", delta=delta
                )
        # value_loss = 0.5 * (pred_target-target) ** 2
        if agg_method == "token-mean" and num_items_in_batch is not None:
            value_mask_f = value_mask.to(per_token_loss.dtype)
            value_loss = (per_token_loss * value_mask_f).sum()
            if torch.is_tensor(num_items_in_batch):
                denom = num_items_in_batch.to(device=value_loss.device, dtype=value_loss.dtype)
            else:
                denom = torch.tensor(float(num_items_in_batch), device=value_loss.device, dtype=value_loss.dtype)

            denom = denom.clamp_min(torch.tensor(1.0, device=value_loss.device, dtype=value_loss.dtype))
            value_loss = value_loss / denom
            if self.args.average_tokens_across_devices:
                value_loss = value_loss * (
                    self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
                )
        else:
            value_loss = self.aggregate_loss(per_token_loss, value_mask, method=agg_method)

        # Training-only diagnostics/logging. Avoid polluting train accumulators when `compute_loss`
        # is called from eval/predict (e.g., `prediction_step`).
        if model.training:
            with torch.no_grad():
                diag_value_pred = self._prediction_to_discounted_value(pred_target, gamma)
                sum_l0, sum_l1, sum_rel_l0, sum_rel_l1, sum_abs_len_l1, sum_rel_len_l1, pos_metrics = self._compute_value_diagnostics(
                    y_hat=diag_value_pred, value_mask=value_mask, value_labels=value_labels, gamma=gamma,
                )

                for key, (abs_sum, rel_sum, val_abs_sum, val_rel_sum, count) in pos_metrics.items():
                    self._diag_ga_pos_abs[key] += abs_sum
                    self._diag_ga_pos_rel[key] += rel_sum
                    self._diag_ga_pos_val_abs_lam1[key] += val_abs_sum
                    self._diag_ga_pos_val_rel_lam1[key] += val_rel_sum
                    self._diag_ga_pos_count[key] += count

                # Accumulate diagnostics across micro-steps (gradient accumulation)
                ga_steps = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
                batch_token_count = float(value_mask.sum().detach())
                self._diag_ga_sum_lam0 += float(sum_l0.detach())
                self._diag_ga_sum_lam1 += float(sum_l1.detach())
                self._diag_ga_sum_rel_lam0 += float(sum_rel_l0.detach())
                self._diag_ga_sum_rel_lam1 += float(sum_rel_l1.detach())
                self._diag_ga_sum_abs_len_lam1 += float(sum_abs_len_l1.detach())
                self._diag_ga_sum_rel_len_lam1 += float(sum_rel_len_l1.detach())
                self._diag_ga_count += batch_token_count
                self._diag_ga_step = (self._diag_ga_step + 1) % ga_steps
                # Only log (and reset) at optimizer-step boundaries aligned with logging_steps.
                # Accumulators keep accumulating across optimizer steps until the next log point,
                # so logged metrics average over all data since the last log — consistent with
                # how HF Trainer reports train_loss.
                if self._diag_ga_step == 0 and self.state is not None and self.args is not None:
                    # At this point global_step has NOT been incremented yet for the
                    # current optimizer step, so the "true" current step is global_step + 1.
                    current_step = self.state.global_step + 1
                    if self.args.logging_steps > 0 and (current_step % self.args.logging_steps == 0):
                        device = value_loss.device
                        sum_l0 = torch.tensor(self._diag_ga_sum_lam0, device=device, dtype=torch.float64)
                        sum_l1 = torch.tensor(self._diag_ga_sum_lam1, device=device, dtype=torch.float64)
                        sum_rel_l0 = torch.tensor(self._diag_ga_sum_rel_lam0, device=device, dtype=torch.float64)
                        sum_rel_l1 = torch.tensor(self._diag_ga_sum_rel_lam1, device=device, dtype=torch.float64)
                        sum_abs_len_l1 = torch.tensor(
                            self._diag_ga_sum_abs_len_lam1, device=device, dtype=torch.float64
                        )
                        sum_rel_len_l1 = torch.tensor(
                            self._diag_ga_sum_rel_len_lam1, device=device, dtype=torch.float64
                        )
                        pos_abs = {
                            k: torch.tensor(v, device=device, dtype=torch.float64)
                            for k, v in self._diag_ga_pos_abs.items()
                        }
                        pos_rel = {
                            k: torch.tensor(v, device=device, dtype=torch.float64)
                            for k, v in self._diag_ga_pos_rel.items()
                        }
                        pos_val_abs_lam1 = {
                            k: torch.tensor(v, device=device, dtype=torch.float64)
                            for k, v in self._diag_ga_pos_val_abs_lam1.items()
                        }
                        pos_val_rel_lam1 = {
                            k: torch.tensor(v, device=device, dtype=torch.float64)
                            for k, v in self._diag_ga_pos_val_rel_lam1.items()
                        }
                        pos_cnt = {
                            k: torch.tensor(v, device=device, dtype=torch.float64)
                            for k, v in self._diag_ga_pos_count.items()
                        }
                        cnt = torch.tensor(self._diag_ga_count, device=device, dtype=torch.float64)  # total tokens
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(sum_l0, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(sum_l1, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(sum_rel_l0, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(sum_rel_l1, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(sum_abs_len_l1, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(sum_rel_len_l1, op=torch.distributed.ReduceOp.SUM)
                            for key in pos_abs:
                                torch.distributed.all_reduce(pos_abs[key], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(pos_rel[key], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(pos_val_abs_lam1[key], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(pos_val_rel_lam1[key], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(pos_cnt[key], op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(cnt, op=torch.distributed.ReduceOp.SUM)

                        if self.accelerator.is_main_process:
                            avg_l0 = (sum_l0 / cnt.clamp_min(1.0)).item()
                            avg_l1 = (sum_l1 / cnt.clamp_min(1.0)).item()
                            avg_rel_l0 = (sum_rel_l0 / cnt.clamp_min(1.0)).item()
                            avg_rel_l1 = (sum_rel_l1 / cnt.clamp_min(1.0)).item()
                            avg_abs_len_l1 = (sum_abs_len_l1 / cnt.clamp_min(1.0)).item()
                            avg_rel_len_l1 = (sum_rel_len_l1 / cnt.clamp_min(1.0)).item()
                            log_payload = {
                                "value_loss_lam0": avg_l0,
                                "value_loss_lam1": avg_l1,
                                "value_rel_loss_lam0": avg_rel_l0,
                                "value_rel_loss_lam1": avg_rel_l1,
                                "len_abs_loss_lam1": avg_abs_len_l1,
                                "len_rel_loss_lam1": avg_rel_len_l1,
                            }
                            for key in pos_abs:
                                denom = pos_cnt[key].clamp_min(1.0)
                                log_payload[f"len_abs_p{key}"] = (pos_abs[key] / denom).item()
                                log_payload[f"len_rel_p{key}"] = (pos_rel[key] / denom).item()
                                log_payload[f"value_abs_lam1_p{key}"] = (pos_val_abs_lam1[key] / denom).item()
                                log_payload[f"value_rel_lam1_p{key}"] = (pos_val_rel_lam1[key] / denom).item()
                            self.log(log_payload)
                        # Reset accumulators after logging
                        self._diag_ga_sum_lam0 = 0.0
                        self._diag_ga_sum_lam1 = 0.0
                        self._diag_ga_sum_rel_lam0 = 0.0
                        self._diag_ga_sum_rel_lam1 = 0.0
                        self._diag_ga_sum_abs_len_lam1 = 0.0
                        self._diag_ga_sum_rel_len_lam1 = 0.0
                        self._diag_ga_count = 0
                        for key in self._diag_ga_pos_keys:
                            self._diag_ga_pos_abs[key] = 0.0
                            self._diag_ga_pos_rel[key] = 0.0
                            self._diag_ga_pos_val_abs_lam1[key] = 0.0
                            self._diag_ga_pos_val_rel_lam1[key] = 0.0
                            self._diag_ga_pos_count[key] = 0.0

        if return_outputs:
            preds_for_metric = value_preds
            combined_labels = torch.stack((value_labels.detach(), value_mask.detach()), dim=-1)
            return value_loss, (preds_for_metric.detach(), combined_labels)

        return value_loss

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()

        if getattr(self.args, "save_safetensors", True):
            from collections import defaultdict

            ptrs = defaultdict(list)
            for name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    ptrs[id(tensor)].append(name)

            for names in ptrs.values():
                if len(names) > 1:
                    names.sort()
                    for name in names[1:]:
                        state_dict.pop(name, None)

        super()._save(output_dir, state_dict)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if isinstance(loss, torch.Tensor):
            loss = loss.mean().detach()

        if prediction_loss_only:
            return loss, None, None

        preds_for_metric, combined_labels = outputs
        preds_for_metric = preds_for_metric.to(torch.float64)
        pred_target = self._raw_to_target_prediction(preds_for_metric)
        len_pred = self._prediction_to_length(pred_target, self.finetuning_args.lenvm_gamma).detach()

        with torch.no_grad():
            value_labels = combined_labels[..., 0].to(torch.float64)
            value_mask = combined_labels[..., 1].to(torch.float64)
            gamma = self.finetuning_args.lenvm_gamma
            diag_value_pred = self._prediction_to_discounted_value(pred_target, gamma)
            sum_l0, sum_l1, sum_rel_l0, sum_rel_l1, sum_abs_len_l1, sum_rel_len_l1, pos_metrics = self._compute_value_diagnostics(
                y_hat=diag_value_pred, value_mask=value_mask, value_labels=value_labels, gamma=gamma,
            )
            cnt = value_mask.sum().clamp_min(torch.tensor(1.0, device=value_mask.device, dtype=value_mask.dtype))
            prefix = getattr(self, "_nontrain_metric_prefix", "predict")
            self._accumulate_nontrain_diag(
                prefix=prefix,
                sum_l0=sum_l0,
                sum_l1=sum_l1,
                sum_rel_l0=sum_rel_l0,
                sum_rel_l1=sum_rel_l1,
                sum_abs_len_l1=sum_abs_len_l1,
                sum_rel_len_l1=sum_rel_len_l1,
                pos_metrics=pos_metrics,
                cnt=cnt,
            )

        return loss, len_pred, combined_labels.detach()

    @override
    def evaluate(
        self,
        eval_dataset: Optional["Dataset"] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        self._nontrain_metric_prefix = metric_key_prefix
        try:
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self._flush_nontrain_logs(metric_key_prefix)
            self._nontrain_metric_prefix = None

    @override
    def predict(
        self,
        test_dataset: "Dataset",
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
    ) -> "PredictionOutput":
        self._nontrain_metric_prefix = metric_key_prefix
        try:
            return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self._flush_nontrain_logs(metric_key_prefix)
            self._nontrain_metric_prefix = None

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save model predictions to `output_dir`.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            preds = torch.as_tensor(predict_results.predictions)
            combined_labels = torch.as_tensor(predict_results.label_ids)
            if combined_labels.ndim != preds.ndim + 1 or combined_labels.shape[-1] != 2:
                raise ValueError("Labels must contain stacked values and masks for length predictions.")

            values = combined_labels[..., 0]
            masks = combined_labels[..., 1]

            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
                values = values.unsqueeze(0)
                masks = masks.unsqueeze(0)

            for pred_seq, value_seq, mask_seq in zip(preds, values, masks):
                pred_flat = pred_seq.reshape(-1)
                value_flat = value_seq.reshape(-1)
                mask_flat = mask_seq.reshape(-1)
                valid = (mask_flat > 0) & torch.isfinite(pred_flat) & torch.isfinite(value_flat)
                pred_vals = [round(float(x), 4) for x in pred_flat[valid]]
                label_vals = [round(float(x), 4) for x in value_flat[valid]]
                writer.write(
                    json.dumps(
                        {"preds": pred_vals, "labels": label_vals, "num_valid": len(pred_vals)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
