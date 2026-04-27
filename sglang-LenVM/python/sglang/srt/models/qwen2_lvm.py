# Copyright 2026
#
# Length-Value Model (LVM) wrapper for SGLang Runtime.
#
# This model returns per-candidate length/value predictions (tokenwise value head)
# and is intended to be used with the custom `/tree_value` endpoint.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class MLP2SiLUValueHead(nn.Module):
    """Two-layer value head: Linear(H→H) → SiLU → Linear(H→1).

    This matches the LenVM value head used in LlamaFactory.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        if x.dtype != self.summary.weight.dtype:
            x = x.to(self.summary.weight.dtype)
        x = self.fc(x)
        x = self.act(x)
        x = self.summary(x)
        return x


class Qwen2ForLengthValueModel(Qwen2ForCausalLM):
    """Qwen2 base model + length/value head.

    Forward returns per-request candidate values as an embedding-like output:
      - `EmbeddingPoolerOutput.embeddings` is `List[Tensor]`, each `Tensor` has shape [N_candidates]

    Requires `forward_batch.spec_info` to contain:
      - `tree_value_prefix_lens`: List[int] (prefix length L, per request)
      - `tree_value_candidate_lens`: List[int] (candidate count N, per request)
    """

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Reuse SGLang's native Qwen2ForCausalLM init so weight loading works correctly
        # (pp_group, tie_word_embeddings, etc.).
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.v_head = MLP2SiLUValueHead(config.hidden_size)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "Qwen2ForLengthValueModel is only used for embedding-style outputs"
        spec = getattr(forward_batch, "spec_info", None)

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        token_values = self.v_head(hidden_states).squeeze(-1)  # [num_extend_tokens]

        # If spec_info is provided, return candidate-only values (tree_value).
        prefix_lens = getattr(spec, "tree_value_prefix_lens", None) if spec is not None else None
        cand_lens = getattr(spec, "tree_value_candidate_lens", None) if spec is not None else None
        if prefix_lens is not None and cand_lens is not None:
            extend_lens = forward_batch.extend_seq_lens.tolist()
            cached_prefix_lens = (
                forward_batch.extend_prefix_lens.tolist()
                if forward_batch.extend_prefix_lens is not None
                else [0] * len(extend_lens)
            )

            out: list[torch.Tensor] = []
            offset = 0
            for i, ext_len in enumerate(extend_lens):
                vals_i = token_values[offset : offset + ext_len]
                offset += ext_len

                L = int(prefix_lens[i])
                N = int(cand_lens[i])
                P = int(cached_prefix_lens[i])
                cand_offset = max(L - P, 0)
                out.append(vals_i[cand_offset : cand_offset + N])
            return EmbeddingPoolerOutput(embeddings=out)

        # Otherwise (e.g., /encode), return tokenwise values for the forwarded tokens.
        # This keeps the model usable as a regular embedding model while preserving LVM semantics.
        if forward_batch.extend_seq_lens is None:
            # Fallback: treat whole tensor as one sequence.
            return EmbeddingPoolerOutput(embeddings=[token_values])

        extend_lens = forward_batch.extend_seq_lens.tolist()
        out: list[torch.Tensor] = []
        offset = 0
        for ext_len in extend_lens:
            out.append(token_values[offset : offset + ext_len])
            offset += ext_len
        return EmbeddingPoolerOutput(embeddings=out)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load base weights from the main checkpoint plus v_head weights from value_head.safetensors.

        In this LenVM setup, the value head is stored separately (e.g., value_head.safetensors) and
        is NOT guaranteed to be included in the main HF weight shards. If v_head is not loaded,
        outputs will collapse to an almost-constant value across tokens.
        """

        # NOTE: `weights` can be a generator. We need to materialize it because we
        # consume it multiple times (base weights + v_head).
        weights_list = list(weights)

        # 1) Load base model weights using Qwen2ForCausalLM's loader logic.
        # Filter out v_head tensors since the base loader won't know them.
        #
        # Special case: some checkpoints only provide `lm_head.weight` (tied) but not
        # `model.embed_tokens.weight`. SGLang's native loader skips `lm_head.weight`
        # when `tie_word_embeddings=True` (single-rank), assuming embed_tokens is present.
        # For LenVM checkpoints, we mirror transformers behavior by loading embed_tokens from lm_head.
        names = {name for name, _ in weights_list}
        visual_weight_names = sorted(
            name for name, _ in weights_list if name.startswith("visual.")
        )
        if visual_weight_names:
            raise RuntimeError(
                "Text-only Qwen2 LenVM checkpoint unexpectedly contains visual weights. "
                "This usually means the checkpoint should load through a VLM LenVM architecture "
                f"instead. Unexpected tensors: {visual_weight_names[:8]!r}"
            )
        base_weights = [(name, w) for name, w in weights_list if not name.startswith("v_head.")]
        if (
            getattr(self.config, "tie_word_embeddings", False)
            and "model.embed_tokens.weight" not in names
            and "lm_head.weight" in names
        ):
            lm_head_w = next(w for n, w in weights_list if n == "lm_head.weight")
            base_weights.append(("model.embed_tokens.weight", lm_head_w))
        super().load_weights(base_weights)

        # 2) Load value head weights.
        # Prefer v_head.* tensors if present in the incoming iterator; otherwise read from value_head.safetensors.
        v_state = {
            name[len("v_head.") :]: w
            for name, w in weights_list
            if name.startswith("v_head.")
        }
        if not v_state:
            try:
                from safetensors.torch import safe_open

                # Prefer the actual model path associated with this config (works for in-proc
                # multi-model setups where global server_args.model_path points to the decode model).
                model_path = (
                    getattr(self.config, "_name_or_path", None)
                    or getattr(self.config, "name_or_path", None)
                    or get_global_server_args().model_path
                )
                vh_path = f"{model_path}/value_head.safetensors"
                with safe_open(vh_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.startswith("v_head."):
                            v_state[k[len("v_head.") :]] = f.get_tensor(k)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load LenVM value head weights. Expected `v_head.*` in checkpoint "
                    "or a separate `value_head.safetensors` under the model directory."
                ) from e

        missing, unexpected = self.v_head.load_state_dict(v_state, strict=False)
        if missing:
            raise RuntimeError(f"Missing v_head weights: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected LenVM value head weights: {unexpected}")

        return


EntryClass = [
    Qwen2ForLengthValueModel,
]

