# Copyright 2026
#
# Length-Value Model (LenVM) wrapper for Qwen2.5-VL.

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_lvm import MLP2SiLUValueHead
from sglang.srt.server_args import get_global_server_args


class Qwen2_5_VLForLengthValueModel(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL backbone with a LenVM value head."""

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.v_head = MLP2SiLUValueHead(config.hidden_size)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = True,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        assert get_embedding, (
            "Qwen2_5_VLForLengthValueModel is only used for embedding-style outputs"
        )

        if self.is_mrope_enabled and forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        else:
            aux_hidden_states = None

        if not self.pp_group.is_last_rank:
            return hidden_states

        token_values = self.v_head(hidden_states).squeeze(-1)

        spec = getattr(forward_batch, "spec_info", None)
        prefix_lens = (
            getattr(spec, "tree_value_prefix_lens", None) if spec is not None else None
        )
        cand_lens = (
            getattr(spec, "tree_value_candidate_lens", None) if spec is not None else None
        )

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

                prefix_len = int(prefix_lens[i])
                cand_len = int(cand_lens[i])
                cached_prefix_len = int(cached_prefix_lens[i])
                cand_offset = max(prefix_len - cached_prefix_len, 0)
                out.append(vals_i[cand_offset : cand_offset + cand_len])
            return EmbeddingPoolerOutput(embeddings=out)

        if forward_batch.extend_seq_lens is None:
            return EmbeddingPoolerOutput(embeddings=[token_values])

        extend_lens = forward_batch.extend_seq_lens.tolist()
        out: list[torch.Tensor] = []
        offset = 0
        for ext_len in extend_lens:
            out.append(token_values[offset : offset + ext_len])
            offset += ext_len
        return EmbeddingPoolerOutput(embeddings=out)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights_list = list(weights)

        base_weights = [(n, w) for n, w in weights_list if not n.startswith("v_head.")]
        super().load_weights(base_weights)

        v_state = {
            n[len("v_head.") :]: w for n, w in weights_list if n.startswith("v_head.")
        }
        if not v_state:
            try:
                from safetensors.torch import safe_open

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


EntryClass = [Qwen2_5_VLForLengthValueModel]
