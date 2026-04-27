# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from transformers.utils import cached_file

from ...extras import logging
from ...extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


class MultiModalValueHeadModel(nn.Module):
    r"""A lightweight value-head wrapper compatible with image-text models.

    TRL's ``AutoModelForCausalLMWithValueHead`` is built around text-only causal
    LM assumptions. Some multimodal models such as Qwen2.5-VL load through
    ``AutoModelForImageTextToText`` and do not consistently follow that API.
    This wrapper keeps the same high-level contract used by RM/LenVM trainers:
    the wrapped model is exposed as ``pretrained_model`` and ``forward`` returns
    ``(logits, loss, values)`` with the value head applied on the last hidden
    states.
    """

    def __init__(self, pretrained_model: "PreTrainedModel", summary_dropout_prob: Optional[float] = None) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.generation_config = pretrained_model.generation_config

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.config, "text_config", None)
            hidden_size = getattr(hidden_size, "hidden_size", None)

        if hidden_size is None:
            raise ValueError("Cannot infer hidden size for multimodal value head.")

        if summary_dropout_prob is None:
            summary_dropout_prob = getattr(self.config, "summary_dropout_prob", 0.0)

        self.v_head = nn.Sequential()
        self.v_head.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        self.v_head.summary = nn.Linear(hidden_size, 1)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            pretrained_model = self.__dict__.get("pretrained_model")
            if pretrained_model is not None and hasattr(pretrained_model, name):
                return getattr(pretrained_model, name)
            raise exc

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs.setdefault("use_cache", False)

        outputs = self.pretrained_model(*args, **kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Multimodal value-head wrapper requires hidden_states from the base model.")

        last_hidden_state = hidden_states[-1]
        value_input = self.v_head.dropout(last_hidden_state)
        if value_input.dtype != self.v_head.summary.weight.dtype:
            value_input = value_input.to(self.v_head.summary.weight.dtype)

        values = self.v_head.summary(value_input).squeeze(-1)
        logits = getattr(outputs, "logits", None)
        loss = getattr(outputs, "loss", None)
        return logits, loss, values


def load_valuehead_params(path_or_repo_id: str, model_args: "ModelArguments") -> dict[str, torch.Tensor]:
    r"""Load value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = cached_file(filename=V_HEAD_SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = cached_file(filename=V_HEAD_WEIGHTS_NAME, **kwargs)
        return torch.load(vhead_file, map_location="cpu", weights_only=True)
    except Exception as err:
        err_text = str(err)

    logger.info_rank0(f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}.")
    logger.info_rank0("Ignore the above message if you are not resuming the training of a value head model.")
    return None


def prepare_valuehead_model(model: "PreTrainedModel") -> None:
    if getattr(model.config, "model_type", None) == "llava":
        setattr(model, "lm_head", model.language_model.get_output_embeddings())
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if getattr(model.config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if getattr(model.config, "model_type", None) == "internlm2":
        setattr(model, "lm_head", model.output)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])
