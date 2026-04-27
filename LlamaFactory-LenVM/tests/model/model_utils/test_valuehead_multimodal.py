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

import importlib
import sys
import types
from types import SimpleNamespace

import torch

from llamafactory.model.model_utils.valuehead import MultiModalValueHeadModel


class DummyOutput(SimpleNamespace):
    pass


class DummyImageTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, model_type="qwen2_5_vl")
        self.generation_config = SimpleNamespace()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        batch_size, seq_len = kwargs["input_ids"].shape
        hidden = torch.arange(batch_size * seq_len * 4, dtype=torch.float32).reshape(batch_size, seq_len, 4)
        logits = torch.zeros(batch_size, seq_len, 8, dtype=torch.float32)
        return DummyOutput(logits=logits, loss=torch.tensor(0.0), hidden_states=(hidden, hidden + 1.0))


def test_multimodal_valuehead_forward_preserves_mm_inputs():
    base_model = DummyImageTextModel()
    model = MultiModalValueHeadModel(base_model, summary_dropout_prob=0.0)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    pixel_values = torch.randn(2, 3, 2, 2)
    image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]])

    logits, loss, values = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    assert logits.shape == (2, 3, 8)
    assert loss.item() == 0.0
    assert values.shape == (2, 3)
    assert torch.equal(base_model.last_kwargs["pixel_values"], pixel_values)
    assert torch.equal(base_model.last_kwargs["image_grid_thw"], image_grid_thw)
    assert base_model.last_kwargs["output_hidden_states"] is True
    assert base_model.last_kwargs["return_dict"] is True
    assert base_model.last_kwargs["use_cache"] is False


def test_loader_uses_multimodal_valuehead_wrapper(monkeypatch):
    sys.modules.setdefault("trl", types.SimpleNamespace(AutoModelForCausalLMWithValueHead=object()))
    loader = importlib.import_module("llamafactory.model.loader")

    class DummyConfig:
        model_type = "qwen2_5_vl"

    class DummyAutoModelForImageTextToText:
        _model_mapping = {DummyConfig: object()}

        @staticmethod
        def from_pretrained(**kwargs):
            return DummyImageTextModel()

        @staticmethod
        def from_config(config, trust_remote_code=False):
            return DummyImageTextModel()

    class FailingTRLValueHead:
        @staticmethod
        def from_pretrained(model, **kwargs):
            raise AssertionError("text-only TRL wrapper should not be used for image-text models")

    monkeypatch.setattr(loader, "_get_init_kwargs", lambda model_args: {})
    monkeypatch.setattr(loader, "load_config", lambda model_args: DummyConfig())
    monkeypatch.setattr(loader, "patch_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "apply_liger_kernel", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "patch_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "register_autoclass", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "init_adapter", lambda config, model, *args, **kwargs: model)
    monkeypatch.setattr(loader, "patch_valuehead_model", lambda model: None)
    monkeypatch.setattr(loader, "load_valuehead_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "count_parameters", lambda model: (0, 1))
    monkeypatch.setattr(loader, "AutoModelForImageTextToText", DummyAutoModelForImageTextToText)
    monkeypatch.setattr(loader, "AutoModelForCausalLMWithValueHead", FailingTRLValueHead)

    model_args = SimpleNamespace(
        model_name_or_path="dummy-qwen2.5-vl",
        adapter_name_or_path=None,
        cache_dir=None,
        hf_hub_token=None,
        trust_remote_code=False,
        mixture_of_depths=None,
        train_from_scratch=False,
        valuehead_dropout=0.0,
        use_kt=False,
        use_unsloth=False,
        use_v1_kernels=False,
        print_param_status=False,
    )
    finetuning_args = SimpleNamespace(stage="lenvm")

    model = loader.load_model(tokenizer=object(), model_args=model_args, finetuning_args=finetuning_args, add_valuehead=True)

    assert isinstance(model, MultiModalValueHeadModel)
