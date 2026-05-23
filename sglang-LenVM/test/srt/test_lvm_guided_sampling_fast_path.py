from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _load_lvm_guided_sampling(monkeypatch):
    """Load lvm_guided_sampling.py with light stubs for heavyweight SGLang deps."""

    stubs = {
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.sampling": _module("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": _module(
            "sglang.srt.sampling.sampling_params", TOP_K_ALL=-1
        ),
        "sglang.srt.utils": _module("sglang.srt.utils"),
        "sglang.srt.utils.common": _module(
            "sglang.srt.utils.common", dynamic_import=lambda spec: None
        ),
        "sglang.srt.server_args": _module(
            "sglang.srt.server_args", get_global_server_args=lambda: SimpleNamespace()
        ),
        "sglang.srt.lvm": _module("sglang.srt.lvm"),
        "sglang.srt.lvm.lvm_value_utils": _module(
            "sglang.srt.lvm.lvm_value_utils",
            force_eos_value_zero=lambda token_ids, token_values, req: None,
            get_eos_token_ids=lambda req: set(),
        ),
        "sglang.srt.configs": _module("sglang.srt.configs"),
        "sglang.srt.configs.model_config": _module(
            "sglang.srt.configs.model_config", ModelConfig=object
        ),
        "sglang.srt.managers": _module("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": _module(
            "sglang.srt.managers.schedule_batch", Req=object
        ),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    source = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "lvm"
        / "lvm_guided_sampling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lvm_guided_sampling_under_test", source
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def _req(rid: str, custom_params: dict | None = None):
    if custom_params is None:
        custom_params = {"value_scale": -100.0, "value_mode": "centered_exp"}
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[11, 12],
        output_ids=[21],
        sampling_params=SimpleNamespace(custom_params=custom_params),
    )


def test_gpu_fast_path_keeps_candidate_ids_on_device_with_top_p(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )
    sampler._fn = mod.lvm_combined_guidance

    probs = torch.tensor(
        [
            [0.40, 0.25, 0.20, 0.10, 0.05, 0.0, 0.0, 0.0],
            [0.35, 0.25, 0.20, 0.11, 0.09, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pending = sampler._build_pending(
        probs=probs,
        reqs=[_req("r0"), _req("r1")],
        temperatures=torch.ones(2),
        top_ps=torch.tensor([0.75, 0.80], dtype=torch.float32),
        top_ks=torch.tensor([5, 5], dtype=torch.int64),
        min_ps=torch.zeros(2),
        enable_gpu_candidate_compact=True,
    )

    assert pending is not None
    assert pending.gpu_candidates is not None
    assert pending.candidate_ids_send == []
    assert pending.candidate_probs_send == []
    assert pending.candidate_lens_send == [3, 4]
    assert pending.prefix_ids_send == [[11, 12, 21], [11, 12, 21]]

    padded_probs, padded_ids, valid_mask = pending.gpu_candidates
    assert valid_mask.sum(dim=1).tolist() == pending.candidate_lens_send
    assert padded_ids[valid_mask].tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert torch.all(padded_probs[valid_mask] > 0)


def test_default_path_keeps_cpu_candidate_lists_for_fallback(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )
    sampler._fn = mod.lvm_combined_guidance

    probs = torch.tensor(
        [
            [0.40, 0.25, 0.20, 0.10, 0.05, 0.0, 0.0, 0.0],
            [0.35, 0.25, 0.20, 0.11, 0.09, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pending = sampler._build_pending(
        probs=probs,
        reqs=[_req("r0"), _req("r1")],
        temperatures=torch.ones(2),
        top_ps=torch.tensor([0.75, 0.80], dtype=torch.float32),
        top_ks=torch.tensor([5, 5], dtype=torch.int64),
        min_ps=torch.zeros(2),
    )

    assert pending is not None
    assert pending.gpu_candidates is None
    assert pending.candidate_lens_send is None
    assert pending.candidate_ids_send == [[0, 1, 2], [0, 1, 2, 3]]
    assert [len(p) for p in pending.candidate_probs_send] == [3, 4]


def test_unmodified_rows_keep_sampling_filters_when_guidance_applies(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )
    sampler._fn = mod.lvm_combined_guidance

    probs = torch.tensor(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.40, 0.30, 0.20],
        ],
        dtype=torch.float32,
    )
    pending = sampler._build_pending(
        probs=probs,
        reqs=[_req("r0"), _req("r1", {})],
        temperatures=torch.ones(2),
        top_ps=torch.ones(2),
        top_ks=torch.tensor([2, 2], dtype=torch.int64),
        min_ps=torch.zeros(2),
    )

    assert pending is not None
    assert pending.candidate_ids_send == [[0, 1]]
    assert torch.allclose(
        pending.guided[1],
        torch.tensor([0.0, 0.40, 0.30, 0.0], dtype=torch.float32),
    )


def test_neutral_centered_exp_scale_skips_lvm(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )
    sampler._fn = mod.lvm_combined_guidance

    probs = torch.tensor([[0.40, 0.25, 0.20, 0.10, 0.05]], dtype=torch.float32)
    pending = sampler._build_pending(
        probs=probs,
        reqs=[_req("r0", {"value_scale": 0.0, "value_mode": "centered_exp"})],
        temperatures=torch.ones(1),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([5], dtype=torch.int64),
        min_ps=torch.zeros(1),
        enable_gpu_candidate_compact=True,
    )

    assert pending is None


def test_neutral_mul_scale_zero_skips_lvm(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )
    sampler._fn = mod.lvm_combined_guidance

    probs = torch.tensor([[0.40, 0.25, 0.20, 0.10, 0.05]], dtype=torch.float32)
    pending = sampler._build_pending(
        probs=probs,
        reqs=[_req("r0", {"value_scale": 0.0, "value_mode": "mul"})],
        temperatures=torch.ones(1),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([5], dtype=torch.int64),
        min_ps=torch.zeros(1),
        enable_gpu_candidate_compact=True,
    )

    assert pending is None


def test_apply_without_guidance_does_not_initialize_inproc(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )

    def fail_if_called():
        raise AssertionError("in-proc provider should not initialize for baseline rows")

    sampler._get_inproc_provider = fail_if_called

    probs = torch.tensor([[0.40, 0.25, 0.20, 0.10, 0.05]], dtype=torch.float32)
    out = sampler.apply(
        probs=probs,
        reqs=[_req("r0", {})],
        temperatures=torch.ones(1),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([5], dtype=torch.int64),
        min_ps=torch.zeros(1),
    )

    assert out is None


def test_gpu_guidance_handles_mixed_modes(monkeypatch):
    mod = _load_lvm_guided_sampling(monkeypatch)
    sampler = mod.LvmGuidedSampler(
        mod.LvmGuidedConfig(url=None, timeout=1.0, bypass_cache=False, fn_spec=None)
    )

    padded_probs = torch.tensor(
        [[0.60, 0.40], [0.55, 0.45]], dtype=torch.float32
    )
    padded_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    valid_mask = torch.ones_like(padded_probs, dtype=torch.bool)
    values = torch.tensor([[0.20, 0.80], [0.10, 0.90]], dtype=torch.float32)
    gpu_embeddings = torch.logit(values)

    pending = mod.PendingLvmResult(
        req_list=[
            _req("r0", {"value_scale": 1.0, "value_mode": "centered_exp"}),
            _req("r1", {"value_scale": 0.0, "value_mode": "mul"}),
        ],
        device=torch.device("cpu"),
        guided=torch.zeros((2, 4), dtype=torch.float32),
        send_batch_indices=[0, 1],
        prefix_ids_send=[],
        candidate_ids_send=[],
        candidate_probs_send=[],
        gpu_candidates=(padded_probs, padded_ids, valid_mask),
    )

    sampler._apply_guidance_gpu(pending, gpu_embeddings)

    expected_row0 = padded_probs[0] * torch.exp(values[0])
    expected_row0 = expected_row0 / expected_row0.sum()
    assert torch.allclose(pending.guided[0, :2], expected_row0, atol=1e-6)
    assert torch.allclose(pending.guided[1, :2], padded_probs[1], atol=1e-6)
