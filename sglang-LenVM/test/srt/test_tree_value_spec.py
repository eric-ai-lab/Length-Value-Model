from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_tree_value_spec(monkeypatch):
    spec_info = types.ModuleType("sglang.srt.speculative.spec_info")

    class SpecInput:
        def __init__(self, spec_input_type):
            self.spec_input_type = spec_input_type

    class SpecInputType:
        EAGLE_VERIFY = 1

    spec_info.SpecInput = SpecInput
    spec_info.SpecInputType = SpecInputType

    attention_utils = types.ModuleType("sglang.srt.layers.attention.utils")

    class FakeCreateKvIndicesKernel:
        def __getitem__(self, _grid):
            def launch(
                req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                _kv_start_idx,
                kv_indices,
                _stride,
            ):
                for i in range(req_pool_indices.numel()):
                    start = int(kv_indptr[i].item())
                    end = int(kv_indptr[i + 1].item())
                    pool_idx = int(req_pool_indices[i].item())
                    kv_indices[start:end] = req_to_token[
                        pool_idx, : int(paged_kernel_lens[i].item())
                    ].to(kv_indices.dtype)

            return launch

    attention_utils.create_flashinfer_kv_indices_triton = FakeCreateKvIndicesKernel()

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.speculative",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.spec_info", spec_info)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.utils", attention_utils
    )

    source = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "lvm"
        / "tree_value_spec.py"
    )
    spec = importlib.util.spec_from_file_location("_tree_value_spec_under_test", source)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_fused_prefix_and_candidate_mask(monkeypatch):
    mod = _load_tree_value_spec(monkeypatch)

    custom_mask, positions = mod.build_tree_value_custom_mask_and_positions(
        prefix_lens=[4],
        candidate_lens=[3],
        cached_prefix_lens=[3],
        device=torch.device("cpu"),
    )

    assert positions.tolist() == [3, 4, 4, 4]
    assert custom_mask.view(4, 7).int().tolist() == [
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1],
    ]


def test_tree_value_spec_generates_flashinfer_args(monkeypatch):
    mod = _load_tree_value_spec(monkeypatch)
    custom_mask, positions = mod.build_tree_value_custom_mask_and_positions(
        prefix_lens=[4],
        candidate_lens=[3],
        cached_prefix_lens=[3],
        device=torch.device("cpu"),
    )
    spec_info = mod.TreeValueSpecInput(
        custom_mask=custom_mask,
        positions=positions,
        tree_value_prefix_lens=[4],
        tree_value_candidate_lens=[3],
        tree_value_cached_prefix_lens=[3],
    )

    req_to_token = torch.arange(20, dtype=torch.int64).view(2, 10)
    kv_indices, kv_indptr, qo_indptr, returned_mask = (
        spec_info.generate_attn_arg_prefill(
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            paged_kernel_lens=torch.tensor([7], dtype=torch.int32),
            paged_kernel_lens_sum=7,
            req_to_token=req_to_token,
        )
    )

    assert qo_indptr.tolist() == [0, 4]
    assert kv_indptr.tolist() == [0, 7]
    assert kv_indices.tolist() == [10, 11, 12, 13, 14, 15, 16]
    assert returned_mask is custom_mask
