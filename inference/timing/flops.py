"""Layer-level inference FLOPs estimator for LenVM vs baseline.

Reads HuggingFace ``config.json`` to count weight matmuls layer-by-layer
instead of relying on the ``2 * N_params`` rule of thumb. The decomposition
matters for LenVM analysis because the value model adds (1 + k) forwards
per generated token, and the relative weight of attention-vs-linear FLOPs
shifts at long sequence lengths.

For each transformer layer with hidden size ``d``, ``H_q`` attention heads,
``H_kv`` key/value heads, head dim ``h``, FFN dim ``ff`` (SwiGLU: gate + up
+ down), per-token FLOPs are:

* attention projections: ``2*d*(H_q*h) + 2*2*d*(H_kv*h) + 2*(H_q*h)*d``
  (Q, K, V, output projections; Qwen2.5 uses GQA so K/V are smaller).
* attention compute at position ``p``: ``2 * H_q * h * p * 2`` (Q@K^T
  and attention@V over a KV cache of length ``p``).
* SwiGLU MLP: ``2 * d * ff * 3`` (gate + up projections share input; down
  projection writes back).
* LM head (final layer only): ``2 * d * V``.

Prefill over a prompt of length ``S`` runs every token in parallel but
each token still attends to the lower-triangular prefix, so attention
FLOPs scale as ``S * (S+1) / 2`` rather than ``S``.

LenVM-guided decoding adds, per output token:

* one ``tree_value_extend`` forward (catch the LenVM cache up by the
  newly-accepted token),
* ``k`` ``tree_value`` forwards (score the top-k candidates).

Prefix caching: when the same prompt is shared by multiple samples (n>1),
the base/LenVM prefill is charged once per unique prompt and decode is
charged per sample. The analyzer assumes
``unique_prompts == meta['max_questions']`` and
``samples == unique_prompts * meta['n_samples_per_q']``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Fallback model dimensions, used only when we cannot locate a config.json
# (e.g. paths that point to a non-existent model). Sourced from the Qwen2.5
# model cards. Update if you add new base/LenVM checkpoints.
_FALLBACK_CONFIGS: Dict[str, Dict[str, int]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": dict(num_hidden_layers=24, hidden_size=896,  num_attention_heads=14, num_key_value_heads=2,  intermediate_size=4864,  vocab_size=151936),
    "Qwen/Qwen2.5-1.5B-Instruct": dict(num_hidden_layers=28, hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,  intermediate_size=8960,  vocab_size=151936),
    "Qwen/Qwen2.5-3B-Instruct":   dict(num_hidden_layers=36, hidden_size=2048, num_attention_heads=16, num_key_value_heads=2,  intermediate_size=11008, vocab_size=151936),
    "Qwen/Qwen2.5-7B-Instruct":   dict(num_hidden_layers=28, hidden_size=3584, num_attention_heads=28, num_key_value_heads=4,  intermediate_size=18944, vocab_size=152064),
    "Qwen/Qwen2.5-14B-Instruct":  dict(num_hidden_layers=48, hidden_size=5120, num_attention_heads=40, num_key_value_heads=8,  intermediate_size=13824, vocab_size=152064),
}


@dataclass(frozen=True)
class ModelConfig:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    # Head type used at the top of the stack. "lm_head" = 2*d*V vocab projection
    # (standard causal LM). "value_head" = small MLP -> scalar (LenVM checkpoints
    # ship a MLP2SiLUValueHead: d->d Linear + d->1 Linear, see
    # sglang/srt/models/qwen2_lvm.py::MLP2SiLUValueHead).
    head_type: str = "lm_head"
    value_head_hidden: Optional[int] = None  # MLP hidden dim if head_type == "value_head"
    value_head_out_dim: int = 1
    source: str = ""

    @classmethod
    def from_dict(cls, cfg: dict, source: str = "", *, head_type: str = "lm_head",
                  value_head_hidden: Optional[int] = None,
                  value_head_out_dim: int = 1) -> "ModelConfig":
        d = cfg["hidden_size"]
        Hq = cfg["num_attention_heads"]
        Hkv = cfg.get("num_key_value_heads", Hq)
        h = cfg.get("head_dim") or (d // Hq)
        return cls(
            num_hidden_layers=cfg["num_hidden_layers"],
            hidden_size=d,
            num_attention_heads=Hq,
            num_key_value_heads=Hkv,
            head_dim=h,
            intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            head_type=head_type,
            value_head_hidden=value_head_hidden if value_head_hidden is not None else d,
            value_head_out_dim=value_head_out_dim,
            source=source,
        )

    @classmethod
    def load(cls, name_or_path: str, *, head_type: str = "auto",
             value_head_out_dim: int = 1) -> "ModelConfig":
        """Locate config.json for a HF model name or a local path.

        head_type="auto" (default) inspects the directory for value_head.safetensors,
        treating its presence as a LenVM-style value-head checkpoint. Pass
        head_type="value_head" or "lm_head" to force the choice.
        """
        cfg_path = _find_config_json(name_or_path)
        if cfg_path is not None:
            cfg_dict = json.loads(cfg_path.read_text())
            ht = head_type if head_type != "auto" else _autodetect_head(cfg_path.parent, cfg_dict)
            return cls.from_dict(
                cfg_dict,
                source=str(cfg_path),
                head_type=ht,
                value_head_out_dim=value_head_out_dim,
            )
        fb = _FALLBACK_CONFIGS.get(name_or_path)
        if fb is None:
            base = name_or_path.rstrip("/").split("/")[-1]
            for key, val in _FALLBACK_CONFIGS.items():
                if key.split("/")[-1] == base:
                    fb = val
                    break
        if fb is None:
            raise FileNotFoundError(
                f"Could not locate config.json for {name_or_path!r} and no fallback "
                f"dimensions are registered. Add one to _FALLBACK_CONFIGS."
            )
        ht = head_type if head_type != "auto" else "lm_head"
        return cls.from_dict(fb, source=f"fallback:{name_or_path}", head_type=ht,
                             value_head_out_dim=value_head_out_dim)


def _autodetect_head(model_dir: Path, cfg_dict: dict) -> str:
    """Return 'value_head' if a value_head.safetensors sits next to config.json,
    or the loaded architecture is a known value-head class. Otherwise 'lm_head'.
    """
    if (model_dir / "value_head.safetensors").exists():
        return "value_head"
    for arch in cfg_dict.get("architectures", []) or []:
        if "LengthValueModel" in arch or "ValueModel" in arch or "ValueHead" in arch:
            return "value_head"
    return "lm_head"


def _find_config_json(name_or_path: str) -> Optional[Path]:
    """Find a HuggingFace-style config.json on disk."""
    p = Path(name_or_path)
    if p.is_dir():
        cfg = p / "config.json"
        if cfg.exists():
            return cfg
    if p.is_file() and p.name == "config.json":
        return p
    # HF cache: $HF_HOME/hub/models--<org>--<name>/snapshots/<sha>/config.json
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    cache_root = Path(hf_home) / "hub"
    if cache_root.exists() and "/" in name_or_path and not name_or_path.startswith("."):
        repo_dir = cache_root / ("models--" + name_or_path.replace("/", "--"))
        snapshots = repo_dir / "snapshots"
        if snapshots.exists():
            for snap in snapshots.iterdir():
                cfg = snap / "config.json"
                if cfg.exists():
                    return cfg
    # Local download dir convention (download_data_and_model.sh writes here):
    if "/" in name_or_path and not name_or_path.startswith("."):
        local_dir = Path("./models") / name_or_path
        cfg = local_dir / "config.json"
        if cfg.exists():
            return cfg
    return None


# ---- Per-token forward FLOPs decomposition ---------------------------------


def per_layer_linear_flops(cfg: ModelConfig) -> int:
    """FLOPs for one token through a single layer's attention projections + MLP."""
    d = cfg.hidden_size
    Hq, Hkv, h = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    ff = cfg.intermediate_size
    q_proj = 2 * d * (Hq * h)
    k_proj = 2 * d * (Hkv * h)
    v_proj = 2 * d * (Hkv * h)
    o_proj = 2 * (Hq * h) * d
    # SwiGLU MLP: gate, up, down. gate and up are both d->ff, down is ff->d.
    mlp = 2 * d * ff + 2 * d * ff + 2 * ff * d
    return q_proj + k_proj + v_proj + o_proj + mlp


def per_layer_attn_compute_flops(cfg: ModelConfig, seq_len: int) -> int:
    """FLOPs for QK^T and attn@V for one token attending to seq_len positions."""
    # 2 * H_q * head_dim * seq_len for each of (QK^T) and (attn @ V)
    return 2 * 2 * cfg.num_attention_heads * cfg.head_dim * seq_len


def lm_head_flops(cfg: ModelConfig) -> int:
    """Vocab projection 2 * hidden * vocab_size."""
    return 2 * cfg.hidden_size * cfg.vocab_size


def value_head_flops(cfg: ModelConfig) -> int:
    """MLP2SiLUValueHead: hidden->hidden (fc) + hidden->out_dim (summary).

    SiLU activation is negligible vs the two matmuls. The summary projection
    is tiny when out_dim=1 (the default) but kept for completeness.
    """
    fc = 2 * cfg.hidden_size * (cfg.value_head_hidden or cfg.hidden_size)
    summary = 2 * (cfg.value_head_hidden or cfg.hidden_size) * cfg.value_head_out_dim
    return fc + summary


def head_flops(cfg: ModelConfig) -> int:
    """FLOPs at the top of the stack, dispatched by ModelConfig.head_type."""
    if cfg.head_type == "value_head":
        return value_head_flops(cfg)
    return lm_head_flops(cfg)


def forward_token_flops(cfg: ModelConfig, position: int) -> Dict[str, int]:
    """Decompose one decode-step FLOPs at the given (1-indexed) position."""
    lin = per_layer_linear_flops(cfg) * cfg.num_hidden_layers
    attn = per_layer_attn_compute_flops(cfg, position) * cfg.num_hidden_layers
    lmh = head_flops(cfg)
    return {"linear": lin, "attention": attn, "lm_head": lmh, "total": lin + attn + lmh}


def prefill_flops(cfg: ModelConfig, prompt_len: int) -> Dict[str, int]:
    """Decompose prefill FLOPs over a prompt of length prompt_len.

    Linear ops scale linearly with prompt_len. Attention is over the
    lower-triangular causal mask, so per-layer attention compute sums to
    2 * H_q * h * (S * (S+1) / 2 * 2) = 2 * H_q * h * S * (S+1).
    """
    if prompt_len <= 0:
        return {"linear": 0, "attention": 0, "lm_head": 0, "total": 0}
    lin = per_layer_linear_flops(cfg) * cfg.num_hidden_layers * prompt_len
    attn_per_layer = 2 * 2 * cfg.num_attention_heads * cfg.head_dim * prompt_len * (prompt_len + 1) // 2
    attn = attn_per_layer * cfg.num_hidden_layers
    lmh = head_flops(cfg) * prompt_len
    return {"linear": lin, "attention": attn, "lm_head": lmh, "total": lin + attn + lmh}


def decode_flops_sum(cfg: ModelConfig, prompt_len: int, output_len: int) -> Dict[str, int]:
    """Sum decode-step FLOPs for output_len tokens, attending to a growing KV cache."""
    if output_len <= 0:
        return {"linear": 0, "attention": 0, "lm_head": 0, "total": 0}
    lin_per_token = per_layer_linear_flops(cfg) * cfg.num_hidden_layers
    lmh_per_token = head_flops(cfg)
    # Closed-form: sum_{t=1..L} (prompt_len + t) = L*prompt_len + L*(L+1)/2
    pos_sum = output_len * prompt_len + output_len * (output_len + 1) // 2
    attn = 2 * 2 * cfg.num_attention_heads * cfg.head_dim * pos_sum * cfg.num_hidden_layers
    lin = lin_per_token * output_len
    lmh = lmh_per_token * output_len
    return {"linear": lin, "attention": attn, "lm_head": lmh, "total": lin + attn + lmh}


def implied_param_count(cfg: ModelConfig) -> int:
    """Approximate parameter count from config (matches 2N rule by 2*N≈linear FLOPs/token)."""
    return per_layer_linear_flops(cfg) * cfg.num_hidden_layers // 2 + lm_head_flops(cfg) // 2


# ---- Run-level aggregation -------------------------------------------------


def baseline_run_flops(
    cfg: ModelConfig,
    *,
    unique_prompts: int,
    samples_per_prompt: int,
    mean_prompt_tokens: float,
    mean_output_tokens: float,
) -> Dict[str, int]:
    """Total baseline FLOPs assuming prefix caching: prefill once per question, decode per sample."""
    pre = prefill_flops(cfg, int(round(mean_prompt_tokens)))
    dec = decode_flops_sum(cfg, int(round(mean_prompt_tokens)), int(round(mean_output_tokens)))
    out = {
        "prefill": {k: v * unique_prompts for k, v in pre.items()},
        "decode":  {k: v * unique_prompts * samples_per_prompt for k, v in dec.items()},
    }
    out["total"] = {k: out["prefill"][k] + out["decode"][k] for k in pre.keys()}
    return out


def lvm_extra_flops(
    cfg: ModelConfig,
    *,
    unique_prompts: int,
    samples_per_prompt: int,
    mean_prompt_tokens: float,
    mean_output_tokens: float,
    k_candidates: int,
    candidate_cost_multiplier: float = 1.0,
) -> Dict[str, int]:
    """LenVM-only extra FLOPs (on top of baseline) per output token:

    * one ``tree_value_extend`` forward (catch the value-model KV up with the
      just-accepted token; a single-token decode at the current position).
    * ``k_candidates`` value-model forwards (score the top-k candidate tokens;
      each is a single-token decode at the position right after the extend).

    ``candidate_cost_multiplier`` is a knob for future LenVM implementations
    that share work across candidates (e.g. batched single-forward scoring
    that amortizes some MLP / attention cost across the k candidates). The
    default of ``1.0`` matches the current sglang-LenVM in-proc path, where
    each candidate is a separate single-token forward sharing only the
    extended KV cache. Set to e.g. ``0.6`` if a future implementation can
    batch the k candidates into one forward.

    Whether ``cfg.head_type`` is ``lm_head`` or ``value_head`` controls
    whether each forward is charged the 2*d*V vocab projection (causal LM)
    or the much smaller MLP2SiLUValueHead (LenVM checkpoints).
    """
    pre = prefill_flops(cfg, int(round(mean_prompt_tokens)))
    dec = decode_flops_sum(cfg, int(round(mean_prompt_tokens)), int(round(mean_output_tokens)))
    candidate_scale = unique_prompts * samples_per_prompt * k_candidates * candidate_cost_multiplier
    out = {
        "lenvm_prefill":    {k: v * unique_prompts for k, v in pre.items()},
        "lenvm_extend":     {k: v * unique_prompts * samples_per_prompt for k, v in dec.items()},
        "lenvm_candidates": {k: int(v * candidate_scale) for k, v in dec.items()},
    }
    out["total"] = {k: out["lenvm_prefill"][k] + out["lenvm_extend"][k] + out["lenvm_candidates"][k] for k in pre.keys()}
    return out
