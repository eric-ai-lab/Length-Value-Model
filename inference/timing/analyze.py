"""Compare baseline vs LenVM-guided timing runs produced by lenvm_timing.sh.

Inputs (from --results-dir):
  baseline.timing.jsonl   per-step records from the no-LenVM server
  lenvm.timing.jsonl      per-step records from the LenVM-enabled server
  baseline.meta.json      end-to-end wall clock + token counts (run_timing.py)
  lenvm.meta.json         same, for LenVM run

Outputs (in --results-dir):
  summary.csv             one row per setting with e2e + per-step aggregates
  summary.json            same as CSV but JSON
  per_step_breakdown.png  stacked bar of per-step decomposition

The per-step decomposition is what reviewers asked for: how much of a decoding
step is base-model forward (inferred from "step not under sampler"), how much
is sampler-side preprocess, how much is the LenVM forward + guidance overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from inference.timing.flops import (
    ModelConfig,
    baseline_run_flops,
    lvm_extra_flops,
)


_PER_STEP_KEYS = [
    "t_sampler_total_ms",
    "t_pre_lvm_ms",
    "t_lvm_apply_outer_ms",
    "t_lvm_build_pending_ms",
    "t_lvm_forward_ms",
    "t_lvm_apply_guidance_ms",
    "t_sample_ms",
]


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def _filter_warmup(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the small-batch greedy warmup steps SGLang emits at server start
    (POST /generate handshake with a single token, bs=1, is_greedy=True)."""
    return [r for r in records if not r.get("is_greedy", False)]


def _agg(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-step records into mean/p50/p95/sum per key."""
    out: Dict[str, Any] = {"n_steps": len(records)}
    for key in _PER_STEP_KEYS:
        vals: List[float] = []
        for r in records:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            continue
        out[f"{key}_mean"] = sum(vals) / len(vals)
        out[f"{key}_p50"] = _percentile(vals, 0.5)
        out[f"{key}_p95"] = _percentile(vals, 0.95)
        out[f"{key}_sum"] = sum(vals)
        out[f"{key}_count"] = len(vals)
    batch_sizes = [int(r["batch_size"]) for r in records if isinstance(r.get("batch_size"), int)]
    if batch_sizes:
        out["batch_size_mean"] = sum(batch_sizes) / len(batch_sizes)
        out["batch_size_p50"] = _percentile(batch_sizes, 0.5)
        out["batch_size_p95"] = _percentile(batch_sizes, 0.95)
    lvm_active = sum(1 for r in records if r.get("lvm_active"))
    out["n_steps_with_lvm"] = lvm_active
    return out


def _load_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _row_for(tag: str, meta: Dict[str, Any], agg: Dict[str, Any]) -> Dict[str, Any]:
    summary = meta.get("summary") or {}
    row: Dict[str, Any] = {
        "tag": tag,
        "wall_clock_s": meta.get("wall_clock_s"),
        "total_output_tokens": summary.get("total_output_tokens"),
        "total_prompt_tokens": summary.get("total_prompt_tokens"),
        "throughput_output_tokens_per_s": meta.get("throughput_output_tokens_per_s"),
        "n_requests": summary.get("n_requests"),
        "output_tokens_mean": summary.get("output_tokens_mean"),
        "output_tokens_p95": summary.get("output_tokens_p95"),
        "latency_s_mean": summary.get("latency_s_mean"),
        "latency_s_p95": summary.get("latency_s_p95"),
        "value_scale": meta.get("value_scale"),
        "top_k": meta.get("top_k"),
    }
    row.update(agg)
    return row


def _add_flops(
    row: Dict[str, Any],
    *,
    meta: Dict[str, Any],
    base_cfg: Optional[ModelConfig],
    lvm_cfg: Optional[ModelConfig],
    is_lvm: bool,
) -> None:
    """Attach layer-level theoretical FLOPs + achieved-FLOPs/sec columns.

    Splits prefill (charged once per unique prompt, assumes SGLang prefix cache)
    from decode (charged per sample), and breaks each into linear / attention /
    lm_head components.
    """
    if base_cfg is None:
        return
    unique_prompts = int(meta.get("max_questions") or 0)
    samples_per = int(meta.get("n_samples_per_q") or 1)
    prompt_total = int(row.get("total_prompt_tokens") or 0)
    output_total = int(row.get("total_output_tokens") or 0)
    if unique_prompts <= 0 or output_total <= 0:
        return
    # total_prompt_tokens is summed once per unique question (dedup in summarize).
    # total_output_tokens is summed across all samples per question, then summed across questions.
    mean_prompt = prompt_total / unique_prompts
    mean_output = output_total / (unique_prompts * samples_per)

    base = baseline_run_flops(
        base_cfg,
        unique_prompts=unique_prompts,
        samples_per_prompt=samples_per,
        mean_prompt_tokens=mean_prompt,
        mean_output_tokens=mean_output,
    )
    total = base["total"]["total"]
    row["base_pflops_total"] = total / 1e15
    row["base_pflops_prefill"] = base["prefill"]["total"] / 1e15
    row["base_pflops_decode"] = base["decode"]["total"] / 1e15
    row["base_pflops_linear"] = base["total"]["linear"] / 1e15
    row["base_pflops_attention"] = base["total"]["attention"] / 1e15
    row["base_pflops_lm_head"] = base["total"]["lm_head"] / 1e15

    if is_lvm and lvm_cfg is not None:
        k = row.get("top_k")
        if k is not None and int(k) >= 1:
            extra = lvm_extra_flops(
                lvm_cfg,
                unique_prompts=unique_prompts,
                samples_per_prompt=samples_per,
                mean_prompt_tokens=mean_prompt,
                mean_output_tokens=mean_output,
                k_candidates=int(k),
            )
            total += extra["total"]["total"]
            row["lvm_pflops_prefill"] = extra["lenvm_prefill"]["total"] / 1e15
            row["lvm_pflops_extend"] = extra["lenvm_extend"]["total"] / 1e15
            row["lvm_pflops_candidates"] = extra["lenvm_candidates"]["total"] / 1e15
            row["lvm_pflops_total"] = extra["total"]["total"] / 1e15

    row["theoretical_pflops_total"] = total / 1e15
    # Per-decode-token cost (excluding prefill share) for a quick "GFLOPs/tok" feel
    decode_total = base["decode"]["total"] + (row.get("lvm_pflops_extend", 0.0) + row.get("lvm_pflops_candidates", 0.0)) * 1e15
    n_decode_tokens = unique_prompts * samples_per * mean_output
    if n_decode_tokens > 0:
        row["theoretical_gflops_per_output_token"] = decode_total / n_decode_tokens / 1e9

    wall = row.get("wall_clock_s") or 0
    if wall > 0:
        row["achieved_tflops_per_s"] = total / wall / 1e12


def _ratio_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute lenvm / baseline ratios for numeric columns. Assumes rows[0]=baseline, rows[1]=lenvm."""
    if len(rows) != 2:
        return {}
    base, lvm = rows
    ratio: Dict[str, Any] = {"tag": "ratio (lvm/base)"}
    for key, b_val in base.items():
        l_val = lvm.get(key)
        if isinstance(b_val, (int, float)) and isinstance(l_val, (int, float)) and b_val:
            ratio[key] = l_val / b_val
    return ratio


def _stacked_bar(rows: List[Dict[str, Any]], out_png: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable, skipping plot: {e}")
        return None

    labels = [r["tag"] for r in rows]
    sections = ["t_pre_lvm_ms_mean", "t_lvm_apply_outer_ms_mean", "t_sample_ms_mean"]
    legend_names = ["pre-LVM (preprocess+softmax)", "LenVM apply (forward+guidance)", "sample kernel"]
    values_per_section = [[float(r.get(s, 0.0) or 0.0) for r in rows] for s in sections]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = [0.0] * len(labels)
    for sec_vals, name in zip(values_per_section, legend_names):
        ax.bar(labels, sec_vals, bottom=bottoms, label=name)
        bottoms = [b + v for b, v in zip(bottoms, sec_vals)]

    for i, r in enumerate(rows):
        total = bottoms[i]
        ax.text(i, total, f"{total:.2f} ms", ha="center", va="bottom")

    ax.set_ylabel("Mean per-step latency inside Sampler.forward (ms)")
    ax.set_title("LenVM vs baseline: sampler-side per-step decomposition")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def _flops_component_bar(rows: List[Dict[str, Any]], out_png: Path) -> Optional[Path]:
    """Stacked bar of theoretical PFLOPs by component (base linear/attn/lm_head + LenVM)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable, skipping plot: {e}")
        return None

    if not any(r.get("base_pflops_linear") for r in rows):
        return None

    labels = [r["tag"] for r in rows]
    sections = [
        ("base_pflops_linear",     "base linear (Q/K/V/O + MLP)"),
        ("base_pflops_attention",  "base attention (QK^T + attn@V)"),
        ("base_pflops_lm_head",    "base lm_head"),
        ("lvm_pflops_extend",      "LenVM extend forward"),
        ("lvm_pflops_candidates",  "LenVM candidate forwards"),
        ("lvm_pflops_prefill",     "LenVM prefill"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * len(labels)
    for key, name in sections:
        vals = [float(r.get(key) or 0.0) for r in rows]
        if max(vals) <= 0:
            continue
        ax.bar(labels, vals, bottom=bottoms, label=name)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, total in enumerate(bottoms):
        ax.text(i, total, f"{total:.2f} PFLOPs", ha="center", va="bottom")
    ax.set_ylabel("Theoretical FLOPs (PFLOPs)")
    ax.set_title("Theoretical inference FLOPs by component")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def _lvm_sub_breakdown(rows: List[Dict[str, Any]], out_png: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable, skipping plot: {e}")
        return None

    lenvm_row = next((r for r in rows if r.get("n_steps_with_lvm", 0) > 0), None)
    if lenvm_row is None:
        return None

    sub_keys = ["t_lvm_build_pending_ms_mean", "t_lvm_forward_ms_mean", "t_lvm_apply_guidance_ms_mean"]
    sub_names = ["build_pending", "LenVM forward (extend+launch+collect)", "apply_guidance"]
    sub_vals = [float(lenvm_row.get(k, 0.0) or 0.0) for k in sub_keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sub_names, sub_vals)
    for i, v in enumerate(sub_vals):
        ax.text(i, v, f"{v:.2f} ms", ha="center", va="bottom")
    ax.set_ylabel("Mean per-step (ms)")
    ax.set_title("LenVM apply() internal breakdown")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def _print_tables(rows: List[Dict[str, Any]]) -> None:
    timing_cols = [
        ("tag", "tag"),
        ("wall_clock_s", "e2e_s"),
        ("throughput_output_tokens_per_s", "tok/s"),
        ("output_tokens_mean", "out_tok_mean"),
        ("n_steps", "n_steps"),
        ("t_sampler_total_ms_mean", "samp_total_ms"),
        ("t_pre_lvm_ms_mean", "pre_ms"),
        ("t_lvm_apply_outer_ms_mean", "lvm_apply_ms"),
        ("t_sample_ms_mean", "sample_ms"),
    ]
    flops_total_cols = [
        ("tag", "tag"),
        ("theoretical_gflops_per_output_token", "GFLOPs/tok"),
        ("base_pflops_total", "base PFLOPs"),
        ("lvm_pflops_total", "lvm PFLOPs"),
        ("theoretical_pflops_total", "total PFLOPs"),
        ("achieved_tflops_per_s", "TFLOPs/s"),
        ("wall_clock_s", "e2e_s"),
    ]
    flops_component_cols = [
        ("tag", "tag"),
        ("base_pflops_linear", "base.linear"),
        ("base_pflops_attention", "base.attn"),
        ("base_pflops_lm_head", "base.lm_head"),
        ("base_pflops_prefill", "base.prefill"),
        ("base_pflops_decode", "base.decode"),
        ("lvm_pflops_prefill", "lvm.prefill"),
        ("lvm_pflops_extend", "lvm.extend"),
        ("lvm_pflops_candidates", "lvm.cands"),
    ]
    _print_one(rows, timing_cols)
    if any("theoretical_pflops_total" in r for r in rows):
        print()
        print("== FLOPs headline ==")
        _print_one(rows, flops_total_cols)
        print()
        print("== FLOPs by component (PFLOPs) ==")
        _print_one(rows, flops_component_cols)


def _print_one(rows: List[Dict[str, Any]], cols: List[tuple]) -> None:
    widths = {k: max(len(label), max(len(_fmt(r.get(k))) for r in rows)) for k, label in cols}
    header = " | ".join(f"{label:>{widths[k]}}" for k, label in cols)
    print(header)
    print("-+-".join("-" * widths[k] for k, _ in cols))
    for r in rows:
        print(" | ".join(f"{_fmt(r.get(k)):>{widths[k]}}" for k, _ in cols))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--baseline-tag", default="baseline")
    p.add_argument("--lenvm-tag", default="lenvm")
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base generation model name (for theoretical FLOPs lookup).",
    )
    p.add_argument(
        "--lvm-model",
        default="namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct",
        help="LenVM checkpoint name (for theoretical FLOPs lookup).",
    )
    args = p.parse_args()

    base_cfg: Optional[ModelConfig] = None
    lvm_cfg: Optional[ModelConfig] = None
    try:
        base_cfg = ModelConfig.load(args.base_model)
    except FileNotFoundError as e:
        print(f"warning: {e}; skipping baseline FLOPs")
    try:
        lvm_cfg = ModelConfig.load(args.lvm_model)
    except FileNotFoundError as e:
        print(f"warning: {e}; skipping LenVM FLOPs")
    if base_cfg is not None:
        print(f"base config (source={base_cfg.source}): L={base_cfg.num_hidden_layers} d={base_cfg.hidden_size} "
              f"Hq={base_cfg.num_attention_heads} Hkv={base_cfg.num_key_value_heads} h={base_cfg.head_dim} "
              f"ff={base_cfg.intermediate_size} V={base_cfg.vocab_size} head={base_cfg.head_type}")
    if lvm_cfg is not None:
        print(f"lvm config  (source={lvm_cfg.source}): L={lvm_cfg.num_hidden_layers} d={lvm_cfg.hidden_size} "
              f"Hq={lvm_cfg.num_attention_heads} Hkv={lvm_cfg.num_key_value_heads} h={lvm_cfg.head_dim} "
              f"ff={lvm_cfg.intermediate_size} V={lvm_cfg.vocab_size} head={lvm_cfg.head_type}")

    rd = args.results_dir
    rows: List[Dict[str, Any]] = []
    for tag, is_lvm in ((args.baseline_tag, False), (args.lenvm_tag, True)):
        meta = _load_meta(rd / f"{tag}.meta.json")
        records = _filter_warmup(list(_iter_records(rd / f"{tag}.timing.jsonl")))
        agg = _agg(records)
        row = _row_for(tag, meta, agg)
        _add_flops(row, meta=meta, base_cfg=base_cfg, lvm_cfg=lvm_cfg, is_lvm=is_lvm)
        rows.append(row)

    ratio = _ratio_row(rows)
    display_rows = rows + ([ratio] if ratio else [])

    summary_path = rd / "summary.json"
    summary_path.write_text(json.dumps(display_rows, indent=2))

    csv_path = rd / "summary.csv"
    if display_rows:
        keys = sorted({k for r in display_rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in display_rows:
                w.writerow({k: r.get(k) for k in keys})

    plot_path = _stacked_bar(rows, rd / "per_step_breakdown.png")
    sub_plot_path = _lvm_sub_breakdown(rows, rd / "lvm_apply_breakdown.png")
    flops_plot_path = _flops_component_bar(rows, rd / "flops_breakdown.png")

    print(f"summary.json   -> {summary_path}")
    print(f"summary.csv    -> {csv_path}")
    if plot_path:
        print(f"timing plot    -> {plot_path}")
    if sub_plot_path:
        print(f"lvm sub-plot   -> {sub_plot_path}")
    if flops_plot_path:
        print(f"flops plot     -> {flops_plot_path}")
    print()
    _print_tables(display_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
