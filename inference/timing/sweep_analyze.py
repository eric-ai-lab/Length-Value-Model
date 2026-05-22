"""Aggregate timing results from multiple ``lenvm_timing.sh`` runs into a top-k sweep.

Each input ``--results-dirs`` entry is a directory produced by a single
``lenvm_timing.sh`` invocation (so it contains ``summary.json`` + per-stage
``*.meta.json`` + ``*.timing.jsonl``). This script extracts ``LENVM_TOP_K`` from
the LenVM ``meta.json`` and aggregates the per-run metrics into a single CSV +
plot, plus a stdout table. Use this to answer:

- "How does the wall-clock slowdown scale with the LenVM candidate-set size?"
- "Does the theoretical FLOPs ratio track the measured wall-clock ratio as k
  grows, or does the GPU-utilization gap widen?"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_RATIO_KEYS = [
    "wall_clock_s",
    "throughput_output_tokens_per_s",
    "theoretical_pflops_total",
    "achieved_tflops_per_s",
    "t_sampler_total_ms_mean",
    "t_lvm_apply_outer_ms_mean",
    "t_lvm_forward_ms_mean",
]


def _load_summary(rd: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return (baseline_row, lenvm_row, ratio_row) from a summary.json."""
    rows: List[Dict[str, Any]] = json.loads((rd / "summary.json").read_text())
    base = next(r for r in rows if r.get("tag") == "baseline")
    lvm = next(r for r in rows if r.get("tag") == "lenvm")
    ratio = next((r for r in rows if "ratio" in str(r.get("tag"))), {})
    return base, lvm, ratio


def _infer_k(rd: Path, lvm_row: Dict[str, Any]) -> Optional[int]:
    k = lvm_row.get("top_k")
    if isinstance(k, int) and k > 0:
        return k
    # Fall back to parsing the directory name (e.g. sweep_q50_n16_k5).
    m = re.search(r"_k(\d+)", rd.name)
    if m:
        return int(m.group(1))
    return None


def _percent(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x * 100:.1f}%"


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_table(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]]) -> None:
    widths = {k: max(len(label), max(len(_fmt(r.get(k))) for r in rows)) for k, label in cols}
    header = " | ".join(f"{label:>{widths[k]}}" for k, label in cols)
    print(header)
    print("-+-".join("-" * widths[k] for k, _ in cols))
    for r in rows:
        print(" | ".join(f"{_fmt(r.get(k)):>{widths[k]}}" for k, _ in cols))


def _plot(rows: List[Dict[str, Any]], out_png: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable, skipping plot: {e}")
        return None

    rows = [r for r in rows if r.get("k") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    flops_ratio = [r.get("flops_ratio") for r in rows]
    wall_ratio = [r.get("wall_ratio") for r in rows]
    util_ratio = [r.get("achieved_tflops_ratio") for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, flops_ratio, marker="o", label="theoretical FLOPs ratio (LenVM / baseline)")
    ax.plot(ks, wall_ratio, marker="s", label="measured wall-clock ratio")
    if any(v is not None for v in util_ratio):
        ax.plot(ks, util_ratio, marker="^", linestyle="--",
                label="achieved TFLOPs/s ratio (≤1 means utilization loss)")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("LenVM candidate-set size k")
    ax.set_ylabel("Ratio (LenVM / baseline)")
    ax.set_title("LenVM overhead vs candidate-set size")
    ax.set_xticks(ks)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dirs", type=Path, nargs="+", required=True,
                   help="Directories produced by lenvm_timing.sh, one per top-k value.")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for rd in args.results_dirs:
        if not (rd / "summary.json").exists():
            print(f"skipping {rd} — no summary.json")
            continue
        base, lvm, ratio = _load_summary(rd)
        k = _infer_k(rd, lvm)
        row: Dict[str, Any] = {
            "k": k,
            "results_dir": str(rd),
            "base_wall_s": base.get("wall_clock_s"),
            "lvm_wall_s": lvm.get("wall_clock_s"),
            "wall_ratio": ratio.get("wall_clock_s"),
            "base_tok_per_s": base.get("throughput_output_tokens_per_s"),
            "lvm_tok_per_s": lvm.get("throughput_output_tokens_per_s"),
            "tok_per_s_ratio": ratio.get("throughput_output_tokens_per_s"),
            "base_pflops": base.get("theoretical_pflops_total"),
            "lvm_pflops": lvm.get("theoretical_pflops_total"),
            "flops_ratio": ratio.get("theoretical_pflops_total"),
            "base_achieved_tflops_per_s": base.get("achieved_tflops_per_s"),
            "lvm_achieved_tflops_per_s": lvm.get("achieved_tflops_per_s"),
            "achieved_tflops_ratio": ratio.get("achieved_tflops_per_s"),
            "lvm_apply_ms_mean": lvm.get("t_lvm_apply_outer_ms_mean"),
            "lvm_forward_ms_mean": lvm.get("t_lvm_forward_ms_mean"),
            "lvm_build_pending_ms_mean": lvm.get("t_lvm_build_pending_ms_mean"),
            "lvm_apply_guidance_ms_mean": lvm.get("t_lvm_apply_guidance_ms_mean"),
        }
        if row["wall_ratio"] and row["flops_ratio"]:
            # Utilization gap = wall-clock ratio / theoretical FLOPs ratio.
            # >1 means LenVM is slower wall-clock than its raw extra compute alone would predict.
            row["utilization_gap"] = row["wall_ratio"] / row["flops_ratio"]
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r.get("k") if r.get("k") is not None else 999))

    summary_json = args.output_dir / "sweep_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2))

    csv_path = args.output_dir / "sweep_summary.csv"
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

    plot_path = _plot(rows, args.output_dir / "topk_sweep.png")

    print(f"summary.json -> {summary_json}")
    print(f"summary.csv  -> {csv_path}")
    if plot_path:
        print(f"plot         -> {plot_path}")
    print()
    _print_table(
        rows,
        cols=[
            ("k", "k"),
            ("base_wall_s", "base_s"),
            ("lvm_wall_s", "lvm_s"),
            ("wall_ratio", "wall_ratio"),
            ("flops_ratio", "flops_ratio"),
            ("utilization_gap", "util_gap"),
            ("lvm_apply_ms_mean", "apply_ms"),
            ("lvm_forward_ms_mean", "lvm_fwd_ms"),
            ("lvm_tok_per_s", "lvm_tok/s"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
