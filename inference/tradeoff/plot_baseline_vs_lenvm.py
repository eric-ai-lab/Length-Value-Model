#!/usr/bin/env python3
"""
Overlay baseline (budget sweep, length_vs_passk.csv) vs centered-exp runs (*.summary.json)
on shared axes: x = average response length, y = pass@1 (expected).

By default reads baseline from length_vs_passk.csv and method points from
centered_exp_*.summary.json. Each run overwrites the tidy export
baseline_vs_centered_exp_pass1.csv with freshly parsed numbers — edits to that
file alone will be reverted. To plot hand-edited values, use --plot-from-export
or --from-export-csv. Use --no-write-export-csv to skip overwriting the export
when still parsing from sources. Use --use-cache to try the JSON cache first.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CACHE_VERSION = 2
PASS_K = 1

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_BASELINE_CSV = (
    _SCRIPT_DIR
    / "results_lvm_3b-1.5b/budget_eval_baseline_q500_n64_p1.0_topk-1_minp0.01/length_vs_passk.csv"
)
_DEFAULT_CENTERED_DIR = _SCRIPT_DIR / "results_lvm_centered_exp_3b-1.5b"
_BASELINE_CSV_NAME = "length_vs_passk.csv"
_EXPORT_CSV_NAME = "baseline_vs_centered_exp_pass1.csv"


def _stat_fingerprint(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def load_baseline_csv(path: Path) -> Tuple[List[float], List[float], Dict[int, List[float]]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    budgets: List[float] = []
    lengths: List[float] = []
    pass_by_k: Dict[int, List[float]] = {}
    for row in rows:
        budgets.append(float(row["budget"]))
        lengths.append(float(row["avg_capped_tokens"]))
        for key, val in row.items():
            if key.startswith("pass@"):
                k = int(key.split("@", 1)[1])
                pass_by_k.setdefault(k, []).append(float(val))
    return budgets, lengths, pass_by_k


def _iter_centered_summary_paths(directory: Path) -> List[Path]:
    pattern = re.compile(r"centered_exp_(-?\d+)\.summary\.json$")
    out: List[Path] = []
    for p in sorted(directory.glob("*.summary.json")):
        if pattern.search(p.name):
            out.append(p)
    return out


def load_centered_exp_summaries(
    directory: Path,
) -> List[Tuple[float, Dict[int, float], str, float]]:
    """Each tuple: avg length per choice, pass@k map, run tag, centered_exp scale from filename."""
    pattern = re.compile(r"centered_exp_(-?\d+)\.summary\.json$")
    out: List[Tuple[float, Dict[int, float], str, float]] = []
    for p in _iter_centered_summary_paths(directory):
        m = pattern.search(p.name)
        assert m is not None
        tag = f"centered_exp_{m.group(1)}"
        scale = float(m.group(1))
        with p.open(encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        tok = data.get("token_usage") or {}
        length = float(tok.get("avg_completion_tokens_per_choice_assuming_total_over_n", 0.0))
        raw = data.get("pass_at_k_expected") or {}
        pass_at_k = {int(k): float(v) for k, v in raw.items()}
        out.append((length, pass_at_k, tag, scale))
    out.sort(key=lambda x: x[0])
    if not out:
        raise ValueError(f"No *.summary.json with centered_exp_* in {directory}")
    return out


def _summaries_fingerprint(centered_dir: Path) -> Dict[str, Dict[str, int]]:
    fps: Dict[str, Dict[str, int]] = {}
    for p in _iter_centered_summary_paths(centered_dir):
        fps[p.name] = _stat_fingerprint(p)
    return fps


def try_load_cache(
    cache_path: Path,
    baseline_csv: Path,
    centered_dir: Path,
) -> Optional[
    Tuple[
        List[float],
        List[float],
        List[float],
        List[float],
        List[str],
        List[float],
        List[float],
    ]
]:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("version") != CACHE_VERSION or payload.get("pass_k") != PASS_K:
        return None
    if Path(payload["baseline_csv"]).resolve() != baseline_csv.resolve():
        return None
    if Path(payload["centered_dir"]).resolve() != centered_dir.resolve():
        return None
    if payload.get("baseline_fp") != _stat_fingerprint(baseline_csv):
        return None
    cur_sum = _summaries_fingerprint(centered_dir)
    if payload.get("summaries_fp") != cur_sum:
        return None
    base_lengths = [float(x) for x in payload["base_lengths"]]
    base_y = [float(x) for x in payload["base_pass_at_1"]]
    mlen = [float(x) for x in payload["method_lengths"]]
    my = [float(x) for x in payload["method_pass_at_1"]]
    tags = list(payload["method_tags"])
    base_budgets = [float(x) for x in payload["base_budgets"]]
    method_scales = [float(x) for x in payload["method_scales"]]
    return base_lengths, base_y, mlen, my, tags, base_budgets, method_scales


def save_cache(
    cache_path: Path,
    baseline_csv: Path,
    centered_dir: Path,
    base_lengths: List[float],
    base_y: List[float],
    base_budgets: List[float],
    method_points: List[Tuple[float, Dict[int, float], str, float]],
) -> None:
    mlen = [p[0] for p in method_points]
    my = [p[1][PASS_K] for p in method_points]
    tags = [p[2] for p in method_points]
    method_scales = [p[3] for p in method_points]
    payload = {
        "version": CACHE_VERSION,
        "pass_k": PASS_K,
        "baseline_csv": str(baseline_csv.resolve()),
        "centered_dir": str(centered_dir.resolve()),
        "baseline_fp": _stat_fingerprint(baseline_csv),
        "summaries_fp": _summaries_fingerprint(centered_dir),
        "base_budgets": base_budgets,
        "base_lengths": base_lengths,
        "base_pass_at_1": base_y,
        "method_lengths": mlen,
        "method_pass_at_1": my,
        "method_tags": tags,
        "method_scales": method_scales,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cache_csv_path(cache_path: Path) -> Path:
    """Same directory as JSON cache: .../baseline_vs_centered_exp_pass1.cache.json -> .../baseline_vs_centered_exp_pass1.csv"""
    name = cache_path.name
    if name.endswith(".cache.json"):
        return cache_path.with_name(name[: -len(".cache.json")] + ".csv")
    return cache_path.with_suffix(".csv")


def _fmt_csv_num(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return str(x)


def write_pass1_csv(
    path: Path,
    base_budgets: List[float],
    base_lengths: List[float],
    base_y: List[float],
    mlen: List[float],
    my: List[float],
    method_tags: List[str],
    method_scales: List[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["series", "token_budget", "exp_scale", "run_tag", "avg_length", "pass_at_1"]
        )
        for budget, L, p in zip(base_budgets, base_lengths, base_y):
            w.writerow(["baseline", _fmt_csv_num(budget), "", "", f"{L:.6f}", f"{p:.6f}"])
        for L, p, tag, scale in zip(mlen, my, method_tags, method_scales):
            w.writerow(
                ["centered_exp", "", _fmt_csv_num(scale), tag, f"{L:.6f}", f"{p:.6f}"]
            )


_TAG_SCALE_RE = re.compile(r"^centered_exp_(-?\d+)$")


def _scale_from_tag(tag: str) -> Optional[float]:
    m = _TAG_SCALE_RE.match(tag.strip())
    return float(m.group(1)) if m else None


def load_pass1_export_csv(
    path: Path,
) -> Tuple[List[float], List[float], List[float], List[float], List[str], List[float], List[float]]:
    """Load tidy export written by write_pass1_csv."""
    base_lengths: List[float] = []
    base_y: List[float] = []
    mlen: List[float] = []
    my: List[float] = []
    tags: List[str] = []
    base_budgets: List[float] = []
    method_scales: List[float] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"series", "avg_length", "pass_at_1"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} needs columns {sorted(required)}; got {reader.fieldnames!r}")
        cols = set(reader.fieldnames)
        has_budget_col = "token_budget" in cols
        has_scale_col = "exp_scale" in cols
        for row in reader:
            series = (row.get("series") or "").strip()
            L = float(row["avg_length"])
            p = float(row["pass_at_1"])
            if series == "baseline":
                base_lengths.append(L)
                base_y.append(p)
                tb = (row.get("token_budget") or "").strip() if has_budget_col else ""
                base_budgets.append(float(tb) if tb else float("nan"))
            elif series == "centered_exp":
                mlen.append(L)
                my.append(p)
                tag = (row.get("run_tag") or "").strip()
                tags.append(tag)
                es = (row.get("exp_scale") or "").strip() if has_scale_col else ""
                if es:
                    method_scales.append(float(es))
                else:
                    parsed = _scale_from_tag(tag)
                    method_scales.append(parsed if parsed is not None else float("nan"))
    if not base_lengths:
        raise ValueError(f"{path}: no baseline rows")
    if not mlen:
        raise ValueError(f"{path}: no centered_exp rows")
    # Old CSVs without token_budget: fill NaN -> 0.0 for plot/cache compatibility
    if not has_budget_col or all(b != b for b in base_budgets):  # all NaN
        base_budgets = [0.0] * len(base_lengths)
    else:
        base_budgets = [0.0 if (b != b) else b for b in base_budgets]
    method_scales = [0.0 if (s != s) else s for s in method_scales]
    return base_lengths, base_y, mlen, my, tags, base_budgets, method_scales


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s \\
    --baseline-dir path/to/budget_eval_baseline_q500_n64_p1.0_topk-1_minp0.01 \\
    --centered-dir path/to/results_lvm_centered_exp_3b-1.5b \\
    --output-dir path/to/plots_out

  %(prog)s --baseline-csv /abs/path/length_vs_passk.csv --centered-dir ...
""",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=f"Folder containing {_BASELINE_CSV_NAME} (used when --baseline-csv is omitted)",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help=f"Explicit path to {_BASELINE_CSV_NAME} (overrides --baseline-dir). "
        f"Default with no dirs: {_DEFAULT_BASELINE_CSV}",
    )
    parser.add_argument(
        "--centered-dir",
        type=Path,
        default=_DEFAULT_CENTERED_DIR,
        help="Folder with centered_exp_*.summary.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNG, cache JSON, and CSV (default: centered-dir/plots)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Try JSON cache before parsing (skip length_vs_passk.csv/summaries when cache matches)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache JSON path (default: output-dir/baseline_vs_centered_exp_pass1.cache.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not write JSON cache (still reads CSV + summaries unless --from-export-csv)",
    )
    parser.add_argument(
        "--plot-from-export",
        action="store_true",
        help=f"Plot from output-dir/{_EXPORT_CSV_NAME} instead of parsing length_vs_passk + summaries",
    )
    parser.add_argument(
        "--from-export-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Plot from this tidy export CSV (overrides --plot-from-export path)",
    )
    parser.add_argument(
        "--no-write-export-csv",
        action="store_true",
        help=f"Do not overwrite output-dir/{_EXPORT_CSV_NAME} when parsing from sources",
    )
    parser.add_argument(
        "--write-export-csv",
        action="store_true",
        help="When plotting from export CSV, rewrite that file (default: keep hand-edited file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: output-dir/baseline_vs_centered_exp.png)",
    )
    args = parser.parse_args()

    if args.baseline_csv is not None:
        baseline_csv = args.baseline_csv.resolve()
    elif args.baseline_dir is not None:
        baseline_csv = (args.baseline_dir / _BASELINE_CSV_NAME).resolve()
    else:
        baseline_csv = _DEFAULT_BASELINE_CSV.resolve()

    centered_dir = args.centered_dir.resolve()

    if args.output_dir is not None:
        plots_dir = args.output_dir.resolve()
    else:
        plots_dir = centered_dir / "plots"

    cache_path = args.cache
    if cache_path is None:
        cache_path = plots_dir / "baseline_vs_centered_exp_pass1.cache.json"
    else:
        cache_path = cache_path.resolve()

    csv_path = cache_csv_path(cache_path)

    export_in: Optional[Path] = None
    if args.from_export_csv is not None:
        export_in = args.from_export_csv.resolve()
    elif args.plot_from_export:
        export_in = (plots_dir / _EXPORT_CSV_NAME).resolve()

    cached = None
    base_budgets: List[float] = []
    method_scales: List[float] = []
    if export_in is not None:
        if not export_in.is_file():
            raise FileNotFoundError(f"Export CSV not found: {export_in}")
        (
            base_lengths,
            base_y,
            mlen,
            my,
            tags,
            base_budgets,
            method_scales,
        ) = load_pass1_export_csv(export_in)
        print(f"Loaded plot data from export {export_in}")
    else:
        if args.use_cache and not args.no_cache:
            cached = try_load_cache(cache_path, baseline_csv, centered_dir)
            if cached is not None:
                (
                    base_lengths,
                    base_y,
                    mlen,
                    my,
                    tags,
                    base_budgets,
                    method_scales,
                ) = cached
                print(f"Loaded plot data from cache {cache_path}")

        if cached is None:
            base_budgets, base_lengths, base_pass = load_baseline_csv(baseline_csv)
            yb = base_pass.get(PASS_K)
            if yb is None:
                raise KeyError(f"pass@{PASS_K} not in baseline CSV columns")
            method_points = load_centered_exp_summaries(centered_dir)
            for p in method_points:
                if PASS_K not in p[1]:
                    raise KeyError(f"pass@{PASS_K} missing in summary {p[2]}")
            base_y = yb
            mlen = [p[0] for p in method_points]
            my = [p[1][PASS_K] for p in method_points]
            tags = [p[2] for p in method_points]
            method_scales = [p[3] for p in method_points]
            if not args.no_cache:
                save_cache(
                    cache_path,
                    baseline_csv,
                    centered_dir,
                    base_lengths,
                    base_y,
                    base_budgets,
                    method_points,
                )
                print(f"Wrote plot data cache to {cache_path}")

    # Whether to overwrite tidy export CSV on disk
    if export_in is not None:
        should_write_export = args.write_export_csv
    else:
        should_write_export = not args.no_write_export_csv

    if should_write_export:
        write_pass1_csv(
            csv_path,
            base_budgets,
            base_lengths,
            base_y,
            mlen,
            my,
            tags,
            method_scales,
        )
        print(f"Saved {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = {"baseline": "#1f77b4", "ours": "#d62728"}

    ax.plot(
        base_lengths,
        base_y,
        color=colors["baseline"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="baseline (budget sweep)",
    )
    ax.plot(
        mlen,
        my,
        color=colors["ours"],
        marker="s",
        markersize=4,
        linewidth=1.5,
        label="centered exp (ours)",
    )

    ax.set_xlabel("avg response length (tokens)")
    ax.set_ylabel("pass@1")
    ax.set_title("pass@1 vs length")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Baseline vs centered-exp: length–pass@1 trade-off", fontsize=11, y=1.02)
    fig.tight_layout()

    out = args.output
    if out is None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "baseline_vs_centered_exp.png"
    else:
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
