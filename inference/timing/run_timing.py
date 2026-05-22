"""Run a single sample_eval pass and capture wall-clock + token counts.

Thin wrapper over inference.tradeoff.sample_eval that adds:
- end-to-end wall-clock timing (perf_counter)
- background nvidia-smi sampling at 1 Hz
- aggregate token counts from the run's responses jsonl
- emits <output-dir>/<tag>.meta.json with the above

The server-side per-step timing (SGLANG_LVM_TIMING_LOG) is written by the
sglang process itself; this script does not touch that file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _build_sample_eval_argv(args: argparse.Namespace, tag: str) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "inference.tradeoff.sample_eval",
        "--dataset-name", args.dataset_name,
        "--server-url", args.server_url,
        "--output-dir", str(args.output_dir),
        "--tag", tag,
        "--stage", "run",
        "--max-questions", str(args.max_questions),
        "--max-concurrency", str(args.max_concurrency),
        "--request-timeout", str(args.request_timeout),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--min-p", str(args.min_p),
        "--n", str(args.n),
        "--http-backend", args.http_backend,
    ]
    if args.value_scale is not None:
        cmd += ["--value-scale", str(args.value_scale)]
    if args.value_mode is not None:
        cmd += ["--value-mode", args.value_mode]
    if args.value_gamma is not None:
        cmd += ["--value-gamma", str(args.value_gamma)]
    return cmd


@dataclass
class GpuSample:
    t_offset_s: float
    gpu_util_pct: List[int]
    mem_used_mib: List[int]


def _gpu_sampler(stop_path: Path, output_path: Path, period_s: float = 1.0) -> None:
    """nvidia-smi sampler subprocess body (invoked via -c)."""
    # Not used; we run nvidia-smi from the parent.
    raise SystemExit(0)


def _start_gpu_sampler(samples_out: Path) -> Optional[subprocess.Popen]:
    if not shutil.which("nvidia-smi"):
        return None
    # nvidia-smi --query streams every <period>s and we tee to file.
    # Format: timestamp,index,utilization.gpu,memory.used
    fh = open(samples_out, "w")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used",
            "--format=csv,nounits",
            "-lms", "1000",
        ],
        stdout=fh,
        stderr=subprocess.DEVNULL,
    )
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def _stop_gpu_sampler(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        fh.close()


def _summarize_responses(responses_jsonl: Path) -> Dict[str, Any]:
    """Aggregate token counts and per-question latency from sample_eval output.

    sample_eval writes one row per choice but duplicates the full-request
    `usage` field across every choice's row, so we dedupe by question `idx`
    (counting each request's usage once) before summing. `n_requests` and the
    per-choice latency stats still sample every row.
    """
    n_requests = 0
    total_output_tokens = 0
    total_prompt_tokens = 0
    output_token_counts: List[int] = []
    latencies: List[float] = []
    seen_idx: set = set()
    with responses_jsonl.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_requests += 1
            idx = row.get("idx")
            if idx is not None and idx not in seen_idx:
                seen_idx.add(idx)
                usage = row.get("usage") or {}
                out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                in_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                total_output_tokens += int(out_tok or 0)
                total_prompt_tokens += int(in_tok or 0)
                if out_tok:
                    output_token_counts.append(int(out_tok))
            lat = row.get("latency_s") or row.get("elapsed_s")
            if isinstance(lat, (int, float)):
                latencies.append(float(lat))
    summary: Dict[str, Any] = {
        "n_requests": n_requests,
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
    }
    if output_token_counts:
        output_token_counts.sort()
        n = len(output_token_counts)
        summary["output_tokens_mean"] = sum(output_token_counts) / n
        summary["output_tokens_p50"] = output_token_counts[n // 2]
        summary["output_tokens_p95"] = output_token_counts[int(n * 0.95)]
        summary["output_tokens_max"] = output_token_counts[-1]
    if latencies:
        latencies.sort()
        n = len(latencies)
        summary["latency_s_mean"] = sum(latencies) / n
        summary["latency_s_p50"] = latencies[n // 2]
        summary["latency_s_p95"] = latencies[int(n * 0.95)]
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Output prefix (e.g. baseline, lenvm)")
    p.add_argument("--server-url", required=True)
    p.add_argument("--dataset-name", default="gsm8k")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-questions", type=int, default=50)
    p.add_argument("--max-concurrency", type=int, default=50)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.01)
    p.add_argument("--request-timeout", type=float, default=600000)
    p.add_argument("--http-backend", default="aiohttp")
    p.add_argument("--value-scale", type=float, default=None)
    p.add_argument("--value-mode", default=None)
    p.add_argument("--value-gamma", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # sample_eval writes responses jsonl using the tag in its own canonical
    # naming; we reuse the same tag so the file is predictable.
    sample_eval_tag = (
        f"{args.tag}_q{args.max_questions}_n{args.n}_p{args.top_p}_"
        f"topk{args.top_k}_minp{args.min_p}"
    )

    cmd = _build_sample_eval_argv(args, tag=sample_eval_tag)

    gpu_samples_path = args.output_dir / f"{args.tag}.gpu_samples.csv"
    gpu_proc = _start_gpu_sampler(gpu_samples_path)

    t_start = time.perf_counter()
    t_wall_start = time.time()
    try:
        rc = subprocess.call(cmd)
    finally:
        _stop_gpu_sampler(gpu_proc)
    t_end = time.perf_counter()
    t_wall_end = time.time()

    if rc != 0:
        print(f"sample_eval exited with rc={rc}", file=sys.stderr)
        return rc

    # Match sample_eval's compute_paths: <dataset>.<tag>.responses.jsonl
    responses_path = args.output_dir / f"{args.dataset_name}.{sample_eval_tag}.responses.jsonl"
    summary = _summarize_responses(responses_path) if responses_path.exists() else {}

    wall_clock_s = t_end - t_start
    meta: Dict[str, Any] = {
        "tag": args.tag,
        "server_url": args.server_url,
        "dataset": args.dataset_name,
        "max_questions": args.max_questions,
        "n_samples_per_q": args.n,
        "max_concurrency": args.max_concurrency,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "value_scale": args.value_scale,
        "value_mode": args.value_mode,
        "value_gamma": args.value_gamma,
        "wall_clock_s": wall_clock_s,
        "wall_start_epoch_s": t_wall_start,
        "wall_end_epoch_s": t_wall_end,
        "responses_path": str(responses_path),
        "summary": summary,
        "cmd": cmd,
    }
    if summary.get("total_output_tokens"):
        meta["throughput_output_tokens_per_s"] = (
            summary["total_output_tokens"] / wall_clock_s if wall_clock_s > 0 else 0
        )

    meta_path = args.output_dir / f"{args.tag}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta -> {meta_path}")
    print(f"wall_clock_s={wall_clock_s:.2f} "
          f"out_tokens={summary.get('total_output_tokens', '?')} "
          f"throughput={meta.get('throughput_output_tokens_per_s', 0):.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
