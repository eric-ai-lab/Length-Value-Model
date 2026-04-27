from __future__ import annotations

import argparse
import json
import math
import os
import sys
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as e:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR = e

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _http_post_text(url: str, payload: Dict[str, Any] | None, timeout_s: float) -> str:
    data = b"" if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _normalize_user_text(sample: Dict[str, Any]) -> str:
    conv = sample.get("conversations")
    if not isinstance(conv, list) or not conv:
        raise ValueError("sample.conversations missing or invalid")
    user = next((x for x in conv if x.get("from") in ("human", "user")), None)
    if not user:
        raise ValueError("cannot find user/human message in conversations")
    return str(user.get("value", ""))


def _sigmoid_stable(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def length_to_sigmoid_p(length: float, gamma: float) -> float:
    """Map length L to p in sigmoid space: p(L) = 1 - gamma**L."""
    if not (0.0 < float(gamma) < 1.0):
        raise ValueError(f"gamma must be in (0,1), got {gamma}")
    length = float(length)
    p = 1.0 - (float(gamma) ** length)
    # clamp to [0,1] for safety (in case length is negative due to bad data)
    return float(min(1.0, max(0.0, p)))


def sigmoid_p_to_length(p: float, gamma: float, eps: float) -> float:
    """
    Inverse of `length_to_sigmoid_p`:
      p(L) = 1 - gamma**L
      => L = ln(1 - p) / ln(gamma)
    """
    if not (0.0 < float(gamma) < 1.0):
        raise ValueError(f"gamma must be in (0,1), got {gamma}")
    eps = float(eps)
    pp = float(p)
    # clamp p to (0, 1) for numerical stability
    pp = float(min(1.0 - eps, max(eps, pp)))
    # since gamma in (0,1), ln(gamma) < 0 and ln(1-p) <= 0, so L >= 0
    return float(math.log1p(-pp) / math.log(float(gamma)))


def value_pred_to_length(value_pred: float, gamma: float, eps: float) -> Tuple[float, float]:
    """
    Matches eval_lvm/real_sample_tree_value_visualization.py:
      y_hat = -sigmoid(value_pred)   -> y_hat in (-1, 0)
      length = ln(1 + y_hat) / ln(gamma)
    """
    if not (0.0 < float(gamma) < 1.0):
        raise ValueError(f"gamma must be in (0,1), got {gamma}")
    eps = float(eps)
    # Convert raw head output to discounted return y_hat in (-1, 0)
    x = float(value_pred)
    y_hat = -float(_sigmoid_stable(x))
    # clamp to (-1+eps, 0-eps)
    y = min(-eps, max(-1.0 + eps, y_hat))
    length = math.log1p(y) / min(-eps, math.log(float(gamma)))
    return y_hat, float(length)


@dataclass
class QuestionAggregate:
    lvm_idx: str
    user_text: str
    answer_token_lengths: List[int]

    def gt_mean(self) -> float:
        return float(statistics.mean(self.answer_token_lengths))

    def gt_var(self) -> float:
        # population variance; stable when n==64
        if len(self.answer_token_lengths) <= 1:
            return 0.0
        return float(statistics.pvariance(self.answer_token_lengths))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def group_by_lvm_idx(jsonl_path: Path, expected_samples_per_q: Optional[int]) -> List[QuestionAggregate]:
    """
    Input jsonl is typically "grouped" by lenvm_idx / lvm_idx / index (contiguous blocks), but we do not assume it.
    """
    buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in iter_jsonl(jsonl_path):
        meta = obj.get("meta_info") or {}
        idx = meta.get("lenvm_idx", meta.get("lvm_idx", meta.get("index")))
        if idx is None:
            continue
        buckets[str(idx)].append(obj)

    out: List[QuestionAggregate] = []
    for idx, items in buckets.items():
        # Use the first item's user text.
        user_text = _normalize_user_text(items[0])
        lens: List[int] = []
        for it in items:
            meta = it.get("meta_info") or {}
            atl = meta.get("answer_token_length")
            if atl is None:
                continue
            try:
                lens.append(int(atl))
            except Exception:
                continue
        if not lens:
            continue
        if expected_samples_per_q is not None and len(lens) != expected_samples_per_q:
            # Still keep it, but caller can decide to filter.
            pass
        out.append(QuestionAggregate(lvm_idx=idx, user_text=user_text, answer_token_lengths=lens))
    # deterministic order for reproducibility
    out.sort(key=lambda x: int(x.lvm_idx) if x.lvm_idx.isdigit() else x.lvm_idx)
    return out


def build_prompt_ids(tokenizer, user_text: str) -> List[int]:
    # Prompt ids up to the start of assistant generation.
    prompt_ids: List[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise RuntimeError("tokenizer.apply_chat_template returned empty prompt_ids")
    return [int(x) for x in prompt_ids]


def encode_last_token_value(
    server_url: str,
    input_ids: List[int],
    timeout_s: float,
    retry_on_length_mismatch: bool = True,
) -> float:
    enc_url = server_url.rstrip("/") + "/encode"
    payload = {"input_ids": [input_ids], "bypass_cache": [True]}
    try:
        resp = _http_post_json(enc_url, payload, timeout_s=timeout_s)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from /encode: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call /encode: {e}") from e

    if isinstance(resp, dict) and "embedding" in resp:
        emb = resp["embedding"]
    else:
        emb = resp[0]["embedding"]
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected /encode embedding type: {type(emb)}")

    if len(emb) != len(input_ids):
        if retry_on_length_mismatch:
            # Best-effort flush (may be disabled); then retry once.
            try:
                _http_post_text(server_url.rstrip("/") + "/flush_cache", None, timeout_s=30.0)
            except Exception:
                pass
            return encode_last_token_value(
                server_url=server_url,
                input_ids=input_ids,
                timeout_s=timeout_s,
                retry_on_length_mismatch=False,
            )
        raise RuntimeError(f"/encode embedding length mismatch: got {len(emb)}, expected {len(input_ids)}")

    return float(emb[-1])


def tree_value_last_token_value(
    server_url: str,
    prefix_ids: List[int],
    last_token_id: int,
    timeout_s: float,
) -> float:
    """
    Call /tree_value with:
      input_ids = prefix_ids
      candidate_ids = [last_token_id]
    and return the single predicted raw value.

    This matches "split prefix vs last token" and avoids inserting the candidate into cache.
    """
    url = server_url.rstrip("/") + "/tree_value"
    payload = {"input_ids": [prefix_ids], "candidate_ids": [[int(last_token_id)]]}
    try:
        resp = _http_post_json(url, payload, timeout_s=timeout_s)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from /tree_value: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call /tree_value: {e}") from e

    if isinstance(resp, dict) and "embedding" in resp:
        resp_list = [resp]
    else:
        resp_list = resp
    if not isinstance(resp_list, list) or len(resp_list) != 1:
        raise RuntimeError(f"Unexpected /tree_value response shape: {type(resp)} {resp}")
    emb = resp_list[0].get("embedding")
    if not isinstance(emb, list) or len(emb) != 1:
        raise RuntimeError(f"Expected single-candidate embedding list, got: {emb}")
    return float(emb[0])


def tree_value_batch_last_token_values(
    server_url: str,
    prefix_ids_batch: List[List[int]],
    last_token_id_batch: List[int],
    timeout_s: float,
) -> List[float]:
    """
    Batched /tree_value:
      input_ids:  [B, prefix_len_i]
      candidate_ids: [B, 1]
    Returns a list of length B with the single-candidate raw values.
    """
    if len(prefix_ids_batch) != len(last_token_id_batch):
        raise ValueError("prefix_ids_batch and last_token_id_batch length mismatch")
    url = server_url.rstrip("/") + "/tree_value"
    payload = {
        "input_ids": prefix_ids_batch,
        "candidate_ids": [[int(x)] for x in last_token_id_batch],
    }
    try:
        resp = _http_post_json(url, payload, timeout_s=timeout_s)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from /tree_value: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call /tree_value: {e}") from e

    if isinstance(resp, dict) and "embedding" in resp:
        resp_list = [resp]
    else:
        resp_list = resp
    if not isinstance(resp_list, list) or len(resp_list) != len(prefix_ids_batch):
        raise RuntimeError(f"Unexpected /tree_value response shape: {type(resp)} {resp}")
    out: List[float] = []
    for item in resp_list:
        emb = item.get("embedding")
        if not isinstance(emb, list) or len(emb) != 1:
            raise RuntimeError(f"Expected single-candidate embedding list, got: {emb}")
        out.append(float(emb[0]))
    return out


def encode_batch_last_token_values(
    server_url: str,
    input_ids_batch: List[List[int]],
    timeout_s: float,
    retry_on_length_mismatch: bool = True,
) -> List[float]:
    """
    Batched /encode: returns last-token raw value for each sequence in the batch.
    """
    enc_url = server_url.rstrip("/") + "/encode"
    payload = {"input_ids": input_ids_batch, "bypass_cache": [True] * len(input_ids_batch)}
    try:
        resp = _http_post_json(enc_url, payload, timeout_s=timeout_s)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from /encode: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call /encode: {e}") from e

    if isinstance(resp, dict) and "embedding" in resp:
        resp_list = [resp]
    else:
        resp_list = resp
    if not isinstance(resp_list, list) or len(resp_list) != len(input_ids_batch):
        raise RuntimeError(f"Unexpected /encode response shape: {type(resp)} {resp}")

    out: List[float] = []
    mismatch = False
    for seq_ids, item in zip(input_ids_batch, resp_list):
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Unexpected /encode embedding type: {type(emb)}")
        if len(emb) != len(seq_ids):
            mismatch = True
            break
        out.append(float(emb[-1]))

    if mismatch:
        if retry_on_length_mismatch:
            try:
                _http_post_text(server_url.rstrip("/") + "/flush_cache", None, timeout_s=30.0)
            except Exception:
                pass
            return encode_batch_last_token_values(
                server_url=server_url,
                input_ids_batch=input_ids_batch,
                timeout_s=timeout_s,
                retry_on_length_mismatch=False,
            )
        raise RuntimeError("One or more /encode embedding length mismatches in batch.")

    return out


def chunked(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    if chunk_size <= 0:
        chunk_size = 1
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b if abs(b) > eps else eps)


def sanitize_tag(s: str) -> str:
    """
    Sanitize a string for use in filenames.
    Keeps alphanumerics plus: '-', '_', '.', '='
    """
    s = str(s).strip()
    if not s:
        return "unknown"
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_.=":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:200]


class RunningStats:
    """Online mean/std using Welford's algorithm (population std by default)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self) -> float:
        return float(self.mean) if self.n > 0 else 0.0

    def get_std(self, sample: bool = False) -> float:
        if self.n <= 1:
            return 0.0
        denom = (self.n - 1) if sample else self.n
        return float(math.sqrt(self.M2 / denom))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", type=str, default="http://127.0.0.1:30010")
    ap.add_argument("--model-dir", type=str, required=True, help="HF tokenizer directory")
    ap.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional label appended to output filenames (defaults to basename of --model-dir).",
    )
    ap.add_argument("--data", type=str, required=True, help="Path to a test *.jsonl or *.grouped.jsonl")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--dataset-name", type=str, default=None, help="Optional label for outputs")
    ap.add_argument(
        "--method",
        type=str,
        default="tree_value",
        choices=["tree_value", "encode"],
        help=(
            "tree_value: split prompt into prefix + last token and call /tree_value with 1 candidate (recommended); "
            "encode: call /encode on full prompt and take embedding[-1]."
        ),
    )
    ap.add_argument("--expected-samples-per-question", type=int, default=64)
    ap.add_argument("--allow-incomplete", action="store_true", default=False)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--max-questions", type=int, default=-1)
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Optional small sleep between requests")
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=-1,
        help=(
            "If >0, skip samples whose prompt token length exceeds this value "
            "(do not send request; do not include in summary stats)."
        ),
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max parallel in-flight HTTP requests (parallelism is across batches).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per HTTP request (for both tree_value and encode methods).",
    )
    ap.add_argument(
        "--print-failures",
        action="store_true",
        default=True,
        help="Print errors when a request/tokenization fails (limited by --max-failure-prints).",
    )
    ap.add_argument(
        "--no-print-failures",
        dest="print_failures",
        action="store_false",
        help="Disable printing failures to stderr.",
    )
    ap.add_argument(
        "--max-failure-prints",
        type=int,
        default=50,
        help="Max number of failure messages to print (to avoid flooding the console).",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="If set, abort immediately on any failure (tokenization/HTTP/response shape).",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show a progress bar (uses tqdm if available; otherwise prints periodic progress).",
    )
    ap.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress output.",
    )
    ap.add_argument(
        "--progress-print-every",
        type=int,
        default=200,
        help="When tqdm is unavailable, print progress every N questions (0 disables periodic prints).",
    )
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or data_path.stem
    model_name = sanitize_tag(args.model_name or Path(args.model_dir).name)

    if AutoTokenizer is None:
        raise RuntimeError(
            "Missing dependency: transformers. Install it in your eval env, e.g. `pip install transformers` "
            f"(import error: {_TRANSFORMERS_IMPORT_ERROR})"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)

    expected_n = None if int(args.expected_samples_per_question) <= 0 else int(args.expected_samples_per_question)
    questions = group_by_lvm_idx(data_path, expected_samples_per_q=expected_n)
    if int(args.max_questions) > 0:
        questions = questions[: int(args.max_questions)]

    # Outputs
    out_prefix = f"{dataset_name}.{model_name}"
    per_q_path = out_dir / f"{out_prefix}.per_question.jsonl"
    summary_path = out_dir / f"{out_prefix}.summary.json"

    # Aggregate metrics over questions
    n_used = 0
    sum_gt_mean = 0.0
    sum_gt_var = 0.0
    sum_err = 0.0
    sum_abs_err = 0.0
    sum_rel_err = 0.0
    sum_abs_rel_err = 0.0
    # Sigmoid-space errors (p = sigmoid(v_raw)), where GT is mean over samples of p(L)=1-gamma**L.
    sum_err_sigmoid_p = 0.0
    sum_abs_err_sigmoid_p = 0.0
    sum_rel_err_sigmoid_p = 0.0
    sum_abs_rel_err_sigmoid_p = 0.0
    # Errors where GT length is computed from mean GT sigmoid-p:
    #   GT: mean over samples of p(L)=1-gamma**L -> invert to length
    #   Pred: p_pred = sigmoid(v_raw) -> invert to length
    sum_err_len_from_sigmoid_p_mean = 0.0
    sum_abs_err_len_from_sigmoid_p_mean = 0.0
    sum_rel_err_len_from_sigmoid_p_mean = 0.0
    sum_abs_rel_err_len_from_sigmoid_p_mean = 0.0

    # Std over questions (population std)
    stat_gt_mean = RunningStats()
    stat_gt_var = RunningStats()
    stat_gt_std = RunningStats()
    stat_err = RunningStats()
    stat_abs_err = RunningStats()
    stat_rel_err = RunningStats()
    stat_abs_rel_err = RunningStats()
    stat_err_sigmoid_p = RunningStats()
    stat_abs_err_sigmoid_p = RunningStats()
    stat_rel_err_sigmoid_p = RunningStats()
    stat_abs_rel_err_sigmoid_p = RunningStats()
    stat_err_len_from_sigmoid_p_mean = RunningStats()
    stat_abs_err_len_from_sigmoid_p_mean = RunningStats()
    stat_rel_err_len_from_sigmoid_p_mean = RunningStats()
    stat_abs_rel_err_len_from_sigmoid_p_mean = RunningStats()

    # Best-effort: make sure one bad item doesn't kill the whole run; still surface count.
    n_skipped_incomplete = 0
    n_skipped_overlength = 0
    n_failed_requests = 0
    n_failure_printed = 0

    def maybe_print_failure(msg: str):
        nonlocal n_failure_printed
        if not bool(args.print_failures):
            return
        limit = int(args.max_failure_prints)
        if limit >= 0 and n_failure_printed >= limit:
            return
        n_failure_printed += 1
        print(msg, file=sys.stderr, flush=True)

    def handle_failure(msg: str):
        maybe_print_failure(msg)
        if bool(args.fail_fast):
            raise RuntimeError(msg)

    with per_q_path.open("w", encoding="utf-8") as wf:
        # 1) Precompute prompt token ids (single-thread to avoid tokenizer thread-safety surprises).
        tasks: List[Dict[str, Any]] = []
        pre_it = enumerate(questions)
        if bool(args.progress) and tqdm is not None:
            pre_it = tqdm(pre_it, total=len(questions), desc=f"{dataset_name}:tokenize", unit="q")
        for qi, q in pre_it:
            if expected_n is not None and len(q.answer_token_lengths) != expected_n and not args.allow_incomplete:
                n_skipped_incomplete += 1
                continue
            try:
                prompt_ids = build_prompt_ids(tokenizer, q.user_text)
            except Exception as e:
                n_failed_requests += 1
                handle_failure(f"[{dataset_name}] tokenize_failed lvm_idx={q.lvm_idx}: {e}")
                wf.write(
                    json.dumps(
                        {"lvm_idx": q.lvm_idx, "error": f"tokenize_failed: {e}", "n_samples": len(q.answer_token_lengths)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            if not prompt_ids:
                n_failed_requests += 1
                handle_failure(f"[{dataset_name}] empty_prompt_ids lvm_idx={q.lvm_idx}")
                wf.write(
                    json.dumps(
                        {"lvm_idx": q.lvm_idx, "error": "empty prompt_ids", "n_samples": len(q.answer_token_lengths)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            max_tok = int(args.max_prompt_tokens)
            if max_tok > 0 and len(prompt_ids) > max_tok:
                n_skipped_overlength += 1
                # Record skip for auditing (excluded from summary stats).
                wf.write(
                    json.dumps(
                        {
                            "lvm_idx": q.lvm_idx,
                            "skipped": "over_max_prompt_tokens",
                            "prompt_len": len(prompt_ids),
                            "max_prompt_tokens": max_tok,
                            "n_samples": len(q.answer_token_lengths),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            # Ground truth aggregation in SIGMOID space: map each sample length -> p(L)=1-gamma**L then average.
            try:
                gt_p_list: List[float] = [
                    length_to_sigmoid_p(float(L), gamma=float(args.gamma)) for L in q.answer_token_lengths
                ]
                gt_p_mean = float(statistics.mean(gt_p_list)) if gt_p_list else 0.0
                gt_p_var = float(statistics.pvariance(gt_p_list)) if len(gt_p_list) > 1 else 0.0
                gt_p_std = float(math.sqrt(gt_p_var)) if gt_p_var > 0 else 0.0
                gt_len_from_sigmoid_p_mean = sigmoid_p_to_length(
                    gt_p_mean, gamma=float(args.gamma), eps=float(args.eps)
                )
            except Exception as e:
                n_failed_requests += 1
                handle_failure(f"[{dataset_name}] gt_sigmoid_p_agg_failed lvm_idx={q.lvm_idx}: {e}")
                wf.write(
                    json.dumps(
                        {
                            "lvm_idx": q.lvm_idx,
                            "error": f"gt_sigmoid_p_agg_failed: {e}",
                            "n_samples": len(q.answer_token_lengths),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            tasks.append(
                {
                    "lvm_idx": q.lvm_idx,
                    "n_samples": len(q.answer_token_lengths),
                    "gt_mean": q.gt_mean(),
                    "gt_var": q.gt_var(),
                    "gt_sigmoid_p_mean": gt_p_mean,
                    "gt_sigmoid_p_var": gt_p_var,
                    "gt_sigmoid_p_std": gt_p_std,
                    "gt_len_from_sigmoid_p_mean": float(gt_len_from_sigmoid_p_mean),
                    "prompt_ids": prompt_ids,
                }
            )

        # 2) Parallel HTTP requests over batches.
        batch_size = max(1, int(args.batch_size))
        max_conc = max(1, int(args.max_concurrency))
        batches: List[List[Dict[str, Any]]] = list(chunked(tasks, batch_size))

        def run_batch(batch: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
            if str(args.method) == "encode":
                input_ids_batch = [t["prompt_ids"] for t in batch]
                vals = encode_batch_last_token_values(
                    server_url=args.server_url,
                    input_ids_batch=input_ids_batch,
                    timeout_s=float(args.request_timeout),
                )
                return list(zip(batch, vals))
            # tree_value: prefix + last token
            prefix_ids_batch = [t["prompt_ids"][:-1] for t in batch]
            last_ids = [t["prompt_ids"][-1] for t in batch]
            vals = tree_value_batch_last_token_values(
                server_url=args.server_url,
                prefix_ids_batch=prefix_ids_batch,
                last_token_id_batch=last_ids,
                timeout_s=float(args.request_timeout),
            )
            return list(zip(batch, vals))

        futs = []
        fut_to_batch: Dict[Any, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_conc) as ex:
            for b in batches:
                fut = ex.submit(run_batch, b)
                futs.append(fut)
                fut_to_batch[fut] = b

            done_it = as_completed(futs)
            if bool(args.progress) and tqdm is not None:
                done_it = tqdm(done_it, total=len(futs), desc=f"{dataset_name}:{args.method}", unit="req")

            completed_reqs = 0
            for fut in done_it:
                completed_reqs += 1
                try:
                    pairs = fut.result()
                except Exception as e:
                    # Entire batch failed; record per item.
                    batch = fut_to_batch.get(fut, [])
                    if batch:
                        sample_ids = ",".join(str(x.get("lvm_idx")) for x in batch[:5])
                        handle_failure(
                            f"[{dataset_name}] batch_failed method={args.method} batch_size={len(batch)} "
                            f"sample_lvm_idx=[{sample_ids}] err={e}"
                        )
                    else:
                        handle_failure(f"[{dataset_name}] batch_failed method={args.method} err={e}")
                    for t in batch:
                        n_failed_requests += 1
                        wf.write(
                            json.dumps(
                                {"lvm_idx": t.get("lvm_idx"), "error": str(e), "n_samples": t.get("n_samples")},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                for t, v_raw in pairs:
                    gt_mean = float(t["gt_mean"])
                    gt_var = float(t["gt_var"])
                    gt_std = math.sqrt(gt_var) if gt_var > 0 else 0.0
                    gt_sigmoid_p_mean = float(t.get("gt_sigmoid_p_mean", 0.0))
                    gt_sigmoid_p_var = float(t.get("gt_sigmoid_p_var", 0.0))
                    gt_sigmoid_p_std = float(t.get("gt_sigmoid_p_std", 0.0))
                    gt_len_from_sigmoid_p_mean = float(t.get("gt_len_from_sigmoid_p_mean", 0.0))
                    prompt_ids = t["prompt_ids"]

                    _, pred_len = value_pred_to_length(v_raw, gamma=float(args.gamma), eps=float(args.eps))
                    if not math.isfinite(float(pred_len)) or not math.isfinite(float(v_raw)):
                        handle_failure(
                            f"[{dataset_name}] non_finite_pred lvm_idx={t['lvm_idx']} v_raw={v_raw} pred_len={pred_len}"
                        )
                    err = pred_len - gt_mean
                    abs_err = abs(err)
                    rel_err = safe_div(err, gt_mean)
                    abs_rel_err = abs(rel_err)

                    pred_sigmoid_p = float(_sigmoid_stable(float(v_raw)))
                    err_sigmoid_p = pred_sigmoid_p - gt_sigmoid_p_mean
                    abs_err_sigmoid_p = abs(err_sigmoid_p)
                    rel_err_sigmoid_p = safe_div(err_sigmoid_p, gt_sigmoid_p_mean)
                    abs_rel_err_sigmoid_p = abs(rel_err_sigmoid_p)

                    pred_len_from_sigmoid_p = sigmoid_p_to_length(
                        pred_sigmoid_p, gamma=float(args.gamma), eps=float(args.eps)
                    )
                    err_len_from_sigmoid_p_mean = pred_len_from_sigmoid_p - gt_len_from_sigmoid_p_mean
                    abs_err_len_from_sigmoid_p_mean = abs(err_len_from_sigmoid_p_mean)
                    rel_err_len_from_sigmoid_p_mean = safe_div(
                        err_len_from_sigmoid_p_mean, gt_len_from_sigmoid_p_mean
                    )
                    abs_rel_err_len_from_sigmoid_p_mean = abs(rel_err_len_from_sigmoid_p_mean)

                    sum_gt_mean += gt_mean
                    sum_gt_var += gt_var
                    sum_err += err
                    sum_abs_err += abs_err
                    sum_rel_err += rel_err
                    sum_abs_rel_err += abs_rel_err
                    sum_err_sigmoid_p += err_sigmoid_p
                    sum_abs_err_sigmoid_p += abs_err_sigmoid_p
                    sum_rel_err_sigmoid_p += rel_err_sigmoid_p
                    sum_abs_rel_err_sigmoid_p += abs_rel_err_sigmoid_p
                    sum_err_len_from_sigmoid_p_mean += err_len_from_sigmoid_p_mean
                    sum_abs_err_len_from_sigmoid_p_mean += abs_err_len_from_sigmoid_p_mean
                    sum_rel_err_len_from_sigmoid_p_mean += rel_err_len_from_sigmoid_p_mean
                    sum_abs_rel_err_len_from_sigmoid_p_mean += abs_rel_err_len_from_sigmoid_p_mean
                    n_used += 1

                    stat_gt_mean.update(gt_mean)
                    stat_gt_var.update(gt_var)
                    stat_gt_std.update(gt_std)
                    stat_err.update(err)
                    stat_abs_err.update(abs_err)
                    stat_rel_err.update(rel_err)
                    stat_abs_rel_err.update(abs_rel_err)
                    stat_err_sigmoid_p.update(err_sigmoid_p)
                    stat_abs_err_sigmoid_p.update(abs_err_sigmoid_p)
                    stat_rel_err_sigmoid_p.update(rel_err_sigmoid_p)
                    stat_abs_rel_err_sigmoid_p.update(abs_rel_err_sigmoid_p)
                    stat_err_len_from_sigmoid_p_mean.update(err_len_from_sigmoid_p_mean)
                    stat_abs_err_len_from_sigmoid_p_mean.update(abs_err_len_from_sigmoid_p_mean)
                    stat_rel_err_len_from_sigmoid_p_mean.update(rel_err_len_from_sigmoid_p_mean)
                    stat_abs_rel_err_len_from_sigmoid_p_mean.update(abs_rel_err_len_from_sigmoid_p_mean)

                    wf.write(
                        json.dumps(
                            {
                                "lvm_idx": t["lvm_idx"],
                                "method": str(args.method),
                                "n_samples": int(t["n_samples"]),
                                "gt_mean": gt_mean,
                                "gt_var": gt_var,
                                "gt_std": gt_std,
                                "gt_sigmoid_p_mean": gt_sigmoid_p_mean,
                                "gt_sigmoid_p_var": gt_sigmoid_p_var,
                                "gt_sigmoid_p_std": gt_sigmoid_p_std,
                                "gt_len_from_sigmoid_p_mean": gt_len_from_sigmoid_p_mean,
                                "pred_len": pred_len,
                                "pred_value_raw": float(v_raw),
                                "pred_sigmoid_p": float(pred_sigmoid_p),
                                "pred_len_from_sigmoid_p": float(pred_len_from_sigmoid_p),
                                "prompt_len": len(prompt_ids),
                                "err": err,
                                "abs_err": abs_err,
                                "rel_err": rel_err,
                                "abs_rel_err": abs_rel_err,
                                "err_sigmoid_p": float(err_sigmoid_p),
                                "abs_err_sigmoid_p": float(abs_err_sigmoid_p),
                                "rel_err_sigmoid_p": float(rel_err_sigmoid_p),
                                "abs_rel_err_sigmoid_p": float(abs_rel_err_sigmoid_p),
                                "err_len_from_sigmoid_p_mean": float(err_len_from_sigmoid_p_mean),
                                "abs_err_len_from_sigmoid_p_mean": float(abs_err_len_from_sigmoid_p_mean),
                                "rel_err_len_from_sigmoid_p_mean": float(rel_err_len_from_sigmoid_p_mean),
                                "abs_rel_err_len_from_sigmoid_p_mean": float(abs_rel_err_len_from_sigmoid_p_mean),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if float(args.sleep_s) > 0:
                    time.sleep(float(args.sleep_s))
                if completed_reqs % 20 == 0:
                    wf.flush()
                if bool(args.progress) and tqdm is None:
                    every = int(args.progress_print_every)
                    if every > 0 and completed_reqs % max(1, every // max(1, batch_size)) == 0:
                        print(
                            f"[{dataset_name}] req={completed_reqs}/{len(futs)} "
                            f"used={n_used} skipped_incomplete={n_skipped_incomplete} failed={n_failed_requests}",
                            flush=True,
                        )

    summary = {
        "dataset": dataset_name,
        "data": str(data_path),
        "server_url": str(args.server_url),
        "model_dir": str(args.model_dir),
        "model_name": model_name,
        "method": str(args.method),
        "gamma": float(args.gamma),
        "eps": float(args.eps),
        "batch_size": int(args.batch_size),
        "max_concurrency": int(args.max_concurrency),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "expected_samples_per_question": expected_n,
        "allow_incomplete": bool(args.allow_incomplete),
        "num_questions_total": len(questions),
        "num_questions_used": n_used,
        "num_questions_skipped_incomplete": n_skipped_incomplete,
        "num_questions_skipped_overlength": n_skipped_overlength,
        "num_questions_failed_requests": n_failed_requests,
        # Ground truth stats (averaged over questions)
        "mean_gt_mean_len": safe_div(sum_gt_mean, max(1, n_used)),
        "mean_gt_var_len": safe_div(sum_gt_var, max(1, n_used)),
        "mean_gt_std_len": stat_gt_std.get_mean(),
        # Std over questions
        "std_gt_mean_len": stat_gt_mean.get_std(sample=False),
        "std_gt_var_len": stat_gt_var.get_std(sample=False),
        "std_gt_std_len": stat_gt_std.get_std(sample=False),
        # Your requested aggregates (mean over questions)
        "mean_err": safe_div(sum_err, max(1, n_used)),
        "mean_abs_err": safe_div(sum_abs_err, max(1, n_used)),
        "mean_rel_err": safe_div(sum_rel_err, max(1, n_used)),
        "mean_abs_rel_err": safe_div(sum_abs_rel_err, max(1, n_used)),
        # Std of per-question errors
        "std_err": stat_err.get_std(sample=False),
        "std_abs_err": stat_abs_err.get_std(sample=False),
        "std_rel_err": stat_rel_err.get_std(sample=False),
        "std_abs_rel_err": stat_abs_rel_err.get_std(sample=False),
        # Sigmoid-space (p = sigmoid(v_raw)) error aggregates where GT is mean over samples of p(L)=1-gamma**L.
        "mean_err_sigmoid_p": safe_div(sum_err_sigmoid_p, max(1, n_used)),
        "mean_abs_err_sigmoid_p": safe_div(sum_abs_err_sigmoid_p, max(1, n_used)),
        "mean_rel_err_sigmoid_p": safe_div(sum_rel_err_sigmoid_p, max(1, n_used)),
        "mean_abs_rel_err_sigmoid_p": safe_div(sum_abs_rel_err_sigmoid_p, max(1, n_used)),
        "std_err_sigmoid_p": stat_err_sigmoid_p.get_std(sample=False),
        "std_abs_err_sigmoid_p": stat_abs_err_sigmoid_p.get_std(sample=False),
        "std_rel_err_sigmoid_p": stat_rel_err_sigmoid_p.get_std(sample=False),
        "std_abs_rel_err_sigmoid_p": stat_abs_rel_err_sigmoid_p.get_std(sample=False),
        # Length errors computed via sigmoid-p mean:
        #   GT: mean over samples p(L)=1-gamma**L -> invert to length
        #   Pred: sigmoid(v_raw) -> invert to length
        "mean_err_len_from_sigmoid_p_mean": safe_div(sum_err_len_from_sigmoid_p_mean, max(1, n_used)),
        "mean_abs_err_len_from_sigmoid_p_mean": safe_div(sum_abs_err_len_from_sigmoid_p_mean, max(1, n_used)),
        "mean_rel_err_len_from_sigmoid_p_mean": safe_div(sum_rel_err_len_from_sigmoid_p_mean, max(1, n_used)),
        "mean_abs_rel_err_len_from_sigmoid_p_mean": safe_div(sum_abs_rel_err_len_from_sigmoid_p_mean, max(1, n_used)),
        "std_err_len_from_sigmoid_p_mean": stat_err_len_from_sigmoid_p_mean.get_std(sample=False),
        "std_abs_err_len_from_sigmoid_p_mean": stat_abs_err_len_from_sigmoid_p_mean.get_std(sample=False),
        "std_rel_err_len_from_sigmoid_p_mean": stat_rel_err_len_from_sigmoid_p_mean.get_std(sample=False),
        "std_abs_rel_err_len_from_sigmoid_p_mean": stat_abs_rel_err_len_from_sigmoid_p_mean.get_std(sample=False),
        # Extra info that is often useful
        "output_per_question_jsonl": str(per_q_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Reduce tokenizer parallelism noise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

