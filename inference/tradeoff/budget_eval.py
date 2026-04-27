"""Token-budget baseline evaluation.

Given an existing responses.jsonl (from sample_eval.py) and a list of token budgets,
compute pass@k where any choice whose response token count EXCEEDS the budget is
treated as incorrect (regardless of answer correctness).

This lets you simulate a hard token-budget constraint without re-running inference.

Usage example:
    python budget_eval.py \\
        --responses results_lvm_7b-1.5b/math500.baseline_q500_n64_p0.95_topk10_minp0.01.responses.jsonl \\
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \\
        --budgets 200 400 800 1600 3200 \\
        --output-dir results_lvm_7b-1.5b/budget_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer  # type: ignore

    _TRANSFORMERS_OK = True
except Exception:
    AutoTokenizer = None  # type: ignore
    _TRANSFORMERS_OK = False

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify  # type: ignore
    from latex2sympy2_extended import NormalizationConfig  # type: ignore

    _MATH_VERIFY_OK = True
except Exception:
    _MATH_VERIFY_OK = False

try:
    from tqdm import tqdm  # type: ignore

    _TQDM_OK = True
except Exception:
    tqdm = None  # type: ignore
    _TQDM_OK = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_name_or_path: str):
    """Load a HuggingFace tokenizer. Returns an object with .encode(text) -> list."""
    if not _TRANSFORMERS_OK:
        raise RuntimeError(
            "transformers is not installed. Install it with: pip install transformers"
        )
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return tok


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text using the given tokenizer."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Math evaluator (mirrors sample_eval.py)
# ---------------------------------------------------------------------------

def _extract_after_think(text: str) -> str:
    m = re.search(r"</think>(.*)", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def _fallback_extract_answer(text: str) -> str:
    t = _extract_after_think(text)
    boxed = re.findall(r"\\boxed\{([^}]*)\}", t, flags=re.DOTALL)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r"Answer\s*:\s*(.*)$", t, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    if nums:
        return nums[-1].strip()
    return t.strip()


def _fallback_equiv(pred_text: str, gold_text: str) -> bool:
    pred = _fallback_extract_answer(pred_text)
    gold = str(gold_text).strip()
    try:
        return float(pred) == float(gold)
    except Exception:
        pass
    p = re.sub(r"\s+", " ", pred).strip()
    g = re.sub(r"\s+", " ", gold).strip()
    return p == g


class RuleBasedMathEvaluator:
    def __init__(self, use_math_verify: bool = True):
        self.use_math_verify = bool(use_math_verify and _MATH_VERIFY_OK)

    def judge(self, solution_str: str, ground_truth: str) -> Tuple[bool, str]:
        if not self.use_math_verify:
            ok = _fallback_equiv(solution_str, ground_truth)
            return ok, _fallback_extract_answer(solution_str)

        gold = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "NO_EXTRACT"
        return bool(verify(gold, answer)), str(answer)


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k_expected(n: int, c: int, k: int) -> float:
    n, c, k = int(n), int(c), int(k)
    if k <= 0 or n <= 0:
        return 0.0
    k = min(k, n)
    c = max(0, min(c, n))
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    try:
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k))
    except Exception:
        a, b, ratio = n - c, n, 1.0
        for t in range(k):
            ratio *= (a - t) / (b - t)
        return 1.0 - ratio


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_and_count_tokens(
    responses_jsonl: Path,
    tokenizer,
) -> Dict[int, List[Dict[str, Any]]]:
    """Load responses.jsonl and annotate each row with its response token count.

    Returns a dict: idx -> list of row dicts (with added 'response_tokens' field).
    """
    by_idx: Dict[int, List[Dict[str, Any]]] = {}
    rows_iter = iter_jsonl(responses_jsonl)
    if _TQDM_OK:
        rows_iter = tqdm(rows_iter, desc="counting tokens", unit="row")

    for obj in rows_iter:
        idx = obj.get("idx")
        if not isinstance(idx, int):
            continue
        if obj.get("error") is not None:
            continue
        text = obj.get("text")
        if text is None:
            continue
        n_tok = count_tokens(tokenizer, str(text))
        obj = dict(obj)
        obj["response_tokens"] = n_tok
        by_idx.setdefault(idx, []).append(obj)

    return by_idx


def eval_budget(
    by_idx: Dict[int, List[Dict[str, Any]]],
    budget: int,
    evaluator: RuleBasedMathEvaluator,
    pass_ks: List[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute pass@k for a single token budget.

    A choice is considered 'within budget' only if response_tokens <= budget.
    Choices exceeding the budget are treated as incorrect.

    Returns (summary_dict, per_question_rows).
    """
    pass_ks = sorted({int(k) for k in pass_ks if int(k) > 0})

    n_questions = 0
    sum_pass_k_expected: Dict[int, float] = {k: 0.0 for k in pass_ks}
    sum_pass_k_firstk: Dict[int, float] = {k: 0.0 for k in pass_ks}
    sum_within_budget_frac = 0.0
    sum_avg_capped_tokens = 0.0

    per_q_rows: List[Dict[str, Any]] = []

    for idx in sorted(by_idx.keys()):
        rows = by_idx[idx]
        if not rows:
            continue

        gold = str(rows[0].get("answer", ""))
        rows_sorted = sorted(
            rows,
            key=lambda r: int(r.get("choice_idx")) if r.get("choice_idx") is not None else 0,
        )

        n_questions += 1
        within_budget_flags: List[bool] = []
        correct_flags: List[bool] = []
        capped_token_counts: List[int] = []

        for r in rows_sorted:
            n_tok = int(r.get("response_tokens", 0))
            within = n_tok <= budget
            within_budget_flags.append(within)
            capped_token_counts.append(min(n_tok, budget))
            if within:
                ok, _ = evaluator.judge(str(r.get("text", "")), gold)
            else:
                ok = False
            correct_flags.append(ok)

        n_seen = len(correct_flags)
        c = sum(1 for x in correct_flags if x)
        n_within = sum(1 for x in within_budget_flags if x)
        avg_capped_tokens = sum(capped_token_counts) / n_seen if n_seen > 0 else 0.0

        sum_within_budget_frac += n_within / n_seen if n_seen > 0 else 0.0
        sum_avg_capped_tokens += avg_capped_tokens

        pass_k_expected_row: Dict[int, float] = {}
        pass_k_firstk_row: Dict[int, float] = {}
        for k in pass_ks:
            k_eff = min(k, n_seen)
            pk_exp = pass_at_k_expected(n_seen, c, k_eff)
            pk_first = 1.0 if any(correct_flags[:k_eff]) else 0.0
            pass_k_expected_row[k] = pk_exp
            pass_k_firstk_row[k] = pk_first
            sum_pass_k_expected[k] += pk_exp
            sum_pass_k_firstk[k] += pk_first

        per_q_rows.append(
            {
                "idx": idx,
                "answer": gold,
                "budget": budget,
                "n_choices_seen": n_seen,
                "n_within_budget": n_within,
                "n_correct": c,
                "avg_capped_tokens": avg_capped_tokens,
                "correct_first": bool(correct_flags[0]) if correct_flags else False,
                "correct_any": bool(any(correct_flags)) if correct_flags else False,
                "pass_at_k_expected": pass_k_expected_row,
                "pass_at_k_firstk": pass_k_firstk_row,
            }
        )

    summary = {
        "budget": budget,
        "n_questions": n_questions,
        "pass_ks": pass_ks,
        "pass_at_k_expected": {
            str(k): (sum_pass_k_expected[k] / n_questions) if n_questions else 0.0
            for k in pass_ks
        },
        "pass_at_k_firstk": {
            str(k): (sum_pass_k_firstk[k] / n_questions) if n_questions else 0.0
            for k in pass_ks
        },
        "avg_within_budget_frac": (sum_within_budget_frac / n_questions) if n_questions else 0.0,
        "avg_capped_tokens": (sum_avg_capped_tokens / n_questions) if n_questions else 0.0,
    }
    return summary, per_q_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate pass@k under token-budget constraints from existing responses.jsonl"
    )
    p.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to responses.jsonl produced by sample_eval.py",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace tokenizer name or local path (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        required=True,
        help="List of token budgets to evaluate, e.g. --budgets 200 400 800 1600 3200",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write per-budget summary JSON and per-question JSONL files",
    )
    p.add_argument(
        "--no-math-verify",
        action="store_true",
        default=False,
        help="Disable math_verify; use fallback rule-based checker",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def compute_pass_ks_pow2(by_idx: Dict[int, List[Dict[str, Any]]]) -> List[int]:
    max_n = max((len(rows) for rows in by_idx.values()), default=0)
    if max_n <= 0:
        return [1]
    ks: List[int] = []
    k = 1
    while k <= max_n:
        ks.append(k)
        k *= 2
    return ks


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    responses_path = Path(args.responses)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = load_tokenizer(args.tokenizer)

    logger.info("Loading responses and counting tokens: %s", responses_path)
    by_idx = load_and_count_tokens(responses_path, tokenizer)
    logger.info("Loaded %d questions", len(by_idx))

    pass_ks = compute_pass_ks_pow2(by_idx)
    logger.info("pass_ks: %s", pass_ks)

    evaluator = RuleBasedMathEvaluator(use_math_verify=not args.no_math_verify)
    logger.info("use_math_verify=%s", evaluator.use_math_verify)

    all_summaries: List[Dict[str, Any]] = []

    for budget in sorted(args.budgets):
        logger.info("Evaluating budget=%d ...", budget)
        summary, per_q_rows = eval_budget(by_idx, budget, evaluator, pass_ks)

        # Write per-question JSONL
        per_q_path = output_dir / f"budget_{budget}.per_question.jsonl"
        write_jsonl(per_q_path, per_q_rows)

        # Write per-budget summary JSON
        summary_path = output_dir / f"budget_{budget}.summary.json"
        summary["responses_jsonl"] = str(responses_path)
        summary["per_question_jsonl"] = str(per_q_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        all_summaries.append(summary)

        # Print brief report
        print(f"\n--- budget={budget} ---")
        print(f"  n_questions : {summary['n_questions']}")
        print(f"  within_budget_frac (avg): {summary['avg_within_budget_frac']:.3f}")
        print(f"  avg_capped_tokens       : {summary['avg_capped_tokens']:.1f}")
        for k in pass_ks:
            exp = summary["pass_at_k_expected"].get(str(k), 0.0)
            print(f"  pass@{k:<4}  {exp:.4f}")

    # Write combined summary
    combined_path = output_dir / "all_budgets.summary.json"
    combined_path.write_text(
        json.dumps(
            {
                "responses_jsonl": str(responses_path),
                "tokenizer": args.tokenizer,
                "pass_ks": pass_ks,
                "budgets": all_summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote combined summary to %s", combined_path)

    # Build avg_length vs pass@k table and plots
    avg_lengths = [s["avg_capped_tokens"] for s in all_summaries]

    # CSV: rows = budgets, cols = avg_length + pass@k_expected for each k
    csv_path = output_dir / "length_vs_passk.csv"
    header = ["budget", "avg_capped_tokens"] + [f"pass@{k}" for k in pass_ks]
    rows_csv = []
    for s in all_summaries:
        row = [str(s["budget"]), f"{s['avg_capped_tokens']:.4f}"]
        for k in pass_ks:
            row.append(f"{s['pass_at_k_expected'].get(str(k), 0.0):.6f}")
        rows_csv.append(row)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows_csv:
            f.write(",".join(row) + "\n")
    logger.info("Wrote length vs pass@k table to %s", csv_path)

    # Plots: one figure per k, x=avg_capped_tokens, y=pass@k_expected
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for k in pass_ks:
            y_exp = [s["pass_at_k_expected"].get(str(k), 0.0) for s in all_summaries]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(avg_lengths, y_exp, marker="o", markersize=3)
            ax.set_xlabel("avg capped tokens")
            ax.set_ylabel(f"pass@{k}")
            ax.set_title(f"pass@{k} vs avg response length (token budget)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = plots_dir / f"pass_at_{k}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info("Saved plot %s", plot_path)

    except ImportError:
        logger.warning("matplotlib not installed; skipping plots")


if __name__ == "__main__":
    main()
