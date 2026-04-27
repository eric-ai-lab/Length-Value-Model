from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

import datasets

logger = logging.getLogger(__name__)

MATH_QUERY_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}"
)

VQA_MCQ_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}"
)
VQA_OPEN_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}"
)

# All known MathVista Hint prefixes (first line of the query field, without trailing \n).
# Raise ValueError if a query starts with something not in this list.
_MATHVISTA_HINT_PREFIXES = [
    "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.",
    "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.",
    "Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.",
    "Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.",
    "Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.",
]


try:
    # Optional dependency (preferred): rule-based math equivalence verification.
    # Reference implementation: https://github.com/eric-ai-lab/Soft-Thinking/blob/main/matheval.py
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify  # type: ignore
    from latex2sympy2_extended import NormalizationConfig  # type: ignore

    _MATH_VERIFY_OK = True
except Exception:
    _MATH_VERIFY_OK = False
    import warnings
    warnings.warn(
        "math_verify / latex2sympy2_extended not installed — falling back to simple string/float "
        "comparison for answer checking. Install with: pip install math-verify latex2sympy2-extended",
        stacklevel=1,
    )

try:
    import aiohttp  # type: ignore

    _AIOHTTP_OK = True
except Exception:
    aiohttp = None  # type: ignore
    _AIOHTTP_OK = False

try:
    from tqdm import tqdm  # type: ignore

    _TQDM_OK = True
except Exception:
    tqdm = None  # type: ignore
    _TQDM_OK = False


def load_dataset_and_template(dataset_name: str) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    if dataset_name == "math500":
        raw_dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        for ex in raw_dataset:
            data.append(
                {
                    "question": MATH_QUERY_TEMPLATE.format(Question=ex["problem"]),
                    "answer": str(ex["answer"]),
                }
            )
    elif dataset_name == "gsm8k":
        raw_dataset = datasets.load_dataset("openai/gsm8k", "main")
        for ex in raw_dataset["test"]:
            # Extract the answer after ####
            ans = str(ex["answer"]).split("####")[-1].strip()
            data.append(
                {
                    "question": MATH_QUERY_TEMPLATE.format(Question=ex["question"]),
                    "answer": ans,
                }
            )
    elif dataset_name == "aime":
        raw_dataset = datasets.load_dataset("GY2233/AIME-2024-2025", split="train")
        for ex in raw_dataset:
            data.append(
                {
                    "question": MATH_QUERY_TEMPLATE.format(Question=ex["Problem"]),
                    "answer": str(ex["Answer"]),
                }
            )
    elif dataset_name == "amc23":
        raw_dataset = datasets.load_dataset("zwhe99/amc23", split="test")
        for ex in raw_dataset:
            data.append(
                {
                    "question": MATH_QUERY_TEMPLATE.format(Question=ex["question"]),
                    "answer": str(int(ex["answer"])),
                }
            )
    elif dataset_name == "mathvista":
        cache_path = Path(__file__).parent / ".cache_mathvista_testmini.jsonl"
        if cache_path.exists():
            logger.info("loading mathvista from cache: %s", cache_path)
            for obj in iter_jsonl(cache_path):
                data.append(obj)
        else:
            logger.info("building mathvista cache (first time only) ...")
            raw_dataset = datasets.load_dataset("AI4Math/MathVista", split="testmini")
            labels = ["A", "B", "C", "D", "E"]
            _hint_to_suffix = {
                _MATHVISTA_HINT_PREFIXES[0]: "Answer with the option letter, e.g., A, B, C, D.",
                _MATHVISTA_HINT_PREFIXES[1]: "\nAnswer with an integer.",
                _MATHVISTA_HINT_PREFIXES[2]: "\nAnswer with a number with one decimal place.",
                _MATHVISTA_HINT_PREFIXES[3]: "\nAnswer with a number with two decimal places.",
                _MATHVISTA_HINT_PREFIXES[4]: "\nAnswer with a Python list.",
            }
            for ex in raw_dataset:
                query: str = ex["query"]
                question_type: str = ex["question_type"]
                choices: List[str] = ex.get("choices") or []
                question_text, hint = _strip_mathvista_hint(query)
                suffix = _hint_to_suffix[hint]
                if question_type == "multi_choice" and choices:
                    parts = question_text.split("\nChoices:\n", 1)
                    if len(parts) == 2:
                        question_text = parts[0] + "\n\n" + suffix + "\nChoices:\n" + parts[1]
                    else:
                        question_text = question_text + "\n\n" + suffix
                    prompt = VQA_MCQ_TEMPLATE.format(Question=question_text)
                else:
                    prompt = VQA_OPEN_TEMPLATE.format(Question=question_text + suffix)
                item = {
                    "question": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _pil_to_base64(ex["decoded_image"])}},
                        {"type": "text", "text": prompt},
                    ],
                    "answer": str(ex["answer"]),
                    "question_type": question_type,
                    "choices": choices,
                    "precision": ex.get("precision"),
                }
                data.append(item)
            # Write cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info("mathvista cache written: %s (%d items)", cache_path, len(data))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return data


def _pil_to_base64(img: Any) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _strip_mathvista_hint(query: str) -> Tuple[str, str]:
    """Strip the leading Hint line from a MathVista query.

    Raises ValueError if the first line is not a known MathVista hint prefix,
    so that unexpected formats are caught immediately at data-loading time.

    Returns (question_text_without_hint, matched_hint_prefix).
    """
    first_line = query.split("\n")[0]
    if first_line not in _MATHVISTA_HINT_PREFIXES:
        raise ValueError(
            f"MathVista query starts with an unrecognised hint prefix: {first_line!r}\n"
            f"Known prefixes:\n" + "\n".join(f"  {p!r}" for p in _MATHVISTA_HINT_PREFIXES)
        )
    # Strip hint line + the following newline
    return query[len(first_line) + 1:].strip(), first_line


def _judge_mathvista(text: str, item: Dict[str, Any], evaluator: "RuleBasedMathEvaluator") -> Tuple[bool, str]:
    """Judge a MathVista response against the ground truth.

    For MCQ: extracts letter from \\boxed{}, maps to choice text, then uses
    math_verify to compare (handles 145 vs 145° etc.).
    For free_form float with precision: round comparison.
    For free_form integer/list: delegates to RuleBasedMathEvaluator (math_verify).
    """
    extracted = _fallback_extract_answer(text)
    question_type = item.get("question_type", "free_form")
    choices: List[str] = item.get("choices") or []
    gold = str(item["answer"])

    if question_type == "multi_choice" and choices:
        labels = ["A", "B", "C", "D", "E"]
        letter = extracted.strip().upper().strip("()")
        if letter in labels and labels.index(letter) < len(choices):
            predicted = choices[labels.index(letter)]
        else:
            predicted = extracted
        return evaluator.judge(predicted, gold)

    # free_form float with precision: round then compare
    precision = item.get("precision")
    if precision is not None:
        try:
            pred_val = float(extracted)
            gold_val = float(gold)
            p = int(precision)
            ok = round(pred_val, p) == round(gold_val, p)
            return ok, str(round(pred_val, p))
        except Exception:
            pass

    # free_form integer / list / fallback: math_verify
    return evaluator.judge(text, gold)


def _strip_images_from_question(question: Any) -> Any:
    """Remove base64 image data from a question before saving to disk.

    For multimodal questions (list of content parts), replaces image_url items
    with a lightweight placeholder so the saved JSONL stays small.
    Plain string questions are returned unchanged.
    """
    if not isinstance(question, list):
        return question
    stripped = []
    for part in question:
        if isinstance(part, dict) and part.get("type") == "image_url":
            stripped.append({"type": "image_url", "image_url": {"url": "<image_stripped>"}})
        else:
            stripped.append(part)
    return stripped


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)

async def _aiohttp_post_json(
    session: "aiohttp.ClientSession", url: str, payload: Dict[str, Any], timeout_s: float
) -> Any:
    timeout = aiohttp.ClientTimeout(total=float(timeout_s))
    async with session.post(url, json=payload, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {text}")
        return json.loads(text)


def _extract_after_think(text: str) -> str:
    # Soft-Thinking style: keep content after </think> if present.
    m = re.search(r"</think>(.*)", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def _fallback_extract_answer(text: str) -> str:
    """
    Best-effort, dependency-free extraction:
    - last \\boxed{...}
    - after 'Answer:'
    - last number-like token
    """
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
    # numeric compare when possible
    try:
        return float(pred) == float(gold)
    except Exception:
        pass
    # normalize whitespace
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

        # Use math_verify parse/verify (mirrors Soft-Thinking's approach).
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


def _get_choice_text(choice: Dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    if isinstance(msg, dict):
        content = msg.get("content")
        if content is not None:
            return str(content)
    # fallback (some servers may put text directly)
    txt = choice.get("text")
    return "" if txt is None else str(txt)

def _extract_usage(usage: Any) -> Dict[str, Optional[int]]:
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    def to_int(x: Any) -> Optional[int]:
        try:
            return int(x) if x is not None else None
        except Exception:
            return None
    return {
        "prompt_tokens": to_int(usage.get("prompt_tokens")),
        "completion_tokens": to_int(usage.get("completion_tokens")),
        "total_tokens": to_int(usage.get("total_tokens")),
    }


def build_chat_payload(
    question: Any,
    *,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    n: int,
    value_scale: Optional[float],
    value_mode: Optional[str],
    value_entropy_threshold: Optional[float],
    value_gamma: Optional[float],
    value_min: Optional[float],
    boosted_token_ids: Optional[List[int]],
    token_temp_divisor: Optional[float],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "n": int(n),
    }
    # Only include custom_params when explicitly provided.
    custom_params: Dict[str, Any] = {}
    if value_scale is not None:
        custom_params["value_scale"] = float(value_scale)
    if value_mode is not None:
        custom_params["value_mode"] = str(value_mode)
    if value_entropy_threshold is not None:
        custom_params["value_entropy_threshold"] = float(value_entropy_threshold)
    if value_gamma is not None:
        # Used by server-side mode="length_mul" to convert value <-> predicted length.
        custom_params["gamma"] = float(value_gamma)
    if value_min is not None:
        # Skip guidance when current E[v] is below this threshold.
        custom_params["value_min"] = float(value_min)
    if boosted_token_ids is not None and token_temp_divisor is not None:
        custom_params["boosted_token_ids"] = boosted_token_ids
        custom_params["token_temp_divisor"] = float(token_temp_divisor)
    if custom_params:
        payload["custom_params"] = custom_params
    return payload


async def _post_with_retries(
    post_json: Callable[[str, Dict[str, Any], float], Awaitable[Any]],
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    max_retries: int,
    base_sleep_s: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    for attempt in range(max_retries + 1):
        try:
            resp = await post_json(url, payload, timeout_s)
            if isinstance(resp, dict):
                return resp, None
            return {"_raw": resp}, None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            err = f"HTTP {e.code}: {body}"
        except RuntimeError as e:
            # aiohttp backend raises RuntimeError("HTTP {status}: body") on non-2xx
            err = str(e)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        if attempt >= max_retries:
            return None, err
        sleep_s = base_sleep_s * (2**attempt) + random.random() * 0.1
        await asyncio.sleep(sleep_s)
    return None, "UNREACHABLE"


@dataclass
class RunPaths:
    responses_jsonl: Path
    per_question_jsonl: Path
    summary_json: Path


def compute_paths(output_dir: Path, dataset_name: str, tag: str) -> RunPaths:
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}.{tag}".strip(".")
    return RunPaths(
        responses_jsonl=out_dir / f"{stem}.responses.jsonl",
        per_question_jsonl=out_dir / f"{stem}.per_question.jsonl",
        summary_json=out_dir / f"{stem}.summary.json",
    )


async def run_requests(
    data: List[Dict[str, str]],
    *,
    out_path: Path,
    skip_idxs: Optional[set[int]],
    http_backend: str,
    server_url: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    n: int,
    value_scale: Optional[float],
    value_mode: Optional[str],
    value_entropy_threshold: Optional[float],
    value_gamma: Optional[float],
    value_min: Optional[float],
    boosted_token_ids: Optional[List[int]],
    token_temp_divisor: Optional[float],
    max_concurrency: int,
    request_timeout: float,
    max_retries: int,
    base_sleep_s: float,
    max_questions: int,
) -> None:
    url = server_url.rstrip("/") + "/v1/chat/completions"
    started = time.time()

    items = data if int(max_questions) <= 0 else data[: int(max_questions)]
    if skip_idxs:
        items = [it for i, it in enumerate(items) if i not in skip_idxs]
    logger.info(
        "run: n_questions=%d max_concurrency=%d n=%d value_scale=%s value_mode=%s "
        "token_temp_divisor=%s n_boosted_tokens=%s out=%s",
        len(items),
        int(max_concurrency),
        int(n),
        value_scale,
        value_mode,
        token_temp_divisor,
        len(boosted_token_ids) if boosted_token_ids is not None else None,
        out_path,
    )

    queue: asyncio.Queue[Tuple[int, Dict[str, str]]] = asyncio.Queue()
    # Preserve original dataset indices for resume/traceability.
    if int(max_questions) <= 0:
        base_items = data
    else:
        base_items = data[: int(max_questions)]
    for i, it in enumerate(base_items):
        if skip_idxs and i in skip_idxs:
            continue
        queue.put_nowait((i, it))

    total_requests = queue.qsize()
    pbar = None
    pbar_lock = asyncio.Lock()
    if _TQDM_OK:
        pbar = tqdm(total=total_requests, desc="requests", unit="req")

    # Choose HTTP backend.
    # - urllib: uses asyncio.to_thread -> threadpool (compat, no extra deps)
    # - aiohttp: pure async + connection pooling (no threadpool)
    session: Optional["aiohttp.ClientSession"]
    if http_backend == "aiohttp":
        if not _AIOHTTP_OK:
            raise RuntimeError(
                "aiohttp is not installed but --http-backend=aiohttp was requested"
            )
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        session = aiohttp.ClientSession(connector=connector)

        async def post_json(u: str, p: Dict[str, Any], t: float) -> Any:
            assert session is not None
            return await _aiohttp_post_json(session, u, p, t)

    else:
        session = None

        async def post_json(u: str, p: Dict[str, Any], t: float) -> Any:
            return await asyncio.to_thread(_http_post_json, u, p, t)

    async def worker(worker_id: int) -> None:
        while True:
            try:
                i, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            q = item["question"]
            q_save = _strip_images_from_question(q)
            gold = item["answer"]
            payload = build_chat_payload(
                q,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                n=n,
                value_scale=value_scale,
                value_mode=value_mode,
                value_entropy_threshold=value_entropy_threshold,
                value_gamma=value_gamma,
                value_min=value_min,
                boosted_token_ids=boosted_token_ids,
                token_temp_divisor=token_temp_divisor,
            )
            # Build a save-safe copy of the payload with image data stripped out.
            payload_save = {
                **payload,
                "messages": [
                    {**msg, "content": _strip_images_from_question(msg["content"])}
                    if msg.get("role") == "user" else msg
                    for msg in payload.get("messages", [])
                ],
            }
            t0 = time.time()
            resp, err = await _post_with_retries(
                post_json,
                url=url,
                payload=payload,
                timeout_s=float(request_timeout),
                max_retries=int(max_retries),
                base_sleep_s=float(base_sleep_s),
            )
            dt = time.time() - t0
            if err is not None or resp is None:
                logger.warning("request failed idx=%d worker=%d err=%s", i, worker_id, err)
                append_jsonl(
                    out_path,
                    {
                        "idx": i,
                        "question": q_save,
                        "answer": gold,
                        "request": payload_save,
                        "error": err,
                        "elapsed_s": dt,
                    },
                )
                continue

            choices = resp.get("choices")
            usage = resp.get("usage")
            usage_slim = _extract_usage(usage)
            if not isinstance(choices, list) or not choices:
                logger.warning("no choices idx=%d worker=%d", i, worker_id)
                append_jsonl(
                    out_path,
                    {
                        "idx": i,
                        "question": q_save,
                        "answer": gold,
                        "request": payload_save,
                        "usage": usage_slim,
                        "response": resp,
                        "error": "NO_CHOICES",
                        "elapsed_s": dt,
                    },
                )
                continue

            # Save one JSONL line per choice, so eval can compute both pass@1 and pass@any.
            for j, ch in enumerate(choices):
                append_jsonl(
                    out_path,
                    {
                        "idx": i,
                        "choice_idx": j,
                        "question": q_save,
                        "answer": gold,
                        "request": payload_save,
                        "usage": usage_slim,
                        "choice": ch,
                        "text": _get_choice_text(ch),
                        "elapsed_s": dt,
                    },
                )
            if pbar is not None:
                async with pbar_lock:
                    pbar.update(1)
            else:
                if (i + 1) % 50 == 0:
                    logger.info("progress: %d/%d", min(i + 1, total_requests), total_requests)

    n_workers = min(max(1, int(max_concurrency)), queue.qsize() if queue.qsize() > 0 else 1)
    workers = [asyncio.create_task(worker(w)) for w in range(n_workers)]
    try:
        await asyncio.gather(*workers)
    finally:
        if session is not None:
            await session.close()
        if pbar is not None:
            pbar.close()
    total = time.time() - started
    logger.info("run done: wrote %s in %.1fs", out_path, total)


def eval_from_responses(
    responses_jsonl: Path,
    *,
    per_q_path: Path,
    summary_path: Path,
    use_math_verify: bool,
    pass_ks: List[int],
) -> None:
    evaluator = RuleBasedMathEvaluator(use_math_verify=use_math_verify)

    pass_ks = sorted({int(k) for k in pass_ks if int(k) > 0})
    if not pass_ks:
        pass_ks = [1]

    # group by idx
    by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for obj in iter_jsonl(responses_jsonl):
        idx = obj.get("idx")
        if not isinstance(idx, int):
            continue
        by_idx.setdefault(idx, []).append(obj)

    n_questions = 0
    n_has_any = 0
    n_has_first = 0
    n_correct_any = 0
    n_correct_first = 0
    # Token usage stats (sum over requests; usage is duplicated across choices in jsonl,
    # so we only count once per (idx) by taking choice_idx==0 when available).
    sum_prompt_tokens = 0
    sum_completion_tokens = 0
    sum_total_tokens = 0
    n_usage = 0
    # Per-choice normalized token usage (only meaningful if server reports usage as totals over n)
    sum_completion_tokens_per_choice = 0.0
    sum_total_tokens_per_choice = 0.0
    n_usage_per_choice = 0

    # pass@k stats
    # - pass_at_k_expected: standard estimator from "n evaluated samples" with c correct:
    #     pass@k = 1 - C(n-c, k) / C(n, k)
    #   (equivalent to probability that at least one correct appears in a uniformly
    #    random subset of size k from the n samples)
    # - pass_at_k_firstk: empirical "any correct in the first k choices" (order-sensitive)
    sum_pass_k_expected: Dict[int, float] = {k: 0.0 for k in pass_ks}
    sum_pass_k_firstk: Dict[int, float] = {k: 0.0 for k in pass_ks}

    # overwrite per_q output
    if per_q_path.exists():
        per_q_path.unlink()

    def pass_at_k_expected(n: int, c: int, k: int) -> float:
        n = int(n)
        c = int(c)
        k = int(k)
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
            # Fallback: compute via product to reduce overflow risk (still exact in float domain)
            # C(a,k)/C(b,k) = Π_{t=0..k-1} (a-t)/(b-t)
            a = n - c
            b = n
            ratio = 1.0
            for t in range(k):
                ratio *= (a - t) / (b - t)
            return 1.0 - ratio

    for idx in sorted(by_idx.keys()):
        rows = by_idx[idx]
        # skip pure error-only rows
        ok_rows = [r for r in rows if r.get("error") is None and r.get("text") is not None]
        if not ok_rows:
            continue
        n_questions += 1

        gold = str(ok_rows[0].get("answer", ""))
        # Stable choice order by choice_idx if present.
        ok_rows_sorted = sorted(
            ok_rows, key=lambda r: int(r.get("choice_idx")) if r.get("choice_idx") is not None else 0
        )
        texts = [str(r.get("text", "")) for r in ok_rows_sorted]

        judged: List[Tuple[bool, str]] = []
        for t in texts:
            # Use MathVista-specific judge when question_type metadata is present
            if ok_rows_sorted[0].get("question_type") is not None:
                judged.append(_judge_mathvista(t, ok_rows_sorted[0], evaluator))
            else:
                judged.append(evaluator.judge(t, gold))
        correct_flags = [x[0] for x in judged]
        c = int(sum(1 for x in correct_flags if x))
        n_seen = int(len(correct_flags))

        # usage: prefer the first choice line for this idx
        usage_row = None
        for r in ok_rows_sorted:
            if r.get("choice_idx") == 0:
                usage_row = r
                break
        if usage_row is None:
            usage_row = ok_rows_sorted[0]
        usage_slim = _extract_usage(usage_row.get("usage"))

        # Requested number of choices (n) for this request, for per-choice normalization.
        req_n = 1
        try:
            req = usage_row.get("request") or {}
            if isinstance(req, dict) and req.get("n") is not None:
                req_n = max(1, int(req.get("n")))
        except Exception:
            req_n = 1

        if usage_slim["prompt_tokens"] is not None:
            sum_prompt_tokens += int(usage_slim["prompt_tokens"])
        if usage_slim["completion_tokens"] is not None:
            sum_completion_tokens += int(usage_slim["completion_tokens"])
        if usage_slim["total_tokens"] is not None:
            sum_total_tokens += int(usage_slim["total_tokens"])
        if any(v is not None for v in usage_slim.values()):
            n_usage += 1

        # If usage exists, also compute per-choice-normalized numbers.
        # Note: some servers may already report per-choice usage; treat this as a derived metric.
        if usage_slim["completion_tokens"] is not None:
            sum_completion_tokens_per_choice += float(usage_slim["completion_tokens"]) / float(req_n)
            n_usage_per_choice += 1
        if usage_slim["total_tokens"] is not None:
            sum_total_tokens_per_choice += float(usage_slim["total_tokens"]) / float(req_n)

        pass_k_expected_row: Dict[int, float] = {}
        pass_k_firstk_row: Dict[int, float] = {}
        for k in pass_ks:
            k_eff = min(int(k), n_seen)
            pk_exp = pass_at_k_expected(n_seen, c, k_eff)
            pk_first = 1.0 if any(correct_flags[:k_eff]) else 0.0
            pass_k_expected_row[int(k)] = float(pk_exp)
            pass_k_firstk_row[int(k)] = float(pk_first)
            sum_pass_k_expected[int(k)] += float(pk_exp)
            sum_pass_k_firstk[int(k)] += float(pk_first)

        has_any = len(correct_flags) > 0
        has_first = len(correct_flags) > 0  # first available choice line
        if has_any:
            n_has_any += 1
            if any(correct_flags):
                n_correct_any += 1
        if has_first:
            n_has_first += 1
            if correct_flags[0]:
                n_correct_first += 1

        append_jsonl(
            per_q_path,
            {
                "idx": idx,
                "answer": gold,
                "n_choices_seen": len(texts),
                "n_correct": c,
                "correct_first": bool(correct_flags[0]) if correct_flags else False,
                "correct_any": bool(any(correct_flags)) if correct_flags else False,
                "extracted_first": judged[0][1] if judged else "",
                "usage": usage_slim,
                "usage_per_choice_assuming_total_over_n": {
                    "request_n": req_n,
                    "completion_tokens_per_choice": (
                        (float(usage_slim["completion_tokens"]) / float(req_n))
                        if usage_slim["completion_tokens"] is not None
                        else None
                    ),
                    "total_tokens_per_choice": (
                        (float(usage_slim["total_tokens"]) / float(req_n))
                        if usage_slim["total_tokens"] is not None
                        else None
                    ),
                },
                "pass_at_k_expected": pass_k_expected_row,
                "pass_at_k_firstk": pass_k_firstk_row,
            },
        )

    summary = {
        "responses_jsonl": str(responses_jsonl),
        "per_question_jsonl": str(per_q_path),
        "n_questions_used": n_questions,
        "acc_first": (n_correct_first / n_has_first) if n_has_first else 0.0,
        "acc_any": (n_correct_any / n_has_any) if n_has_any else 0.0,
        "use_math_verify": bool(use_math_verify and _MATH_VERIFY_OK),
        "pass_ks": pass_ks,
        "pass_at_k_expected": {
            str(k): (sum_pass_k_expected[k] / n_questions) if n_questions else 0.0 for k in pass_ks
        },
        "pass_at_k_firstk": {
            str(k): (sum_pass_k_firstk[k] / n_questions) if n_questions else 0.0 for k in pass_ks
        },
        "token_usage": {
            "n_requests_with_usage": n_usage,
            "sum_prompt_tokens": sum_prompt_tokens,
            "sum_completion_tokens": sum_completion_tokens,
            "sum_total_tokens": sum_total_tokens,
            "avg_prompt_tokens": (sum_prompt_tokens / n_usage) if n_usage else 0.0,
            "avg_completion_tokens": (sum_completion_tokens / n_usage) if n_usage else 0.0,
            "avg_total_tokens": (sum_total_tokens / n_usage) if n_usage else 0.0,
            "avg_completion_tokens_per_choice_assuming_total_over_n": (
                (sum_completion_tokens_per_choice / n_usage_per_choice) if n_usage_per_choice else 0.0
            ),
            "avg_total_tokens_per_choice_assuming_total_over_n": (
                (sum_total_tokens_per_choice / n_usage_per_choice) if n_usage_per_choice else 0.0
            ),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("eval done: wrote %s (n_questions_used=%d)", summary_path, n_questions)


def compute_pass_ks_pow2_from_responses(responses_jsonl: Path) -> List[int]:
    counts: Dict[int, int] = {}
    for obj in iter_jsonl(responses_jsonl):
        idx = obj.get("idx")
        if not isinstance(idx, int):
            continue
        if obj.get("error") is not None:
            continue
        if obj.get("text") is None:
            continue
        counts[idx] = counts.get(idx, 0) + 1
    max_n_seen = max(counts.values()) if counts else 0
    if max_n_seen <= 0:
        return [1]
    pass_ks: List[int] = []
    k = 1
    while k <= max_n_seen:
        pass_ks.append(k)
        k *= 2
    return pass_ks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", type=str, required=True, choices=["math500", "gsm8k", "aime", "amc23", "mathvista"])
    p.add_argument("--server-url", type=str, required=True, help="Base URL, e.g. http://127.0.0.1:30011")
    p.add_argument("--model", type=str, default="default")
    p.add_argument(
        "--system-prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--stage", type=str, default="all", choices=["run", "eval", "all"])
    p.add_argument("--max-questions", type=int, default=-1)
    p.add_argument(
        "--http-backend",
        type=str,
        default="urllib",
        choices=["urllib", "aiohttp"],
        help=(
            "HTTP client backend. 'urllib' uses threads (no deps). "
            "'aiohttp' is pure-async with connection pooling (no threadpool)."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help=(
            "If responses JSONL already exists, skip idxs that already have at least one "
            "successful record in the file (error=None and non-empty text). "
            "Use --overwrite-responses to force re-run."
        ),
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume; will append to existing responses JSONL (may create duplicates).",
    )
    p.add_argument(
        "--overwrite-responses",
        action="store_true",
        default=False,
        help="If set, delete existing responses JSONL before running.",
    )
    p.add_argument(
        "--resume-policy",
        type=str,
        default="success",
        choices=["success", "any"],
        help=(
            "How to decide an idx is already done when --resume is enabled. "
            "'success': skip only idxs that already have at least one successful choice "
            "(error is None and text is non-empty). "
            "'any': skip idxs that have any record (including errors)."
        ),
    )

    # Sampling / request params (mirror run.sh example)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--n", type=int, default=1)

    # Optional value-guidance params. If omitted, do NOT include custom_params in request.
    p.add_argument("--value-scale", type=float, default=None)
    p.add_argument(
        "--value-mode",
        type=str,
        default=None,
        choices=[None, "mul", "exp", "linear", "length_mul", "centered_exp", "value_bias"],
        help=(
            "If set, include custom_params.value_mode. "
            "'length_mul' scales the expected remaining-length proxy (via gamma mapping) before re-targeting E[value]. "
            "'centered_exp' applies exp(s * (g - E[g])) reweighting with s=value_scale. "
            "'value_bias' converts sigmoid values back to logit space and uses as bias: logits' = log(p) + logit(v) * scale."
        ),
    )
    p.add_argument(
        "--value-gamma",
        type=float,
        default=None,
        help=(
            "If set, include custom_params.gamma used by server-side mode='length_mul' "
            "to convert value <-> predicted remaining length. "
            "If omitted, server uses its default (currently 0.997)."
        ),
    )
    p.add_argument(
        "--value-entropy-threshold",
        type=float,
        default=None,
        help=(
            "If set, include custom_params.value_entropy_threshold (nats). "
            "Server-side LVM guidance may skip /tree_value when candidate distribution entropy is <= threshold."
        ),
    )
    p.add_argument(
        "--value-min",
        type=float,
        default=None,
        help=(
            "If set, include custom_params.value_min. "
            "Server-side LVM guidance is skipped (distribution unchanged) when the current "
            "expected value E[v] is below this threshold."
        ),
    )

    # Token temperature scaling params.
    p.add_argument(
        "--token-temp-divisor",
        type=float,
        default=None,
        help=(
            "If set, apply per-token temperature scaling for boosted tokens. "
            "divisor > 1 → tokens more likely; divisor < 1 → less likely. "
            "Requires --token-temp-scale-file."
        ),
    )
    p.add_argument(
        "--token-temp-scale-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file (list of objects with 'token_id' and 'count_total'). "
            "Tokens with count_total > --token-temp-min-count are used as boosted_token_ids."
        ),
    )
    p.add_argument(
        "--token-temp-min-count",
        type=int,
        default=100,
        help="Minimum count_total threshold for selecting tokens from --token-temp-scale-file (default: 100).",
    )

    # Concurrency / resilience
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument(
        "--threadpool-size",
        type=int,
        default=0,
        help=(
            "Max worker threads for HTTP (used by asyncio.to_thread + urllib). "
            "If 0, use Python default (~min(32, cpu+4)). "
            "Effective concurrency is roughly min(max_concurrency, threadpool_size)."
        ),
    )
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=0.5)

    # Eval knobs
    p.add_argument("--no-math-verify", action="store_true", default=False)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overall_t0 = time.time()
    run_dt_s: Optional[float] = None
    eval_dt_s: Optional[float] = None
    out_dir = Path(args.output_dir)
    paths = compute_paths(out_dir, args.dataset_name, args.tag)

    # Load boosted token IDs from file if provided.
    boosted_token_ids: Optional[List[int]] = None
    if args.token_temp_scale_file is not None:
        with open(args.token_temp_scale_file, "r", encoding="utf-8") as f:
            token_entries = json.load(f)
        min_count = int(args.token_temp_min_count)
        boosted_token_ids = [
            int(e["token_id"])
            for e in token_entries
            if int(e.get("count_total", 0)) > min_count
        ]
        logger.info(
            "loaded %d boosted tokens (count_total > %d) from %s",
            len(boosted_token_ids),
            min_count,
            args.token_temp_scale_file,
        )

    if args.stage in ("run", "all"):
        run_t0 = time.time()
        if args.overwrite_responses and paths.responses_jsonl.exists():
            logger.warning("overwrite enabled: deleting existing %s", paths.responses_jsonl)
            paths.responses_jsonl.unlink()

        skip_idxs: set[int] = set()
        if args.resume and paths.responses_jsonl.exists():
            try:
                seen_any: set[int] = set()
                seen_success: set[int] = set()
                for obj in iter_jsonl(paths.responses_jsonl):
                    idx = obj.get("idx")
                    if isinstance(idx, int):
                        seen_any.add(idx)
                        err = obj.get("error")
                        txt = obj.get("text")
                        if err is None and txt is not None and str(txt).strip() != "":
                            seen_success.add(idx)

                if str(args.resume_policy) == "any":
                    skip_idxs = seen_any
                else:
                    # Default: retry idxs that only have errors/no usable text.
                    skip_idxs = seen_success
                logger.info(
                    "resume enabled (policy=%s): found %d/%d done idxs in %s",
                    str(args.resume_policy),
                    len(skip_idxs),
                    len(seen_any),
                    paths.responses_jsonl,
                )
            except Exception as e:
                logger.warning("resume scan failed (%s); proceeding without resume", e)
                skip_idxs = set()

        data = load_dataset_and_template(args.dataset_name)
        async def _run() -> None:
            if int(args.threadpool_size) > 0:
                loop = asyncio.get_running_loop()
                loop.set_default_executor(
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=int(args.threadpool_size)
                    )
                )
                logger.info("using threadpool_size=%d", int(args.threadpool_size))

            await run_requests(
                data,
                out_path=paths.responses_jsonl,
                skip_idxs=skip_idxs,
                http_backend=str(args.http_backend),
                server_url=args.server_url,
                model=args.model,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                n=args.n,
                value_scale=args.value_scale,
                value_mode=args.value_mode,
                value_entropy_threshold=args.value_entropy_threshold,
                value_gamma=args.value_gamma,
                value_min=args.value_min,
                boosted_token_ids=boosted_token_ids,
                token_temp_divisor=args.token_temp_divisor,
                max_concurrency=args.max_concurrency,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                base_sleep_s=args.retry_sleep,
                max_questions=args.max_questions,
            )

        asyncio.run(_run())
        run_dt_s = time.time() - run_t0
        logger.info("stage run done: elapsed=%.3fs", run_dt_s)

    if args.stage in ("eval", "all"):
        eval_t0 = time.time()
        if not paths.responses_jsonl.exists():
            raise FileNotFoundError(f"Missing responses file: {paths.responses_jsonl}")
        pass_ks = compute_pass_ks_pow2_from_responses(paths.responses_jsonl)
        eval_from_responses(
            paths.responses_jsonl,
            per_q_path=paths.per_question_jsonl,
            summary_path=paths.summary_json,
            use_math_verify=not args.no_math_verify,
            pass_ks=pass_ks,
        )
        eval_dt_s = time.time() - eval_t0
        logger.info("stage eval done: elapsed=%.3fs", eval_dt_s)

    overall_dt_s = time.time() - overall_t0
    timing = {
        "stage": str(args.stage),
        "dataset_name": str(args.dataset_name),
        "tag": str(args.tag),
        "output_dir": str(out_dir),
        "elapsed_total_s": float(overall_dt_s),
        "elapsed_run_s": float(run_dt_s) if run_dt_s is not None else None,
        "elapsed_eval_s": float(eval_dt_s) if eval_dt_s is not None else None,
    }
    # Persist timing into summary.json (single-file convenience).
    try:
        if paths.summary_json.exists():
            summary_obj = json.loads(paths.summary_json.read_text(encoding="utf-8"))
            if not isinstance(summary_obj, dict):
                summary_obj = {}
        else:
            summary_obj = {}
        summary_obj["timing"] = timing
        paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
        paths.summary_json.write_text(
            json.dumps(summary_obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("failed to write timing into summary (%s): %s", paths.summary_json, e)
    logger.info(
        "total done: elapsed_total=%.3fs (run=%s eval=%s); wrote summary=%s",
        overall_dt_s,
        f"{run_dt_s:.3f}s" if run_dt_s is not None else "N/A",
        f"{eval_dt_s:.3f}s" if eval_dt_s is not None else "N/A",
        paths.summary_json,
    )


if __name__ == "__main__":
    main()