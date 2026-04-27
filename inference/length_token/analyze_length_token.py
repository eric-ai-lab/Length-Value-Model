from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _sigmoid_stable(x: float) -> float:
    # Same as eval_first_token_prediction.py (avoid overflow).
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def value_pred_to_length(value_pred: float, gamma: float, eps: float) -> Tuple[float, float]:
    """
    Matches eval_lvm/first_token_prediction/eval_first_token_prediction.py:
      y_hat = -sigmoid(value_pred)   -> y_hat in (-1, 0)
      length = ln(1 + y_hat) / ln(gamma)
    """
    if not (0.0 < float(gamma) < 1.0):
        raise ValueError(f"gamma must be in (0,1), got {gamma}")
    eps = float(eps)
    x = float(value_pred)
    y_hat = -float(_sigmoid_stable(x))
    y = min(-eps, max(-1.0 + eps, y_hat))
    length = math.log1p(y) / min(-eps, math.log(float(gamma)))
    return y_hat, float(length)


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            yield line_no, json.loads(s)


def sanitize_tag(s: str) -> str:
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
    """Online mean/std using Welford (population std by default)."""

    def __init__(self) -> None:
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


class ReservoirSampler:
    """Uniform reservoir sampling for streaming quantiles.

    k semantics:
    - k < 0: keep ALL samples (unbounded, exact quantiles from full buffer)
    - k = 0: disabled (collect nothing)
    - k > 0: classic reservoir of size k
    """

    def __init__(self, k: int, seed: int = 42) -> None:
        self.k = int(k)
        self.rng = random.Random(int(seed))
        self.n_seen = 0
        self.buf: List[float] = []

    def update(self, x: float) -> None:
        if self.k == 0:
            return
        if self.k < 0:
            self.buf.append(float(x))
            self.n_seen += 1
            return
        self.n_seen += 1
        if len(self.buf) < self.k:
            self.buf.append(float(x))
            return
        j = self.rng.randrange(self.n_seen)
        if j < self.k:
            self.buf[j] = float(x)

    def get_quantile(self, q: float) -> float:
        if not self.buf:
            return 0.0
        qq = float(min(1.0, max(0.0, q)))
        xs = sorted(self.buf)
        idx = int(round(qq * (len(xs) - 1)))
        return float(xs[idx])


@dataclass
class PreparedSample:
    lenvm_idx: str
    line_no: int
    meta: Dict[str, Any]
    input_ids: List[int]
    assistant_start: int
    assistant_end: int  # exclusive

    # Optional debug fields
    user_text: str = ""


def conversations_to_messages(conv: Any) -> List[Dict[str, str]]:
    if not isinstance(conv, list) or not conv:
        raise ValueError("sample.conversations missing or invalid")
    out: List[Dict[str, str]] = []
    for item in conv:
        if not isinstance(item, dict):
            continue
        src = item.get("from")
        val = str(item.get("value", ""))
        if src in ("system",):
            out.append({"role": "system", "content": val})
        elif src in ("human", "user"):
            out.append({"role": "user", "content": val})
        elif src in ("gpt", "assistant"):
            out.append({"role": "assistant", "content": val})
    if not out:
        raise ValueError("no valid messages parsed from conversations")
    return out


def extract_user_text(conv: Any) -> str:
    if not isinstance(conv, list):
        return ""
    for item in conv:
        if isinstance(item, dict) and item.get("from") in ("human", "user"):
            return str(item.get("value", ""))
    return ""


def prepare_sample(
    tokenizer,
    obj: Dict[str, Any],
    line_no: int,
) -> PreparedSample:
    meta = obj.get("meta_info") or {}
    idx = meta.get("lenvm_idx", meta.get("index"))
    if idx is None:
        raise ValueError("missing meta_info.lenvm_idx")
    lenvm_idx = str(idx)

    conv = obj.get("conversations")
    messages = conversations_to_messages(conv)
    # Find last assistant message as the generated answer.
    last_assistant_idx: Optional[int] = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_assistant_idx = i
    if last_assistant_idx is None:
        raise ValueError("cannot find assistant message in conversations")
    prefix_messages = messages[:last_assistant_idx]
    assistant_message = messages[last_assistant_idx]

    def _extract_input_ids(x: Any) -> List[int]:
        # transformers>=5 may return BatchEncoding instead of a plain list.
        if isinstance(x, list):
            return [int(t) for t in x]
        if isinstance(x, dict) and "input_ids" in x:
            return [int(t) for t in x["input_ids"]]
        # BatchEncoding is dict-like
        try:
            ids = x["input_ids"]  # type: ignore[index]
            if isinstance(ids, list):
                return [int(t) for t in ids]
        except Exception:
            pass
        raise RuntimeError(f"Unexpected apply_chat_template return type: {type(x)}")

    prompt_ret = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ret = tokenizer.apply_chat_template(
        prefix_messages + [assistant_message],
        tokenize=True,
        add_generation_prompt=False,
    )

    prompt_ids = _extract_input_ids(prompt_ret)
    full_ids = _extract_input_ids(full_ret)
    if not prompt_ids or not full_ids:
        raise RuntimeError(
            "tokenizer.apply_chat_template returned empty input_ids "
            f"(prefix_roles={[m.get('role') for m in prefix_messages]}, "
            f"assistant_len={len(str(assistant_message.get('content', '')))})."
        )

    assistant_start = len(prompt_ids)
    if full_ids[:assistant_start] != prompt_ids:
        raise RuntimeError(
            "Chat template tokenization mismatch: full_ids does not start with prompt_ids "
            "(add_generation_prompt behavior differs)."
        )
    if assistant_start <= 0:
        raise RuntimeError("assistant_start <= 0 (no prompt-last token available)")

    assistant_ids = full_ids[assistant_start:]
    assistant_end = assistant_start + len(assistant_ids)
    if assistant_end <= assistant_start:
        raise RuntimeError("assistant token span is empty")

    # Truncate input_ids to only what we need (causal LM: earlier token outputs should not depend on future tokens).
    input_end = assistant_end
    input_ids = full_ids[:input_end]

    return PreparedSample(
        lenvm_idx=lenvm_idx,
        line_no=int(line_no),
        meta=dict(meta) if isinstance(meta, dict) else {},
        input_ids=input_ids,
        assistant_start=int(assistant_start),
        assistant_end=int(assistant_end),
        user_text=extract_user_text(conv),
    )


def encode_batch_tokenwise_values(
    server_url: str,
    input_ids_batch: List[List[int]],
    timeout_s: float,
    retry_on_length_mismatch: bool = True,
) -> List[List[float]]:
    enc_url = server_url.rstrip("/") + "/encode"
    flush_url = server_url.rstrip("/") + "/flush_cache"

    def _flush_cache_retry_once() -> None:
        # Best-effort flush with one retry on failure.
        try:
            _http_post_text(flush_url, None, timeout_s=30.0)
            return
        except Exception:
            pass
        # Small backoff before retry to avoid immediate repeated 400/timeout bursts.
        time.sleep(1.0)
        _http_post_text(flush_url, None, timeout_s=30.0)

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

    out: List[List[float]] = []
    mismatch = False
    for seq_ids, item in zip(input_ids_batch, resp_list):
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Unexpected /encode embedding type: {type(emb)}")
        if len(emb) != len(seq_ids):
            mismatch = True
            break
        out.append([float(x) for x in emb])

    if mismatch:
        if retry_on_length_mismatch:
            # Best-effort flush then retry once.
            try:
                _flush_cache_retry_once()
            except Exception:
                pass
            return encode_batch_tokenwise_values(
                server_url=server_url,
                input_ids_batch=input_ids_batch,
                timeout_s=timeout_s,
                retry_on_length_mismatch=False,
            )
        raise RuntimeError("One or more /encode embedding length mismatches in batch.")

    return out


def short_text(s: str, limit: int) -> str:
    s = str(s)
    if limit <= 0:
        return ""
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def render_token_wordcloud_from_items(
    items: List[Dict[str, Any]],
    out_png: Path,
    weight_key: str,
    top_k: int,
    min_weight: float,
    min_count_total: int,
    max_count_total: int,
    include_special_tokens: bool,
    include_whitespace_tokens: bool,
    font_path: Optional[str],
) -> Dict[str, Any]:
    # Imported lazily so main stats pipeline still works without this dep.
    from wordcloud import WordCloud  # type: ignore

    freq: Dict[str, float] = {}
    picked = 0

    def _visualize_leading_spaces(s: str) -> str:
        # WordCloud may collapse/ignore leading spaces; make them visible.
        n = len(s) - len(s.lstrip(" "))
        if n <= 0:
            return s
        return ("_" * n) + s[n:]

    def _visualize_token(s: str) -> str:
        if _looks_like_emoji_or_symbol(s) and not any(ch.isalnum() for ch in s):
            return _emoji_token_alias(s)
        return s

    for it in items:
        tok = str(it.get("token_text", ""))
        if not tok:
            continue
        if not include_whitespace_tokens and tok.strip() == "":
            continue
        if not include_special_tokens and tok.startswith("<|") and tok.endswith("|>"):
            continue
        if int(it.get("count_total", 0)) < int(min_count_total):
            continue
        tok = tok.replace("\r", "\\r").replace("\n", "\\n")
        tok = _visualize_leading_spaces(tok)
        tok = _visualize_token(tok)
        if weight_key == "count_total":
            w = float(min(int(it.get("count_total", 0)), int(max_count_total)))
        else:
            w = float(it.get(weight_key, 0.0))
            if weight_key == "mean_score":
                w = abs(w)
        if w < float(min_weight):
            continue
        freq[tok] = freq.get(tok, 0.0) + w
        picked += 1
        if int(top_k) > 0 and picked >= int(top_k):
            break

    if not freq:
        raise RuntimeError(f"No tokens left after filtering for wordcloud: {out_png}")

    wc = WordCloud(
        width=2200,
        height=1200,
        background_color="white",
        collocations=False,
        max_words=max(200, len(freq)),
        font_path=font_path,
    ).generate_from_frequencies(freq)
    wc.to_file(str(out_png))
    out_pdf = out_png.with_suffix(".pdf")
    wc.to_image().save(str(out_pdf), "PDF", resolution=300.0)
    return {
        "path": str(out_png),
        "pdf_path": str(out_pdf),
        "num_words": int(len(freq)),
        "weight_key": str(weight_key),
        "font_path": str(font_path) if font_path else None,
    }


def _looks_like_emoji_or_symbol(s: str) -> bool:
    # WordCloud only uses one font, so we bias toward a symbol-capable font when
    # the token list contains emoji-like tokens such as "✅".
    for ch in s:
        cat = unicodedata.category(ch)
        if cat in {"So", "Sk"}:
            return True
        cp = ord(ch)
        if (
            0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0xFE0F == cp
        ):
            return True
    return False


def _emoji_token_alias(s: str) -> str:
    skip = {"white", "black", "heavy", "medium", "small", "emoji", "variation", "selector"}
    parts: List[str] = []
    for ch in s:
        if ch.isspace():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = f"u{ord(ch):04x}"
        words = [w for w in name.lower().replace("-", " ").split() if w not in skip]
        if not words:
            words = [f"u{ord(ch):04x}"]
        parts.append(words[0])
    if not parts:
        return "emoji"
    return "emoji_" + "_".join(parts[:1])


def auto_pick_wordcloud_font(items: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    # Prefer a general Unicode font for emoji/symbol-heavy token sets; otherwise
    # prefer common Linux CJK fonts.
    emoji_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    ]
    cjk_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]

    candidates = cjk_candidates
    if items is not None:
        for it in items:
            tok = str(it.get("token_text", ""))
            if tok and _looks_like_emoji_or_symbol(tok):
                candidates = emoji_candidates + cjk_candidates
                break

    for p in candidates:
        if Path(p).exists():
            return p
    return None


@dataclass
class TokenAgg:
    token_id: int
    token_text: str
    stats: RunningStats
    count_total: int = 0
    count_above_upper: int = 0
    count_below_lower: int = 0
    max_score: float = float("-inf")
    min_score: float = float("inf")
    sample_deltas: ReservoirSampler | None = None

    def update(self, score: float, lower_threshold: float, upper_threshold: float) -> None:
        self.count_total += 1
        self.stats.update(score)
        if score > upper_threshold:
            self.count_above_upper += 1
        if score < lower_threshold:
            self.count_below_lower += 1
        if score > self.max_score:
            self.max_score = float(score)
        if score < self.min_score:
            self.min_score = float(score)
        if self.sample_deltas is not None:
            self.sample_deltas.update(score)

    def to_dict(self) -> Dict[str, Any]:
        mean = float(self.stats.get_mean())
        std = float(self.stats.get_std(sample=False))
        mx = float(self.max_score if self.max_score != float("-inf") else 0.0)
        mn = float(self.min_score if self.min_score != float("inf") else 0.0)
        d: Dict[str, Any] = {
            "token_id": int(self.token_id),
            "token_text": str(self.token_text),
            "count_total": int(self.count_total),
            # Preferred (score-oriented) names:
            "count_score_above_upper": int(self.count_above_upper),
            "count_score_below_lower": int(self.count_below_lower),
            "mean_score": mean,
            "std_score": std,
            "max_score": mx,
            "min_score": mn,
            # Backward-compatible aliases:
            "count_score_gt_threshold": int(self.count_above_upper),
            "count_delta_gt_threshold": int(self.count_above_upper),
            "mean_delta": mean,
            "std_delta": std,
            "max_delta": mx,
        }
        if self.sample_deltas is not None and self.sample_deltas.buf:
            p95 = float(self.sample_deltas.get_quantile(0.95))
            d["approx_p95_score"] = p95
            d["approx_p95_delta"] = p95
        return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", type=str, default="http://127.0.0.1:30010")
    ap.add_argument("--model-dir", type=str, required=True, help="HF tokenizer directory")
    ap.add_argument("--data", type=str, required=True, help="Path to a *.jsonl or *.grouped.jsonl")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--dataset-name", type=str, default=None, help="Optional label for outputs")
    ap.add_argument("--model-name", type=str, default=None, help="Optional label for outputs")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument(
        "--score",
        type=str,
        default="value_gamma_td",
        choices=["value_gamma_td", "value_rel", "value_delta", "length_delta"],
        help=(
            "Scoring function per step (current token vs previous token). "
            "value_gamma_td (default): gamma*p_t - p_{t-1} + (1-gamma), where p=sigmoid(value_raw); "
            "value_rel: (p_t-p_{t-1})/max(|p_{t-1}|, rel_eps) (signed), where p=sigmoid(value_raw); "
            "value_delta: (p_t-p_{t-1}) (signed), where p=sigmoid(value_raw); "
            "length_delta: (L_t-L_{t-1}) (length derived from value head)."
        ),
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Deprecated symmetric threshold. If set and upper/lower absent: upper=+threshold, lower=-threshold.",
    )
    ap.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="Upper threshold: score > upper_threshold counts as increase.",
    )
    ap.add_argument(
        "--lower-threshold",
        type=float,
        default=None,
        help="Lower threshold: score < lower_threshold counts as decrease.",
    )
    ap.add_argument(
        "--rel-eps",
        type=float,
        default=1e-6,
        help="Denominator floor for value_rel score: denom=max(|v_prev|, rel_eps).",
    )
    ap.add_argument("--max-samples", type=int, default=200, help="Max JSONL records to process (-1 for all).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-concurrency", type=int, default=8, help="Max in-flight /encode requests.")
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument(
        "--offline-mock",
        action="store_true",
        default=False,
        help=(
            "Do not call the server. Instead generate deterministic fake /encode embeddings "
            "(useful for quickly sanity-checking tokenization/output plumbing)."
        ),
    )
    ap.add_argument(
        "--events-user-text-limit",
        type=int,
        default=200,
        help="Truncate user_text saved in events jsonl (chars).",
    )
    ap.add_argument(
        "--events-max-write",
        type=int,
        default=-1,
        help="Max number of threshold-crossing events to write (-1 for all).",
    )
    ap.add_argument(
        "--next-token-window",
        type=int,
        default=5,
        help="For each threshold event, collect next N tokens after current token (default: 5).",
    )
    ap.add_argument(
        "--token-delta-reservoir",
        type=int,
        default=-1,
        help=(
            "Per-token quantile buffer size. "
            "-1 keeps ALL samples (default), 0 disables, >0 uses reservoir of that size."
        ),
    )
    ap.add_argument(
        "--global-delta-reservoir",
        type=int,
        default=200000,
        help="Global score quantile reservoir size (0 disables).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-input", action="store_true", default=True, help="Shuffle input rows before processing.")
    ap.add_argument("--no-shuffle-input", dest="shuffle_input", action="store_false")
    ap.add_argument("--wordcloud", action="store_true", default=True, help="Generate wordcloud PNGs.")
    ap.add_argument("--no-wordcloud", dest="wordcloud", action="store_false")
    ap.add_argument("--wordcloud-top-k", type=int, default=200, help="Top K rows used for wordcloud.")
    ap.add_argument("--wordcloud-min-weight", type=float, default=0.0, help="Minimum token weight in wordcloud.")
    ap.add_argument(
        "--wordcloud-min-count-total",
        type=int,
        default=1,
        help="Only keep tokens whose count_total is at least this value.",
    )
    ap.add_argument(
        "--wordcloud-max-count-total",
        type=int,
        default=50,
        help="Cap count_total at this value before drawing sizes.",
    )
    ap.add_argument(
        "--wordcloud-include-special-tokens",
        action="store_true",
        default=False,
        help="Include special tokens like <|im_end|> in wordcloud.",
    )
    ap.add_argument(
        "--wordcloud-include-whitespace-tokens",
        action="store_true",
        default=False,
        help="Include whitespace-only tokens in wordcloud.",
    )
    ap.add_argument(
        "--wordcloud-font-path",
        type=str,
        default=None,
        help="Optional font path for rendering CJK tokens.",
    )
    ap.add_argument("--progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    ap.add_argument("--fail-fast", action="store_true", default=False)
    args = ap.parse_args()

    if AutoTokenizer is None:
        raise RuntimeError(
            "Missing dependency: transformers. Install it in your eval env, e.g. `pip install transformers` "
            f"(import error: {_TRANSFORMERS_IMPORT_ERROR})"
        )

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or data_path.stem
    model_name = sanitize_tag(args.model_name or Path(args.model_dir).name)
    if args.upper_threshold is None and args.lower_threshold is None and args.threshold is not None:
        upper_threshold = float(args.threshold)
        lower_threshold = -float(args.threshold)
    else:
        upper_threshold = float(args.upper_threshold) if args.upper_threshold is not None else 50.0
        lower_threshold = float(args.lower_threshold) if args.lower_threshold is not None else -50.0
    if not (lower_threshold < upper_threshold):
        raise ValueError(f"Require lower_threshold < upper_threshold, got {lower_threshold} >= {upper_threshold}")

    out_prefix = (
        f"{sanitize_tag(dataset_name)}.{model_name}.{sanitize_tag(args.score)}."
        f"up{sanitize_tag(upper_threshold)}.low{sanitize_tag(lower_threshold)}"
    )

    summary_path = out_dir / f"{out_prefix}.length_token_summary.json"
    tokens_by_count_path = out_dir / f"{out_prefix}.top_tokens_by_count_above_upper.json"
    tokens_by_count_low_path = out_dir / f"{out_prefix}.top_tokens_by_count_below_lower.json"
    tokens_by_mean_path = out_dir / f"{out_prefix}.top_tokens_by_mean_delta.json"
    tokens_by_mean_thr_above_path = out_dir / f"{out_prefix}.top_tokens_by_mean_score_above_upper.json"
    tokens_by_mean_thr_below_path = out_dir / f"{out_prefix}.top_tokens_by_mean_score_below_lower.json"
    token_stats_path = out_dir / f"{out_prefix}.token_stats.jsonl"
    events_path = out_dir / f"{out_prefix}.events_delta_gt_threshold.jsonl"
    failures_path = out_dir / f"{out_prefix}.failures.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)

    batch_size = max(1, int(args.batch_size))
    max_conc = max(1, int(args.max_concurrency))
    max_samples = int(args.max_samples)

    # Global stats (for the chosen score)
    n_lines_seen = 0
    n_samples_prepared = 0
    n_samples_succeeded = 0
    n_samples_failed_prepare = 0
    n_samples_failed_request = 0
    n_events_written = 0

    stat_score = RunningStats()
    global_score_sample = ReservoirSampler(k=int(args.global_delta_reservoir), seed=int(args.seed))

    token_aggs: Dict[int, TokenAgg] = {}
    token_text_cache: Dict[int, str] = {}
    # Key: trigger token id; Value: Counter over following-N-token tuples.
    next_seq_by_trigger_above: Dict[int, Counter[Tuple[int, ...]]] = {}
    next_seq_by_trigger_below: Dict[int, Counter[Tuple[int, ...]]] = {}
    # Key: trigger token id; Value: Counter over previous-N-token tuples.
    prev_seq_by_trigger_above: Dict[int, Counter[Tuple[int, ...]]] = {}
    prev_seq_by_trigger_below: Dict[int, Counter[Tuple[int, ...]]] = {}

    def get_tok_text(tok_id: int) -> str:
        t = token_text_cache.get(tok_id)
        if t is not None:
            return t
        s = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
        token_text_cache[int(tok_id)] = s
        return s

    def decode_token_seq(tok_ids: List[int]) -> str:
        if not tok_ids:
            return ""
        try:
            return tokenizer.decode([int(t) for t in tok_ids], skip_special_tokens=False)
        except Exception:
            # Fallback keeps output available even if tokenizer decode fails for a sequence.
            return "".join(get_tok_text(int(t)) for t in tok_ids)

    rng = random.Random(int(args.seed))

    def get_token_agg(tok_id: int) -> TokenAgg:
        tok_id = int(tok_id)
        agg = token_aggs.get(tok_id)
        if agg is not None:
            return agg
        sample_k = int(args.token_delta_reservoir)
        sampler = ReservoirSampler(k=sample_k, seed=rng.randrange(1 << 30)) if sample_k > 0 else None
        agg = TokenAgg(
            token_id=tok_id,
            token_text=get_tok_text(tok_id),
            stats=RunningStats(),
            sample_deltas=sampler,
        )
        token_aggs[tok_id] = agg
        return agg

    # Open outputs early (streaming).
    wf_events = events_path.open("w", encoding="utf-8")
    wf_fail = failures_path.open("w", encoding="utf-8")

    def record_failure(kind: str, line_no: int, lenvm_idx: str, err: str) -> None:
        wf_fail.write(
            json.dumps(
                {"kind": kind, "line_no": int(line_no), "lenvm_idx": str(lenvm_idx), "error": str(err)},
                ensure_ascii=False,
            )
            + "\n"
        )
        wf_fail.flush()
        if bool(args.fail_fast):
            raise RuntimeError(f"{kind}: line={line_no} lenvm_idx={lenvm_idx} err={err}")

    # Prepare + request pipeline with limited in-flight futures.
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    pending = set()
    fut_to_batch: Dict[Any, List[PreparedSample]] = {}

    def submit_batch(ex: ThreadPoolExecutor, batch: List[PreparedSample]) -> None:
        input_ids_batch = [b.input_ids for b in batch]

        def _run() -> List[List[float]]:
            if bool(args.offline_mock):
                # Deterministic pseudo "embedding" values for offline smoke tests.
                # Only shape matters; values just need to be finite floats.
                base_seed = int(args.seed)
                out: List[List[float]] = []
                for seq in input_ids_batch:
                    # Seeded per-sequence to keep stable across runs.
                    h = 1469598103934665603
                    for t in seq[:16]:
                        h ^= int(t) & 0xFFFFFFFF
                        h *= 1099511628211
                        h &= (1 << 64) - 1
                    rng2 = random.Random((base_seed ^ h) & 0xFFFFFFFF)
                    emb = []
                    for i, tok in enumerate(seq):
                        # Mix token id + position; keep magnitude moderate to avoid extreme sigmoid saturation.
                        x = 0.3 * math.sin(0.01 * float(tok) + 0.07 * float(i)) + 0.05 * rng2.uniform(-1.0, 1.0)
                        emb.append(float(x))
                    out.append(emb)
                return out

            return encode_batch_tokenwise_values(
                server_url=str(args.server_url),
                input_ids_batch=input_ids_batch,
                timeout_s=float(args.request_timeout),
            )

        fut = ex.submit(_run)
        pending.add(fut)
        fut_to_batch[fut] = batch

    def process_batch_result(batch: List[PreparedSample], embs: List[List[float]]) -> None:
        nonlocal n_samples_succeeded, n_events_written
        for ps, emb in zip(batch, embs):
            try:
                if len(emb) != len(ps.input_ids):
                    raise RuntimeError(f"/encode length mismatch in result: got={len(emb)} expected={len(ps.input_ids)}")

                start_idx = ps.assistant_start - 1
                end_idx = ps.assistant_end - 1
                if start_idx < 0 or end_idx <= start_idx or end_idx >= len(ps.input_ids):
                    raise RuntimeError(
                        f"invalid span: start_idx={start_idx} end_idx={end_idx} len={len(ps.input_ids)}"
                    )

                # Iterate assistant tokens and compute delta attributed to current token.
                # current idx goes from assistant_start .. end_idx
                for cur_idx in range(ps.assistant_start, end_idx + 1):
                    prev_idx = cur_idx - 1
                    v_prev_raw = float(emb[prev_idx])
                    v_cur_raw = float(emb[cur_idx])
                    eos_id = getattr(tokenizer, "eos_token_id", None)
                    if eos_id is not None:
                        if int(ps.input_ids[prev_idx]) == int(eos_id):
                            raise RuntimeError(
                                f"Unexpected prev EOS at line={ps.line_no}, lenvm_idx={ps.lenvm_idx}, "
                                f"assistant_step={cur_idx - ps.assistant_start}"
                            )
                        if int(ps.input_ids[cur_idx]) == int(eos_id):
                            v_cur_raw = 0.0

                    # Always map raw head outputs into value space (0,1) before value-based scoring.
                    p_prev = float(_sigmoid_stable(v_prev_raw))
                    p_cur = float(_sigmoid_stable(v_cur_raw))
                    dp = float(p_cur - p_prev)

                    score_name = str(args.score)
                    if score_name == "value_gamma_td":
                        score = float(float(args.gamma) * p_cur - p_prev + (1.0 - float(args.gamma)))
                    elif score_name == "value_rel":
                        denom = max(abs(p_prev), float(args.rel_eps))
                        score = dp / denom
                    elif score_name == "value_delta":
                        score = dp
                    else:
                        # length_delta: signed difference in derived length
                        _, L_prev = value_pred_to_length(v_prev_raw, gamma=float(args.gamma), eps=float(args.eps))
                        _, L_cur = value_pred_to_length(v_cur_raw, gamma=float(args.gamma), eps=float(args.eps))
                        score = float(L_cur - L_prev)

                    stat_score.update(score)
                    global_score_sample.update(score)

                    tok_id_cur = int(ps.input_ids[cur_idx])
                    agg = get_token_agg(tok_id_cur)
                    agg.update(score, lower_threshold=float(lower_threshold), upper_threshold=float(upper_threshold))

                    if score > float(upper_threshold) or score < float(lower_threshold):
                        # Collect "following tokens" after the current token and
                        # "previous tokens" before the current token.
                        nxt_n = max(0, int(args.next_token_window))
                        nxt_start = cur_idx + 1
                        nxt_end = min(len(ps.input_ids), cur_idx + 1 + nxt_n)
                        prv_start = max(0, cur_idx - nxt_n)
                        prv_end = cur_idx
                        next_ids: List[int] = []
                        prev_ids: List[int] = [int(t) for t in ps.input_ids[prv_start:prv_end]]
                        if eos_id is None or int(tok_id_cur) != int(eos_id):
                            # Stop at EOS; include EOS itself but nothing after it.
                            for t in ps.input_ids[nxt_start:nxt_end]:
                                tid = int(t)
                                if eos_id is not None and tid == int(eos_id):
                                    next_ids.append(tid)
                                    break
                                next_ids.append(tid)
                        next_texts = [get_tok_text(tid) for tid in next_ids]
                        prev_texts = [get_tok_text(tid) for tid in prev_ids]
                        next_key = tuple(next_ids)
                        prev_key = tuple(prev_ids)
                        if score > float(upper_threshold):
                            if next_ids:
                                ctr = next_seq_by_trigger_above.get(tok_id_cur)
                                if ctr is None:
                                    ctr = Counter()
                                    next_seq_by_trigger_above[tok_id_cur] = ctr
                                ctr[next_key] += 1
                            if prev_ids:
                                ctr = prev_seq_by_trigger_above.get(tok_id_cur)
                                if ctr is None:
                                    ctr = Counter()
                                    prev_seq_by_trigger_above[tok_id_cur] = ctr
                                ctr[prev_key] += 1
                        if score < float(lower_threshold):
                            if next_ids:
                                ctr = next_seq_by_trigger_below.get(tok_id_cur)
                                if ctr is None:
                                    ctr = Counter()
                                    next_seq_by_trigger_below[tok_id_cur] = ctr
                                ctr[next_key] += 1
                            if prev_ids:
                                ctr = prev_seq_by_trigger_below.get(tok_id_cur)
                                if ctr is None:
                                    ctr = Counter()
                                    prev_seq_by_trigger_below[tok_id_cur] = ctr
                                ctr[prev_key] += 1

                        if int(args.events_max_write) >= 0 and n_events_written >= int(args.events_max_write):
                            continue
                        tok_id_prev = int(ps.input_ids[prev_idx])
                        event = {
                            "lenvm_idx": ps.lenvm_idx,
                            "line_no": int(ps.line_no),
                            "assistant_step": int(cur_idx - ps.assistant_start),  # 0-based within assistant tokens
                            "score_name": str(args.score),
                            "upper_threshold": float(upper_threshold),
                            "lower_threshold": float(lower_threshold),
                            "prev": {
                                "token_id": tok_id_prev,
                                "token_text": get_tok_text(tok_id_prev),
                                "value_raw": float(v_prev_raw),
                                "value_sigmoid": float(p_prev),
                            },
                            "cur": {
                                "token_id": tok_id_cur,
                                "token_text": get_tok_text(tok_id_cur),
                                "value_raw": float(v_cur_raw),
                                "value_sigmoid": float(p_cur),
                            },
                            "dp": float(dp),
                            "direction": ("up" if dp > 0 else ("down" if dp < 0 else "flat")),
                            "score": float(score),
                            "next_token_window": int(args.next_token_window),
                            "next_tokens": {
                                "token_ids": next_ids,
                                "token_texts": next_texts,
                            },
                            "prev_tokens": {
                                "token_ids": prev_ids,
                                "token_texts": prev_texts,
                            },
                            "gamma": float(args.gamma),
                            "rel_eps": float(args.rel_eps),
                            "user_text": short_text(ps.user_text, int(args.events_user_text_limit)),
                        }
                        wf_events.write(json.dumps(event, ensure_ascii=False) + "\n")
                        n_events_written += 1

                n_samples_succeeded += 1
            except Exception as e:
                record_failure("process_failed", ps.line_no, ps.lenvm_idx, str(e))

    # Build iterable (optionally shuffled for randomized reading order).
    if bool(args.shuffle_input):
        all_rows = list(iter_jsonl(data_path))
        rng_in = random.Random(int(args.seed))
        rng_in.shuffle(all_rows)
        it: Iterable[Tuple[int, Dict[str, Any]]] = all_rows
        total_rows: Optional[int] = len(all_rows)
    else:
        it = iter_jsonl(data_path)
        total_rows = None

    if bool(args.progress) and tqdm is not None:
        progress_total = total_rows
        if max_samples > 0:
            if progress_total is None:
                progress_total = int(max_samples)
            else:
                progress_total = min(int(progress_total), int(max_samples))
        it = tqdm(it, desc=f"{dataset_name}:prepare+encode", unit="sample", total=progress_total)

    cur_batch: List[PreparedSample] = []
    try:
        with ThreadPoolExecutor(max_workers=max_conc) as ex:
            for line_no, obj in it:
                n_lines_seen += 1
                if max_samples > 0 and n_lines_seen > max_samples:
                    break
                # Prepare sample (tokenize)
                try:
                    ps = prepare_sample(tokenizer, obj, line_no=int(line_no))
                    n_samples_prepared += 1
                except Exception as e:
                    n_samples_failed_prepare += 1
                    meta = obj.get("meta_info") or {}
                    idx = meta.get("lenvm_idx", meta.get("index", "unknown"))
                    record_failure("prepare_failed", int(line_no), str(idx), str(e))
                    continue

                cur_batch.append(ps)
                if len(cur_batch) >= batch_size:
                    submit_batch(ex, cur_batch)
                    cur_batch = []
                    # Backpressure: limit in-flight requests.
                    while len(pending) >= max_conc:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            pending.remove(fut)
                            batch = fut_to_batch.pop(fut, [])
                            try:
                                embs = fut.result()
                                process_batch_result(batch, embs)
                            except Exception as e:
                                n_samples_failed_request += len(batch)
                                for ps2 in batch:
                                    record_failure("request_failed", ps2.line_no, ps2.lenvm_idx, str(e))

                if n_lines_seen % 200 == 0:
                    wf_events.flush()

            # Submit last partial batch.
            if cur_batch:
                submit_batch(ex, cur_batch)
                cur_batch = []

            # Drain remaining futures.
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.remove(fut)
                    batch = fut_to_batch.pop(fut, [])
                    try:
                        embs = fut.result()
                        process_batch_result(batch, embs)
                    except Exception as e:
                        n_samples_failed_request += len(batch)
                        for ps2 in batch:
                            record_failure("request_failed", ps2.line_no, ps2.lenvm_idx, str(e))
    finally:
        wf_events.flush()
        wf_events.close()
        wf_fail.flush()
        wf_fail.close()

    # Write token stats (full) and top lists.
    token_items = [agg.to_dict() for agg in token_aggs.values()]
    token_items.sort(key=lambda x: (x.get("token_id", 0)))
    with token_stats_path.open("w", encoding="utf-8") as wf:
        for item in token_items:
            wf.write(json.dumps(item, ensure_ascii=False) + "\n")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        token_items_for_ranking = token_items
    else:
        token_items_for_ranking = [x for x in token_items if int(x.get("token_id", -1)) != int(eos_id)]

    top_by_count = sorted(
        token_items_for_ranking,
        key=lambda x: (int(x.get("count_score_above_upper", 0)), float(x.get("mean_score", 0.0))),
        reverse=True,
    )[:200]
    top_by_count_low = sorted(
        token_items_for_ranking,
        key=lambda x: (int(x.get("count_score_below_lower", 0)), -float(x.get("mean_score", 0.0))),
        reverse=True,
    )[:200]
    top_by_mean = sorted(
        token_items_for_ranking,
        key=lambda x: (float(x.get("mean_score", 0.0)), int(x.get("count_total", 0))),
        reverse=True,
    )[:200]

    def build_next_text_map(
        trigger_to_counter: Dict[int, Counter[Tuple[int, ...]]], top_seq_k: int = 20
    ) -> Dict[int, List[str]]:
        seq_map: Dict[int, List[str]] = {}
        for tid, ctr in trigger_to_counter.items():
            seqs: List[str] = []
            for seq_ids, _cnt in ctr.most_common(top_seq_k):
                seq_list = list(seq_ids)
                seqs.append(decode_token_seq(seq_list))
            seq_map[int(tid)] = seqs
        return seq_map

    next_text_map_above = build_next_text_map(next_seq_by_trigger_above, top_seq_k=20)
    next_text_map_below = build_next_text_map(next_seq_by_trigger_below, top_seq_k=20)
    prev_text_map_above = build_next_text_map(prev_seq_by_trigger_above, top_seq_k=20)
    prev_text_map_below = build_next_text_map(prev_seq_by_trigger_below, top_seq_k=20)

    def attach_context_texts_to_top_tokens(
        top_tokens: List[Dict[str, Any]],
        next_seq_map: Dict[int, List[str]],
        prev_seq_map: Dict[int, List[str]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for item in top_tokens:
            tid = int(item.get("token_id", -1))
            out_item = dict(item)
            out_item["top_next_token_seqs"] = next_seq_map.get(tid, [])
            out_item["top_prev_token_seqs"] = prev_seq_map.get(tid, [])
            merged.append(out_item)
        return merged

    top_by_mean_thr_above = sorted(
        [x for x in token_items_for_ranking if float(x.get("mean_score", 0.0)) > float(upper_threshold)],
        key=lambda x: (int(x.get("count_total", 0)), float(x.get("mean_score", 0.0))),
        reverse=True,
    )[:200]
    top_by_mean_thr_below = sorted(
        [x for x in token_items_for_ranking if float(x.get("mean_score", 0.0)) < float(lower_threshold)],
        key=lambda x: (int(x.get("count_total", 0)), -float(x.get("mean_score", 0.0))),
        reverse=True,
    )[:200]

    top_by_count_merged = attach_context_texts_to_top_tokens(
        top_by_count, next_text_map_above, prev_text_map_above
    )
    top_by_count_low_merged = attach_context_texts_to_top_tokens(
        top_by_count_low, next_text_map_below, prev_text_map_below
    )
    top_by_mean_thr_above_merged = attach_context_texts_to_top_tokens(
        top_by_mean_thr_above, next_text_map_above, prev_text_map_above
    )
    top_by_mean_thr_below_merged = attach_context_texts_to_top_tokens(
        top_by_mean_thr_below, next_text_map_below, prev_text_map_below
    )

    tokens_by_count_path.write_text(json.dumps(top_by_count_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokens_by_count_low_path.write_text(json.dumps(top_by_count_low_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokens_by_mean_path.write_text(json.dumps(top_by_mean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokens_by_mean_thr_above_path.write_text(
        json.dumps(top_by_mean_thr_above_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tokens_by_mean_thr_below_path.write_text(
        json.dumps(top_by_mean_thr_below_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    wordcloud_above_info: Optional[Dict[str, Any]] = None
    wordcloud_below_info: Optional[Dict[str, Any]] = None
    wordcloud_error: Optional[str] = None
    if bool(args.wordcloud):
        try:
            wc_font_path = (
                str(args.wordcloud_font_path)
                if args.wordcloud_font_path
                else auto_pick_wordcloud_font(items=top_by_mean_thr_above_merged + top_by_mean_thr_below_merged)
            )
            wc_above_path = out_dir / f"{tokens_by_mean_thr_above_path.stem}.wordcloud.png"
            wc_below_path = out_dir / f"{tokens_by_mean_thr_below_path.stem}.wordcloud.png"
            wordcloud_above_info = render_token_wordcloud_from_items(
                items=top_by_mean_thr_above_merged,
                out_png=wc_above_path,
                weight_key="count_total",
                top_k=int(args.wordcloud_top_k),
                min_weight=float(args.wordcloud_min_weight),
                min_count_total=int(args.wordcloud_min_count_total),
                max_count_total=int(args.wordcloud_max_count_total),
                include_special_tokens=bool(args.wordcloud_include_special_tokens),
                include_whitespace_tokens=bool(args.wordcloud_include_whitespace_tokens),
                font_path=wc_font_path,
            )
            wordcloud_below_info = render_token_wordcloud_from_items(
                items=top_by_mean_thr_below_merged,
                out_png=wc_below_path,
                weight_key="count_total",
                top_k=int(args.wordcloud_top_k),
                min_weight=float(args.wordcloud_min_weight),
                min_count_total=int(args.wordcloud_min_count_total),
                max_count_total=int(args.wordcloud_max_count_total),
                include_special_tokens=bool(args.wordcloud_include_special_tokens),
                include_whitespace_tokens=bool(args.wordcloud_include_whitespace_tokens),
                font_path=wc_font_path,
            )
        except Exception as e:
            wordcloud_error = str(e)

    summary = {
        "dataset": str(dataset_name),
        "data": str(data_path),
        "server_url": str(args.server_url),
        "model_dir": str(args.model_dir),
        "model_name": str(model_name),
        "gamma": float(args.gamma),
        "eps": float(args.eps),
        "score": str(args.score),
        "upper_threshold": float(upper_threshold),
        "lower_threshold": float(lower_threshold),
        "rel_eps": float(args.rel_eps),
        "next_token_window": int(args.next_token_window),
        "max_samples": int(args.max_samples),
        "max_steps": "all_assistant_tokens",
        "batch_size": int(batch_size),
        "max_concurrency": int(max_conc),
        "flush_cache": "only_on_encode_length_mismatch_retry_once",
        "request_timeout": float(args.request_timeout),
        "seed": int(args.seed),
        "num_lines_seen": int(n_lines_seen),
        "num_samples_prepared": int(n_samples_prepared),
        "num_samples_succeeded": int(n_samples_succeeded),
        "num_samples_failed_prepare": int(n_samples_failed_prepare),
        "num_samples_failed_request": int(n_samples_failed_request),
        "num_events_written": int(n_events_written),
        "num_unique_tokens_seen": int(len(token_aggs)),
        "score_mean": float(stat_score.get_mean()),
        "score_std": float(stat_score.get_std(sample=False)),
        "score_p50_approx": float(global_score_sample.get_quantile(0.50)),
        "score_p90_approx": float(global_score_sample.get_quantile(0.90)),
        "score_p99_approx": float(global_score_sample.get_quantile(0.99)),
        "score_min_approx": float(global_score_sample.get_quantile(0.00)),
        "score_max_approx": float(global_score_sample.get_quantile(1.00)),
        "output_events_jsonl": str(events_path),
        "output_failures_jsonl": str(failures_path),
        "output_token_stats_jsonl": str(token_stats_path),
        "output_top_tokens_by_count_above_upper_json": str(tokens_by_count_path),
        "output_top_tokens_by_count_below_lower_json": str(tokens_by_count_low_path),
        "output_top_tokens_by_mean_delta_json": str(tokens_by_mean_path),
        "output_top_tokens_by_mean_score_above_upper_json": str(tokens_by_mean_thr_above_path),
        "output_top_tokens_by_mean_score_below_lower_json": str(tokens_by_mean_thr_below_path),
    }
    if wordcloud_above_info is not None:
        summary["output_wordcloud_above_upper_png"] = str(wordcloud_above_info["path"])
        summary["wordcloud_above_upper_num_words"] = int(wordcloud_above_info["num_words"])
        summary["wordcloud_font_path"] = wordcloud_above_info.get("font_path")
    if wordcloud_below_info is not None:
        summary["output_wordcloud_below_lower_png"] = str(wordcloud_below_info["path"])
        summary["wordcloud_below_lower_num_words"] = int(wordcloud_below_info["num_words"])
    if wordcloud_error is not None:
        summary["wordcloud_error"] = str(wordcloud_error)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Reduce tokenizer parallelism noise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

