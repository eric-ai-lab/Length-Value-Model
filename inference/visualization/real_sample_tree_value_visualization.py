from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from transformers import AutoTokenizer


DEFAULT_SAMPLE_FILE = Path(__file__).with_name("default_text_sample.json")


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 60.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _http_post_text(url: str, payload: Dict[str, Any] | None = None, timeout_s: float = 60.0) -> str:
    data = b"" if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _load_sample(sample_path: str | None) -> Dict[str, Any]:
    path = Path(sample_path) if sample_path is not None else DEFAULT_SAMPLE_FILE

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        first_line = next((line for line in text.splitlines() if line.strip()), None)
        if first_line is None:
            raise ValueError(f"Empty sample file: {path}")
        return json.loads(first_line)
    return json.loads(text)


def _normalize_messages(sample: Dict[str, Any]) -> Tuple[str, str]:
    if "conversations" in sample:
        conv = sample["conversations"]
        assert len(conv) >= 2
        user = next(x for x in conv if x["from"] in ("human", "user"))["value"]
        assistant = next(x for x in conv if x["from"] in ("gpt", "assistant"))["value"]
        return user, assistant

    if "messages" in sample:
        messages = sample["messages"]
        assert len(messages) >= 2
        user = next(x for x in messages if x["role"] in ("human", "user"))["content"]
        assistant = next(x for x in messages if x["role"] in ("gpt", "assistant"))["content"]
        return user, assistant

    if "prompt" in sample and "response" in sample:
        return sample["prompt"], sample["response"]

    if "question" in sample and "answer" in sample:
        return sample["question"], sample["answer"]

    raise ValueError(
        "Unsupported sample format. Expected conversations, messages, prompt/response, or question/answer."
    )


def _tokenize_with_chat_template(tokenizer, user_text: str, assistant_text: str) -> Tuple[List[int], int]:
    # Prompt ids up to the start of assistant generation.
    prompt_ids: List[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Full ids including assistant content (may include trailing special tokens depending on template).
    full_ids: List[int] = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Chat template tokenization mismatch: full_ids does not start with prompt_ids. "
            "This usually means the tokenizer template behaves differently for add_generation_prompt."
        )
    return full_ids, len(prompt_ids)


def _strip_trailing_eos(tokenizer, ids: List[int]) -> List[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return ids
    # Some templates append one eos; strip all trailing eos for "true remaining tokens" calculation.
    end = len(ids)
    while end > 0 and ids[end - 1] == eos:
        end -= 1
    return ids[:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True, help="HF tokenizer directory")
    ap.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help="JSON/JSONL sample file. Supports conversations, messages, prompt/response, or question/answer.",
    )
    ap.add_argument(
        "--base-generator-label",
        type=str,
        default="base generator",
        help="Label for the model/source that generated the assistant response.",
    )
    ap.add_argument(
        "--lenvm-label",
        type=str,
        default="LenVM",
        help="Label for the LenVM checkpoint that predicts remaining length.",
    )
    ap.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:30010",
        help="SGLang server base URL",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="encode",
        choices=["encode", "tree_value"],
        help=(
            "encode: one forward via /encode over the full [prompt+assistant] to get per-token value_pred_raw; "
            "tree_value: many forwards via /tree_value with prefix+true_next_token candidate."
        ),
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.997,
        help="LVM value->length discount gamma. length = ln(1+v)/ln(gamma).",
    )
    ap.add_argument(
        "--output-format",
        type=str,
        default="plain",
        choices=["plain", "markdown"],
        help=(
            "plain: normal terminal output; "
            "markdown: print markdown headings and wrap the metrics table in a fenced code block "
            "for nicer rendering in markdown viewers."
        ),
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for clamping value before converting to length.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "How many next-token steps to evaluate from the assistant response. "
            "Set to -1 to evaluate all available steps."
        ),
    )
    ap.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start step offset within assistant tokens (0 means the first assistant token).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many prefixes to send per /tree_value call",
    )
    ap.add_argument(
        "--token-text-col-width",
        type=int,
        default=20,
        help="Fixed column width reserved for token_text (space-padded).",
    )
    ap.add_argument(
        "--ignore-last-token",
        action="store_true",
        default=False,
        help=(
            "If set, drop the last token after tokenization (after stripping trailing EOS). "
            "Useful when chat template/tokenization sometimes appends an extra trailing token."
        ),
    )
    ap.add_argument(
        "--viz-rel-err",
        action="store_true",
        default=False,
        help="If set, add an ASCII bar visualization column for rel_err% (clipped).",
    )
    ap.add_argument(
        "--viz-width",
        type=int,
        default=24,
        help="Width (characters) of the rel_err% visualization bar.",
    )
    ap.add_argument(
        "--viz-fill-char",
        type=str,
        default="█",
        help="Single-character used for the filled portion of the viz bar.",
    )
    ap.add_argument(
        "--viz-empty-char",
        type=str,
        default="░",
        help="Single-character used for the empty portion of the viz bar.",
    )
    ap.add_argument(
        "--viz-max-pct",
        type=float,
        default=100.0,
        help="rel_err%% value that maps to a full bar (values above are clipped).",
    )
    ap.add_argument(
        "--viz-clip-marker",
        type=str,
        default="▲",
        help=(
            "Single-character marker used at the end of the viz bar when rel_err%% exceeds viz-max-pct. "
            "You can set this to something fun like '🌋' (may affect terminal column alignment)."
        ),
    )
    ap.add_argument(
        "--include-prompt-last",
        action="store_true",
        default=True,
        help=(
            "If set, print an extra first row for the token right before the assistant answer "
            "(prompt last token). This matches LVM training where the last prompt token predicts "
            "the remaining answer length."
        ),
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)

    sample = _load_sample(args.sample_file)
    user_text, assistant_text = _normalize_messages(sample)
    base_generator_label = sample.get("base_generator_label", args.base_generator_label)
    lenvm_label = args.lenvm_label

    if args.output_format == "markdown":
        print("\n### DIALOGUE\n")
        print("#### USER\n")
        print(user_text)
        print("\n---\n")
        print("#### ASSISTANT\n")
        print(assistant_text)
        print("\n---\n")
        print("### METADATA\n")
        print(f"base_generator: {base_generator_label}")
        print(f"lenvm: {lenvm_label}")
        if args.sample_file is not None:
            print(f"sample_file: {args.sample_file}")
        else:
            print(f"sample_file: {DEFAULT_SAMPLE_FILE}")
        print("\n---\n")
    else:
        print("\n" + "=" * 80)
        print("DIALOGUE (raw text)")
        print("=" * 80)
        print("\n[USER]\n")
        print(user_text)
        print("\n" + "-" * 80)
        print("\n[ASSISTANT]\n")
        print(assistant_text)
        print("\n" + "=" * 80 + "\n")
        print("METADATA")
        print("-" * 80)
        print(f"base_generator: {base_generator_label}")
        print(f"lenvm: {lenvm_label}")
        if args.sample_file is not None:
            print(f"sample_file: {args.sample_file}")
        else:
            print(f"sample_file: {DEFAULT_SAMPLE_FILE}")
        print("\n" + "=" * 80 + "\n")

    full_ids, assistant_start = _tokenize_with_chat_template(tokenizer, user_text, assistant_text)

    # Define assistant span and strip trailing eos for "true remaining token" calculation.
    assistant_ids = full_ids[assistant_start:]
    assistant_ids = _strip_trailing_eos(tokenizer, assistant_ids)
    if args.ignore_last_token and len(assistant_ids) > 0:
        assistant_ids = assistant_ids[:-1]
    total_assistant = len(assistant_ids)

    print("base_generator:", base_generator_label)
    print("lenvm:", lenvm_label)
    print("sample_file:", args.sample_file or str(DEFAULT_SAMPLE_FILE))
    print("assistant_start:", assistant_start)
    print("total_full_tokens:", len(full_ids))
    print("assistant_tokens(excl trailing eos):", total_assistant)
    print("server:", args.server_url)
    print("mode:", args.mode)
    print("gamma:", args.gamma)

    # Evaluate steps inside the assistant response.
    # We include the LAST assistant token as well (true_remaining can be 0), which matches
    # the dataset processor that can label the final non-EOS token with remaining length 0.
    start_step = max(0, min(args.start_step, max(0, total_assistant - 1)))
    available_steps = max(0, total_assistant - start_step)
    if args.max_steps is None or int(args.max_steps) < 0:
        max_steps = available_steps
    else:
        max_steps = min(int(args.max_steps), available_steps)

    # Build batched requests: prefix = full_ids[:assistant_start + step], candidate = assistant_ids[step]
    # True remaining after consuming candidate: total_assistant - (step+1)
    rows: List[Tuple[int, int, str, int]] = []
    # Extra first row: the token right before the answer begins (prompt last token).
    # In training, this position's label is the remaining answer length.
    if args.include_prompt_last and start_step == 0 and assistant_start > 0:
        prev_idx = assistant_start - 1
        prev_token_id = full_ids[prev_idx]
        prev_tok_str = tokenizer.decode([prev_token_id], skip_special_tokens=False)
        rows.append((-1, prev_token_id, prev_tok_str, total_assistant))
    for k in range(max_steps):
        step = start_step + k
        token_id = assistant_ids[step]
        # Decode token for readability (may contain whitespace markers).
        tok_str = tokenizer.decode([token_id], skip_special_tokens=False)
        true_remaining = total_assistant - (step + 1)
        rows.append((step, token_id, tok_str, true_remaining))

    url = args.server_url.rstrip("/") + "/tree_value"

    def value_pred_to_length(value_pred: float) -> Tuple[float, float]:
        # Matches LlamaFactory-lvm trainer:
        #   y_hat = -sigmoid(value_pred)   -> y_hat in (-1, 0)
        #   length = ln(1 + y_hat) / ln(gamma)
        eps = float(args.eps)
        gamma = float(args.gamma)
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0,1), got {gamma}")
        # Convert raw head output to discounted return y_hat in (-1, 0)
        y_hat = -float(torch.sigmoid(torch.tensor(float(value_pred))).item())
        # clamp to (-1+eps, 0-eps)
        y = min(-eps, max(-1.0 + eps, y_hat))
        length = math.log1p(y) / min(-eps, math.log(gamma))
        return y_hat, length

    text_w = max(5, int(args.token_text_col_width))

    def fmt_token_text(s: str) -> str:
        s = s.replace("\n", "\\n").replace("\t", "\\t")
        # truncate + pad to fixed width
        if len(s) > text_w:
            s = s[: text_w - 3] + "..."
        return s.ljust(text_w)

    viz_w = max(4, int(args.viz_width))

    def viz_bar(rel_err_pct: float) -> str:
        if not args.viz_rel_err:
            return ""
        max_pct = float(args.viz_max_pct)
        if max_pct <= 0:
            max_pct = 100.0
        raw = float(rel_err_pct)
        clipped = raw > max_pct
        x = max(0.0, min(raw, max_pct))
        filled = int(round((x / max_pct) * viz_w))
        filled = max(0, min(viz_w, filled))
        fill_ch = (str(args.viz_fill_char) or "█")[0]
        empty_ch = (str(args.viz_empty_char) or "░")[0]
        bar = (fill_ch * filled) + (empty_ch * (viz_w - filled))
        # Mark clipping (e.g., rel_err% > viz_max_pct) without changing column width.
        if clipped and viz_w > 0:
            marker = str(args.viz_clip_marker) if args.viz_clip_marker else "!"
            marker = marker[0]
            bar = bar[:-1] + marker
        return bar

    header = (
        f"\n{'step':>5}  {'token_id':>7}  {'token_text':<{text_w}}  "
        f"{'value_pred':>10}  {'y_hat':>10}  {'len_pred':>9}  {'td_err':>9}  {'true':>5}  {'diff':>9}  {'rel_err%':>9}"
    )
    if args.viz_rel_err:
        header += f"  {'viz':<{viz_w}}"
    # In markdown mode, wrap the fixed-width table in a code fence for proper rendering.
    if args.output_format == "markdown":
        print("```")
    print(header)
    sep_line = "-" * (len(header) - 1)
    print(sep_line)

    # The last assistant token (true_remaining=0) is often a template/special token and may be excluded
    # from training by value_mask, so its error can be much larger. We visually separate it.
    last_assistant_step = total_assistant - 1
    last_token_note_printed = False

    if args.mode == "encode":
        # IMPORTANT:
        # /encode may hit radix cache and only forward the extend tail, which would shrink the
        # returned tokenwise vector. Flush cache (best-effort) so we can get full-length tokenwise outputs
        # in a single forward.
        try:
            _http_post_text(args.server_url.rstrip("/") + "/flush_cache", None, timeout_s=30.0)
        except Exception:
            pass

        # One forward: call /encode on the full sequence and reuse tokenwise outputs.
        enc_url = args.server_url.rstrip("/") + "/encode"
        payload = {"input_ids": [full_ids], "bypass_cache": [True]}
        try:
            resp = _http_post_json(enc_url, payload, timeout_s=300.0)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from /encode: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to call /encode: {e}") from e

        # /encode always returns a list for batch; be defensive anyway.
        if isinstance(resp, dict) and "embedding" in resp:
            emb = resp["embedding"]
        else:
            emb = resp[0]["embedding"]
        if not isinstance(emb, list) or len(emb) != len(full_ids):
            raise RuntimeError(
                f"Unexpected /encode embedding length: got {len(emb) if isinstance(emb, list) else type(emb)}, expected {len(full_ids)}"
            )

        y_hats_by_step: Dict[int, float] = {}
        for step, tok_id, tok_str, true_remaining in rows:
            tok_str_clean = fmt_token_text(tok_str)
            # step == -1 means prompt-last token at index assistant_start-1
            idx = (assistant_start - 1) if step == -1 else (assistant_start + step)
            value_pred_raw = float(emb[idx])
            y_hat, l_pred = value_pred_to_length(value_pred_raw)
            p_cur = -y_hat
            p_prev = None if step == -1 else y_hats_by_step.get(step - 1)
            td_err = 0.0 if p_prev is None else float(args.gamma) * p_cur - p_prev + (1.0 - float(args.gamma))
            y_hats_by_step[step] = p_cur
            diff = l_pred - float(true_remaining)
            denom = float(true_remaining) if true_remaining > 0 else 1.0
            rel_err = 100.0 * abs(diff) / denom
            bar = viz_bar(rel_err)
            if step == last_assistant_step and not last_token_note_printed:
                print(sep_line)
            line = (
                f"{step:5d}  {tok_id:7d}  {tok_str_clean}  "
                f"{value_pred_raw:10.4f}  {y_hat:10.6f}  {l_pred:9.3f}  {td_err:9.6f}  {true_remaining:5d}  {diff:9.3f}  {rel_err:9.2f}"
            )
            if args.viz_rel_err:
                line += f"  {bar}"
            print(line)
            if step == last_assistant_step and not last_token_note_printed:
                print(
                    "NOTE: The last assistant token is often not trained (value_mask=0, e.g., template/EOS/end marker), "
                    "so its length error may be much larger."
                )
                last_token_note_printed = True
    else:
        # Many forwards: /tree_value with prefix + true_next_token as the single candidate.
        y_hats_by_step: Dict[int, float] = {}
        for b in range(0, len(rows), args.batch_size):
            chunk = rows[b : b + args.batch_size]
            input_ids_batch = [
                full_ids[: assistant_start + step] for (step, _, _, _) in chunk
            ]
            candidate_ids_batch = [[tok_id] for (_, tok_id, _, _) in chunk]

            payload = {"input_ids": input_ids_batch, "candidate_ids": candidate_ids_batch}
            try:
                resp = _http_post_json(url, payload, timeout_s=120.0)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {e.code} from /tree_value: {body}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to call /tree_value: {e}") from e

            if isinstance(resp, dict) and "embedding" in resp:
                resp_list = [resp]
            else:
                resp_list = resp

            if not isinstance(resp_list, list) or len(resp_list) != len(chunk):
                raise RuntimeError(
                    f"Unexpected /tree_value response shape: {type(resp)} {resp}"
                )

            for i, item in enumerate(resp_list):
                pred_list = item["embedding"]
                if not isinstance(pred_list, list) or len(pred_list) != 1:
                    raise RuntimeError(
                        f"Expected single-candidate embedding list, got: {pred_list}"
                    )
                value_pred_raw = float(pred_list[0])
                y_hat, l_pred = value_pred_to_length(value_pred_raw)

                step, tok_id, tok_str, true_remaining = chunk[i]
                tok_str_clean = fmt_token_text(tok_str)
                p_cur = -y_hat
                p_prev = None if step == -1 else y_hats_by_step.get(step - 1)
                td_err = 0.0 if p_prev is None else float(args.gamma) * p_cur - p_prev + (1.0 - float(args.gamma))
                y_hats_by_step[step] = p_cur
                diff = l_pred - float(true_remaining)
                denom = float(true_remaining) if true_remaining > 0 else 1.0
                rel_err = 100.0 * abs(diff) / denom
                bar = viz_bar(rel_err)
                if step == last_assistant_step and not last_token_note_printed:
                    print(sep_line)
                line = (
                    f"{step:5d}  {tok_id:7d}  {tok_str_clean}  "
                    f"{value_pred_raw:10.4f}  {y_hat:10.6f}  {l_pred:9.3f}  {td_err:9.6f}  {true_remaining:5d}  {diff:9.3f}  {rel_err:9.2f}"
                )
                if args.viz_rel_err:
                    line += f"  {bar}"
                print(line)
                if step == last_assistant_step and not last_token_note_printed:
                    print(
                        "NOTE: The last assistant token is often not trained (value_mask=0, e.g., template/EOS/end marker), "
                        "so its length error may be much larger."
                    )
                    last_token_note_printed = True

            time.sleep(0.02)

    if args.output_format == "markdown":
        print("```")

    # Summary: compute simple correlation/MAE on raw preds vs true remaining (note: units may differ by training).
    # We still print them for quick sanity checks.
    # (This does not assume the model predicts raw token counts.)
    preds: List[float] = []
    trues: List[float] = []
    # Recompute preds by re-calling server in one shot would be costly; instead parse stdout if needed.
    # Keep placeholders for future extension.
    _ = preds, trues


if __name__ == "__main__":
    main()

