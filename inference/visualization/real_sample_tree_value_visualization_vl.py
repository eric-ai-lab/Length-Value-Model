"""Tree-value visualization for VLM LVM (Qwen2.5-VL based).

Usage example:
  python real_sample_tree_value_visualization_vl.py \\
    --model-dir Qwen/Qwen2.5-VL-7B-Instruct \\
    --server-url http://127.0.0.1:30010 \\
    --image-path /path/to/image.png \\
    --gamma 0.997 \\
    --ignore-last-token \\
    --max-steps -1 \\
    --token-text-col-width 10 \\
    --viz-rel-err \\
    --viz-width 20 \\
    --viz-max-pct 100 \\
    --viz-clip-marker '▲' \\
    --output-format markdown

Server startup (standalone VLM LVM embedding server):
  CUDA_VISIBLE_DEVICES=X python -m sglang.launch_server \\
    --model-path /path/to/lvm-a-qwen2.5-vl-7b-instruct-b-qwen2.5-vl-3b-instruct \\
    --json-model-override-args '{"architectures":["Qwen2_5_VLForLengthValueModel"]}' \\
    --is-embedding \\
    --attention-backend triton \\
    --host 0.0.0.0 \\
    --port 30010

Note: --model-dir should be the tokenizer-compatible VL base model
      (e.g. Qwen/Qwen2.5-VL-7B-Instruct), NOT the LVM checkpoint,
      because the LVM checkpoint's tokenizer_config.json has a formatting
      quirk that prevents direct loading.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Default VLM sample: MathVista testmini idx=63
#   Correct answer: (B) sample A  (Sample A has higher temperature / KE)
# ---------------------------------------------------------------------------
_QUESTION_TEXT = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    "Question: The diagrams below show two pure samples of gas in identical closed, "
    "rigid containers. Each colored ball represents one gas particle. Both samples "
    "have the same number of particles. Compare the average kinetic energies of the "
    "particles in each sample. Which sample has the higher temperature?\n\n"
    "Answer with the option letter, e.g., A, B, C, D.\n"
    "Choices:\n"
    "(A) neither; the samples have the same temperature\n"
    "(B) sample A\n"
    "(C) sample B"
)

_ASSISTANT_TEXT = (
    "To determine which sample has the higher temperature, we need to compare the "
    "average kinetic energies of the particles in each sample. The average kinetic "
    "energy of a gas particle is given by the equation:\n\n"
    r"\[ \text{KE}_{\text{avg}} = \frac{1}{2} m v^2 \]"
    "\n\nwhere \\( m \\) is the mass of the particle and \\( v \\) is the average "
    "speed of the particle.\n\n"
    "Let's calculate the average kinetic energy for each sample:\n\n"
    "### Sample A\n"
    "- Mass of each particle (\\( m_A \\)) = 40 u\n"
    "- Average speed of each particle (\\( v_A \\)) = 950 m/s\n\n"
    r"\[ \text{KE}_{\text{avg, A}} = \frac{1}{2} \times 40 \, \text{u} \times (950 \, \text{m/s})^2 \]"
    "\n\n### Sample B\n"
    "- Mass of each particle (\\( m_B \\)) = 32 u\n"
    "- Average speed of each particle (\\( v_B \\)) = 750 m/s\n\n"
    r"\[ \text{KE}_{\text{avg, B}} = \frac{1}{2} \times 32 \, \text{u} \times (750 \, \text{m/s})^2 \]"
    "\n\nNow, let's compare the kinetic energies:\n\n"
    r"\[ \text{KE}_{\text{avg, A}} = \frac{1}{2} \times 40 \times 950^2 = 20 \times 902500 = 18050000 \, \text{u} \cdot \text{m}^2/\text{s}^2 \]"
    "\n\n"
    r"\[ \text{KE}_{\text{avg, B}} = \frac{1}{2} \times 32 \times 750^2 = 16 \times 562500 = 8960000 \, \text{u} \cdot \text{m}^2/\text{s}^2 \]"
    "\n\nSince the kinetic energy of the particles in Sample A is higher than that in "
    "Sample B, the temperature of Sample A is higher than that of Sample B.\n\n"
    "Therefore, the correct answer is:\n\n"
    "\\boxed{B}"
)

_DEFAULT_IMAGE_PATH = str(
    Path(__file__).parent / "mathvista_idx63.png"
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 300.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_text(url: str, payload: Optional[Dict] = None, timeout_s: float = 60.0) -> str:
    data = b"" if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# VLM tokenization helpers
# ---------------------------------------------------------------------------

def _load_processor(model_dir: str):
    """Load Qwen2.5-VL processor. Falls back to AutoProcessor."""
    from transformers import AutoProcessor
    # Use_fast=False for the image processor avoids the 'fast processor' warning
    # that changes output slightly.
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)


def _image_to_b64_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(suffix, "image/png")
    return f"data:{mime};base64,{data}"


def _tokenize_vl_with_chat_template(
    processor,
    user_text: str,
    assistant_text: str,
    image_path: str,
    system_text: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> Tuple[List[int], int, str, str]:
    """Tokenize a VLM conversation (image + text).

    Returns:
        full_ids        – expanded token IDs (local; used for assistant_ids & row building)
        assistant_start – index of the first assistant token in LOCAL full_ids
        full_text       – raw apply_chat_template string for the FULL conversation
        prompt_text     – raw apply_chat_template string for the PROMPT only

    Both text strings are passed to the server as the `text` field so the server
    handles image expansion with its own processor.  assistant_start_server is
    obtained by calling /encode on prompt_text first (see main()).
    """
    from PIL import Image
    img = Image.open(image_path)

    messages_prompt = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text},
        ]},
    ]
    messages_full = messages_prompt + [{"role": "assistant", "content": assistant_text}]

    prompt_text = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )

    # Process locally only to determine assistant_ids (text tokens, not image-dependent).
    inputs_prompt = processor(text=[prompt_text], images=[img], return_tensors="pt")
    inputs_full = processor(text=[full_text], images=[img], return_tensors="pt")

    prompt_ids: List[int] = inputs_prompt["input_ids"][0].tolist()
    full_ids: List[int] = inputs_full["input_ids"][0].tolist()

    # Sanity-check: the full sequence must start with the prompt.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"VLM tokenization mismatch: full_ids[:{len(prompt_ids)}] != prompt_ids. "
            "This usually indicates a mismatch in image preprocessing settings."
        )

    return full_ids, len(prompt_ids), full_text, prompt_text


def _strip_trailing_eos(tokenizer, ids: List[int]) -> List[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return ids
    end = len(ids)
    while end > 0 and ids[end - 1] == eos:
        end -= 1
    return ids[:end]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Tree-value visualization for VLM LVM (Qwen2.5-VL based)."
    )
    ap.add_argument("--model-dir", type=str, required=True,
                    help="HF tokenizer/processor directory (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    ap.add_argument("--server-url", type=str, default="http://127.0.0.1:30010",
                    help="Standalone VLM LVM embedding server base URL")
    ap.add_argument("--image-path", type=str, default=_DEFAULT_IMAGE_PATH,
                    help="Path to the image file (default: mathvista_idx63.png)")
    ap.add_argument("--gamma", type=float, default=0.997,
                    help="LVM value→length discount gamma. length = ln(1+y_hat)/ln(gamma).")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Numerical epsilon for length conversion.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Max assistant steps to evaluate (-1 = all).")
    ap.add_argument("--start-step", type=int, default=0,
                    help="Start offset within assistant tokens.")
    ap.add_argument("--ignore-last-token", action="store_true", default=False,
                    help="Drop the last token after EOS stripping.")
    ap.add_argument("--token-text-col-width", type=int, default=10,
                    help="Fixed column width for the token_text column.")
    ap.add_argument("--viz-rel-err", action="store_true", default=False,
                    help="Add ASCII bar visualization for rel_err%%.")
    ap.add_argument("--viz-width", type=int, default=20)
    ap.add_argument("--viz-fill-char", type=str, default="█")
    ap.add_argument("--viz-empty-char", type=str, default="░")
    ap.add_argument("--viz-max-pct", type=float, default=100.0)
    ap.add_argument("--viz-clip-marker", type=str, default="▲")
    ap.add_argument("--include-prompt-last", action="store_true", default=True,
                    help="Print an extra row for the last prompt token.")
    ap.add_argument("--output-format", type=str, default="plain",
                    choices=["plain", "markdown"])
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load processor
    # ------------------------------------------------------------------
    print(f"Loading processor from {args.model_dir} …")
    processor = _load_processor(args.model_dir)
    tokenizer = processor.tokenizer

    # ------------------------------------------------------------------
    # Tokenize the VLM sample
    # ------------------------------------------------------------------
    user_text = _QUESTION_TEXT
    assistant_text = _ASSISTANT_TEXT
    image_path = args.image_path

    if not Path(image_path).exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Generate it with:\n"
            "  python -c \"import json,base64; ..."
            " (see test_sglang_lvm_vl.sh for instructions)\""
        )

    print(f"Tokenizing conversation with image: {image_path} …")
    full_ids, assistant_start_local, full_text, prompt_text = _tokenize_vl_with_chat_template(
        processor, user_text, assistant_text, image_path
    )

    # assistant_ids are purely text tokens (same regardless of image resolution).
    assistant_ids = full_ids[assistant_start_local:]
    assistant_ids = _strip_trailing_eos(tokenizer, assistant_ids)
    if args.ignore_last_token and len(assistant_ids) > 0:
        assistant_ids = assistant_ids[:-1]
    total_assistant = len(assistant_ids)

    # Count image pad tokens for info (local)
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    num_image_tokens = sum(1 for t in full_ids if t == image_pad_id)

    # ------------------------------------------------------------------
    # Print dialogue header
    # ------------------------------------------------------------------
    if args.output_format == "markdown":
        print("\n### DIALOGUE\n")
        print("#### USER (image + text)\n")
        print(f"[IMAGE: {Path(image_path).name}]")
        print(user_text)
        print("\n---\n")
        print("#### ASSISTANT\n")
        print(assistant_text)
        print("\n---\n")
    else:
        print("\n" + "=" * 80)
        print("DIALOGUE (VLM)")
        print("=" * 80)
        print(f"\n[USER (image: {Path(image_path).name})]\n")
        print(user_text)
        print("\n" + "-" * 80)
        print("\n[ASSISTANT]\n")
        print(assistant_text)
        print("\n" + "=" * 80 + "\n")

    print(f"local assistant_start : {assistant_start_local}")
    print(f"local total_full_tokens: {len(full_ids)} ({num_image_tokens} image-pad tokens)")
    print(f"assistant_tokens (excl trailing EOS): {total_assistant}")
    print(f"server: {args.server_url}")
    print(f"gamma:  {args.gamma}")

    # ------------------------------------------------------------------
    # Build rows: (step, token_id, token_text, true_remaining)
    # ------------------------------------------------------------------
    start_step = max(0, min(args.start_step, max(0, total_assistant - 1)))
    available = max(0, total_assistant - start_step)
    max_steps = available if (args.max_steps is None or int(args.max_steps) < 0) \
        else min(int(args.max_steps), available)

    # Rows are built from LOCAL assistant_ids (text tokens, image-resolution-independent).
    # The "prompt last" token index uses assistant_start_local (local sequence).
    rows: List[Tuple[int, int, str, int]] = []
    if args.include_prompt_last and start_step == 0 and assistant_start_local > 0:
        prev_id = full_ids[assistant_start_local - 1]
        prev_str = tokenizer.decode([prev_id], skip_special_tokens=False)
        rows.append((-1, prev_id, prev_str, total_assistant))
    for k in range(max_steps):
        step = start_step + k
        tok_id = assistant_ids[step]
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        rows.append((step, tok_id, tok_str, total_assistant - (step + 1)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def value_pred_to_length(value_pred: float) -> Tuple[float, float]:
        gamma = float(args.gamma)
        eps = float(args.eps)
        y_hat = -float(torch.sigmoid(torch.tensor(float(value_pred))).item())
        y = min(-eps, max(-1.0 + eps, y_hat))
        length = math.log1p(y) / min(-eps, math.log(gamma))
        return y_hat, length

    text_w = max(5, int(args.token_text_col_width))

    def fmt_token_text(s: str) -> str:
        s = s.replace("\n", "\\n").replace("\t", "\\t")
        return s[:text_w - 3] + "..." if len(s) > text_w else s.ljust(text_w)

    viz_w = max(4, int(args.viz_width))

    def viz_bar(rel_err_pct: float) -> str:
        if not args.viz_rel_err:
            return ""
        max_pct = float(args.viz_max_pct) or 100.0
        raw = float(rel_err_pct)
        clipped = raw > max_pct
        x = max(0.0, min(raw, max_pct))
        filled = max(0, min(viz_w, int(round(x / max_pct * viz_w))))
        fill_ch = (str(args.viz_fill_char) or "█")[0]
        empty_ch = (str(args.viz_empty_char) or "░")[0]
        bar = fill_ch * filled + empty_ch * (viz_w - filled)
        if clipped and viz_w > 0:
            bar = bar[:-1] + (str(args.viz_clip_marker) or "!")[0]
        return bar

    header = (
        f"\n{'step':>5}  {'token_id':>7}  {'token_text':<{text_w}}  "
        f"{'value_pred':>10}  {'y_hat':>10}  {'len_pred':>9}  {'true':>5}  {'diff':>9}  {'rel_err%':>9}"
    )
    if args.viz_rel_err:
        header += f"  {'viz':<{viz_w}}"

    if args.output_format == "markdown":
        print("```")
    print(header)
    sep_line = "-" * (len(header) - 1)
    print(sep_line)

    # ------------------------------------------------------------------
    # Call /encode with text + image_data
    # We send the raw apply_chat_template strings (NOT pre-expanded input_ids) so
    # that the server's VL processor handles image expansion itself.
    #
    # Because the server's image processor may use different min/max_pixels than
    # the local processor (e.g. 3B-Instruct vs 7B-Instruct defaults), the server
    # may produce a different number of image tokens.  We therefore determine
    # assistant_start_server by encoding the PROMPT ONLY first and measuring the
    # returned embedding length, rather than relying on the local value.
    # ------------------------------------------------------------------
    print("Flushing server cache …", flush=True)
    try:
        _http_post_text(args.server_url.rstrip("/") + "/flush_cache", timeout_s=30.0)
    except Exception:
        pass

    image_b64_url = _image_to_b64_data_url(image_path)
    enc_url = args.server_url.rstrip("/") + "/encode"

    # Step 1: encode prompt only → server-side assistant_start
    # IMPORTANT: flush cache first so we get a fresh full-sequence embedding below.
    print("Sending /encode (prompt only) to determine server assistant_start …", flush=True)
    try:
        resp_prompt = _http_post_json(
            enc_url,
            {"text": prompt_text, "image_data": image_b64_url},
            timeout_s=300.0,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from /encode (prompt): {body}") from e

    emb_prompt = (resp_prompt["embedding"] if isinstance(resp_prompt, dict)
                  else resp_prompt[0]["embedding"])
    assistant_start_server = len(emb_prompt)
    print(f"server assistant_start : {assistant_start_server}  "
          f"(local: {assistant_start_local}, diff={assistant_start_server - assistant_start_local})")

    # Step 2: flush cache again so the full-conversation encode is NOT served from the
    # radix cache.  Without this second flush the server reuses the 141-token prompt
    # prefix cached by step 1 and returns only the 497 NEW (assistant) tokens.
    # We'd then read emb[141] = the 141st assistant token instead of the 1st.
    print("Flushing server cache (pre-full encode) …", flush=True)
    try:
        _http_post_text(args.server_url.rstrip("/") + "/flush_cache", timeout_s=30.0)
    except Exception:
        pass

    # Step 3: encode full conversation (cache empty → server returns all tokens)
    print("Sending /encode (full conversation) …", flush=True)
    try:
        resp = _http_post_json(
            enc_url,
            {"text": full_text, "image_data": image_b64_url},
            timeout_s=600.0,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from /encode (full): {body}") from e

    if isinstance(resp, dict) and "embedding" in resp:
        emb = resp["embedding"]
    else:
        emb = resp[0]["embedding"]

    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected /encode embedding type: {type(emb)}")

    server_total = len(emb)
    print(f"server total tokens: {server_total}  (local: {len(full_ids)})")

    # Use server-side assistant_start for all embedding indexing.
    assistant_start = assistant_start_server

    # Server-side total assistant tokens (for true_remaining baseline).
    # The server may tokenize the assistant text differently from local (e.g. different
    # number of image tokens), so use the server's count for accurate error reporting.
    total_assistant_server = server_total - assistant_start_server
    print(f"server assistant tokens: {total_assistant_server}  (local: {total_assistant})")

    if assistant_start >= server_total:
        raise RuntimeError(
            f"assistant_start ({assistant_start}) >= server embedding length ({server_total})."
        )

    # Rebuild rows using server-side total_assistant for true_remaining.
    rows_server: List[Tuple[int, int, str, int]] = []
    for step, tok_id, tok_str, _ in rows:
        true_remaining_server = (total_assistant_server if step == -1
                                 else total_assistant_server - (step + 1))
        rows_server.append((step, tok_id, tok_str, true_remaining_server))
    rows = rows_server

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    last_assistant_step = total_assistant - 1
    last_token_note_printed = False

    for step, tok_id, tok_str, true_remaining in rows:
        tok_str_clean = fmt_token_text(tok_str)
        idx = (assistant_start - 1) if step == -1 else (assistant_start + step)
        value_pred_raw = float(emb[idx])
        y_hat, l_pred = value_pred_to_length(value_pred_raw)
        diff = l_pred - float(true_remaining)
        denom = float(true_remaining) if true_remaining > 0 else 1.0
        rel_err = 100.0 * abs(diff) / denom
        bar = viz_bar(rel_err)
        if step == last_assistant_step and not last_token_note_printed:
            print(sep_line)
        line = (
            f"{step:5d}  {tok_id:7d}  {tok_str_clean}  "
            f"{value_pred_raw:10.4f}  {y_hat:10.6f}  {l_pred:9.3f}  {true_remaining:5d}  "
            f"{diff:9.3f}  {rel_err:9.2f}"
        )
        if args.viz_rel_err:
            line += f"  {bar}"
        print(line)
        if step == last_assistant_step and not last_token_note_printed:
            print(
                "NOTE: The last assistant token is often not trained (value_mask=0), "
                "so its length error may be much larger."
            )
            last_token_note_printed = True

    if args.output_format == "markdown":
        print("```")


if __name__ == "__main__":
    main()
