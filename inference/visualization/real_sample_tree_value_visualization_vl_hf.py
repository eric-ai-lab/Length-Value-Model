"""Tree-value visualization for VLM LVM using 🤗 Transformers directly.

No SGLang server required.  Loads the model + value head locally, runs one
forward pass on the full conversation, and prints the per-token length-value
table.

Usage example:
  python real_sample_tree_value_visualization_vl_hf.py \\
    --model-path /path/to/lvm-a-qwen2.5-vl-7b-instruct-b-qwen2.5-vl-3b-instruct \\
    --image-path /path/to/mathvista_idx63.png \\
    --gamma 0.997 \\
    --ignore-last-token \\
    --max-steps -1 \\
    --token-text-col-width 10 \\
    --viz-rel-err \\
    --viz-width 20 \\
    --viz-max-pct 100 \\
    --output-format markdown

Notes:
  - --model-path is the LVM checkpoint directory (contains model.safetensors
    and value_head.safetensors).
  - Runs in bfloat16 on GPU(s) automatically via device_map="auto".
  - For CPU-only machines add --dtype float32.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn


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
    "- Average speed of each particle (\\( m_B \\)) = 750 m/s\n\n"
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

_SYSTEM_TEXT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


# ---------------------------------------------------------------------------
# Value head (must match the architecture trained in LlamaFactory-lvm)
# ---------------------------------------------------------------------------

class MLP2SiLUValueHead(nn.Module):
    """Two-layer value head: Linear(H→H) → SiLU → Linear(H→1)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        if x.dtype != self.summary.weight.dtype:
            x = x.to(self.summary.weight.dtype)
        x = self.fc(x)
        x = self.act(x)
        return self.summary(x)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_vhead(model_path: str, dtype: torch.dtype):
    """Load VLM backbone + value head from the LVM checkpoint directory."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    print(f"Loading VLM backbone from {model_path} …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"  hidden_size = {hidden_size}")

    print("Loading value head weights …")
    from safetensors.torch import load_file
    vh_path = Path(model_path) / "value_head.safetensors"
    if not vh_path.exists():
        raise FileNotFoundError(f"value_head.safetensors not found at {vh_path}")
    raw = load_file(str(vh_path), device="cpu")

    v_head = MLP2SiLUValueHead(hidden_size)
    state = {k[len("v_head."):]: v for k, v in raw.items() if k.startswith("v_head.")}
    missing, unexpected = v_head.load_state_dict(state, strict=True)
    if missing:
        raise RuntimeError(f"Missing v_head weights: {missing}")

    # Move v_head to the same device as the last transformer layer
    last_device = next(reversed(list(model.parameters()))).device
    v_head = v_head.to(dtype=dtype, device=last_device)
    v_head.eval()
    print(f"  v_head loaded, moved to device: {last_device}")
    return model, v_head


def _load_processor(model_path: str):
    from transformers import AutoProcessor
    print(f"Loading processor from {model_path} …")
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize_vl(
    processor,
    user_text: str,
    assistant_text: str,
    image_path: str,
    system_text: str = _SYSTEM_TEXT,
) -> Tuple[dict, int, int]:
    """Build the full conversation input tensors and return them.

    Returns:
        inputs           – dict of tensors ready for model(**inputs)
        assistant_start  – index of the first assistant token in the full sequence
        total_assistant  – number of assistant tokens (EOS-stripped)
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

    inputs_prompt = processor(text=[prompt_text], images=[img], return_tensors="pt")
    inputs_full = processor(text=[full_text], images=[img], return_tensors="pt")

    prompt_ids: List[int] = inputs_prompt["input_ids"][0].tolist()
    full_ids: List[int] = inputs_full["input_ids"][0].tolist()

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "VLM tokenization mismatch: full[: len(prompt)] != prompt. "
            "This may happen if apply_chat_template changes behavior between calls."
        )

    assistant_start = len(prompt_ids)

    # Strip trailing EOS from the assistant portion
    eos = processor.tokenizer.eos_token_id
    assistant_ids = full_ids[assistant_start:]
    if eos is not None:
        while assistant_ids and assistant_ids[-1] == eos:
            assistant_ids.pop()
    total_assistant = len(assistant_ids)

    return inputs_full, assistant_start, total_assistant, full_ids


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_forward(model, v_head, inputs: dict) -> torch.Tensor:
    """Run one forward pass and return per-token value predictions [seq_len].

    Uses output_hidden_states=True to extract the final layer hidden states,
    then applies the value head.
    """
    device = next(iter(model.parameters())).device
    inputs_on_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    outputs = model(**inputs_on_device, output_hidden_states=True)
    # hidden_states is a tuple of (num_layers+1) tensors, each [1, seq_len, hidden_size]
    # The last element is after the final layer norm.
    last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
    last_hidden = last_hidden[0]              # [seq_len, hidden_size]
    value_preds = v_head(last_hidden).squeeze(-1)  # [seq_len]
    return value_preds.float()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value_pred_to_length(value_pred: float, gamma: float, eps: float) -> Tuple[float, float]:
    y_hat = -float(torch.sigmoid(torch.tensor(float(value_pred))).item())
    y = min(-eps, max(-1.0 + eps, y_hat))
    length = math.log1p(y) / min(-eps, math.log(gamma))
    return y_hat, length


def _strip_trailing_eos(ids: List[int], eos_id: int | None) -> List[int]:
    if eos_id is None:
        return ids
    end = len(ids)
    while end > 0 and ids[end - 1] == eos_id:
        end -= 1
    return ids[:end]


def _fmt_token(s: str, width: int) -> str:
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    return (s[: width - 3] + "...") if len(s) > width else s.ljust(width)


def _viz_bar(rel_err_pct: float, viz_w: int, max_pct: float,
             fill_ch: str, empty_ch: str, clip_marker: str) -> str:
    raw = float(rel_err_pct)
    clipped = raw > max_pct
    x = max(0.0, min(raw, max_pct))
    filled = max(0, min(viz_w, int(round(x / max_pct * viz_w))))
    bar = fill_ch * filled + empty_ch * (viz_w - filled)
    if clipped and viz_w > 0:
        bar = bar[:-1] + clip_marker[0]
    return bar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Tree-value visualization for VLM LVM (Transformers-based, no server)."
    )
    ap.add_argument("--model-path", type=str, required=True,
                    help="LVM checkpoint directory (contains model.safetensors + value_head.safetensors)")
    ap.add_argument("--image-path", type=str, default=_DEFAULT_IMAGE_PATH,
                    help="Path to the image file (default: mathvista_idx63.png)")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model dtype (default: bfloat16)")
    ap.add_argument("--gamma", type=float, default=0.997,
                    help="LVM value→length discount gamma. length = ln(1+y_hat)/ln(gamma).")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Numerical epsilon for length conversion.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Max assistant steps to evaluate (-1 = all).")
    ap.add_argument("--start-step", type=int, default=0,
                    help="Start offset within assistant tokens.")
    ap.add_argument("--ignore-last-token", action="store_true", default=False,
                    help="Drop the last assistant token.")
    ap.add_argument("--token-text-col-width", type=int, default=10)
    ap.add_argument("--viz-rel-err", action="store_true", default=False)
    ap.add_argument("--viz-width", type=int, default=20)
    ap.add_argument("--viz-fill-char", type=str, default="█")
    ap.add_argument("--viz-empty-char", type=str, default="░")
    ap.add_argument("--viz-max-pct", type=float, default=100.0)
    ap.add_argument("--viz-clip-marker", type=str, default="▲")
    ap.add_argument("--include-prompt-last", action="store_true", default=True)
    ap.add_argument("--output-format", type=str, default="plain",
                    choices=["plain", "markdown"])
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    model, v_head = _load_model_and_vhead(args.model_path, dtype)
    processor = _load_processor(args.model_path)
    tokenizer = processor.tokenizer

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    image_path = args.image_path
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Tokenizing conversation with image: {image_path} …")
    inputs_full, assistant_start, total_assistant, full_ids = _tokenize_vl(
        processor, _QUESTION_TEXT, _ASSISTANT_TEXT, image_path
    )
    seq_len = inputs_full["input_ids"].shape[1]

    # Count image-pad tokens
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    num_image_tokens = sum(1 for t in full_ids if t == image_pad_id)

    if args.ignore_last_token and total_assistant > 0:
        total_assistant -= 1

    print(f"assistant_start    : {assistant_start}")
    print(f"total sequence len : {seq_len}  ({num_image_tokens} image-pad tokens)")
    print(f"assistant tokens   : {total_assistant}")

    # ------------------------------------------------------------------
    # Print dialogue header
    # ------------------------------------------------------------------
    if args.output_format == "markdown":
        print("\n### DIALOGUE\n")
        print(f"#### USER (image + text)\n\n[IMAGE: {Path(image_path).name}]\n")
        print(_QUESTION_TEXT)
        print("\n---\n")
        print("#### ASSISTANT\n")
        print(_ASSISTANT_TEXT)
        print("\n---\n")
    else:
        print("\n" + "=" * 80)
        print("DIALOGUE (VLM)")
        print("=" * 80)
        print(f"\n[USER (image: {Path(image_path).name})]\n")
        print(_QUESTION_TEXT)
        print("\n" + "-" * 80)
        print("\n[ASSISTANT]\n")
        print(_ASSISTANT_TEXT)
        print("\n" + "=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    print("Running forward pass …", flush=True)
    t0 = __import__("time").time()
    value_preds = _run_forward(model, v_head, inputs_full)  # [seq_len]
    elapsed = __import__("time").time() - t0
    print(f"Forward pass done in {elapsed:.1f}s. value_preds shape: {value_preds.shape}")

    # ------------------------------------------------------------------
    # Build rows
    # ------------------------------------------------------------------
    eos_id = tokenizer.eos_token_id
    assistant_ids = _strip_trailing_eos(full_ids[assistant_start:], eos_id)
    if args.ignore_last_token and len(assistant_ids) > 0:
        assistant_ids = assistant_ids[:-1]
    total_assistant = len(assistant_ids)

    start_step = max(0, min(args.start_step, max(0, total_assistant - 1)))
    available = max(0, total_assistant - start_step)
    max_steps = available if args.max_steps < 0 else min(args.max_steps, available)

    rows: List[Tuple[int, int, str, int]] = []
    if args.include_prompt_last and start_step == 0 and assistant_start > 0:
        prev_id = full_ids[assistant_start - 1]
        prev_str = tokenizer.decode([prev_id], skip_special_tokens=False)
        rows.append((-1, prev_id, prev_str, total_assistant))
    for k in range(max_steps):
        step = start_step + k
        tok_id = assistant_ids[step]
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        rows.append((step, tok_id, tok_str, total_assistant - (step + 1)))

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    text_w = max(5, args.token_text_col_width)
    viz_w = max(4, args.viz_width)
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

    last_assistant_step = total_assistant - 1
    last_token_note_printed = False

    for step, tok_id, tok_str, true_remaining in rows:
        idx = (assistant_start - 1) if step == -1 else (assistant_start + step)
        if idx < 0 or idx >= len(value_preds):
            print(f"  [WARNING: idx {idx} out of range {len(value_preds)}]")
            continue
        value_pred_raw = float(value_preds[idx].item())
        y_hat, l_pred = _value_pred_to_length(value_pred_raw, args.gamma, args.eps)
        diff = l_pred - float(true_remaining)
        denom = float(true_remaining) if true_remaining > 0 else 1.0
        rel_err = 100.0 * abs(diff) / denom

        tok_str_clean = _fmt_token(tok_str, text_w)
        if step == last_assistant_step and not last_token_note_printed:
            print(sep_line)

        line = (
            f"{step:5d}  {tok_id:7d}  {tok_str_clean}  "
            f"{value_pred_raw:10.4f}  {y_hat:10.6f}  {l_pred:9.3f}  {true_remaining:5d}  "
            f"{diff:9.3f}  {rel_err:9.2f}"
        )
        if args.viz_rel_err:
            bar = _viz_bar(rel_err, viz_w, args.viz_max_pct or 100.0,
                           (args.viz_fill_char or "█")[0],
                           (args.viz_empty_char or "░")[0],
                           (args.viz_clip_marker or "▲")[0])
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
