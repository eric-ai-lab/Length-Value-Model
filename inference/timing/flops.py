"""Theoretical inference FLOPs estimator for LenVM vs baseline.

Uses the standard 2*N forward rule: a transformer forward pass over T tokens
costs ~2 * N_params * T FLOPs (matmul-dominated; ignores attention's O(S)
contribution at long context, fine for ballpark inference-overhead numbers).

Contrasting theoretical FLOPs ratio against measured wall-clock ratio
isolates GPU utilization loss from raw compute increase. Reviewers in PR #2
asked specifically for "inference FLOPs", so we surface both.
"""

from __future__ import annotations

from typing import Optional


# Well-known parameter counts (params, including embeddings/lm_head) for the
# Qwen2.5 family used in the LenVM paper. Update via --base-model-params /
# --lvm-model-params CLI args if you swap base/LenVM checkpoints.
MODEL_PARAMS_BILLIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.49,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.54,
    "Qwen/Qwen2.5-3B-Instruct": 3.09,
    "Qwen/Qwen2.5-7B-Instruct": 7.61,
    "Qwen/Qwen2.5-14B-Instruct": 14.8,
    "Qwen/Qwen2.5-32B-Instruct": 32.5,
    "Qwen/Qwen2.5-72B-Instruct": 72.7,
    # LenVM checkpoints are base-model + a thin value head; value head is
    # negligible (~1M params), so we charge the LenVM forward at the base
    # model's compute cost.
    "namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct": 1.54,
    "namezz/lvm-instruct-0327-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct": 1.54,
    "namezz/lvm-a-qwen2.5-7b-instruct-b-qwen2.5-0.5b-instruct": 0.49,
    "namezz/lvm-rel-a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct": 3.09,
}


def resolve_params(model_name_or_path: str) -> Optional[float]:
    """Return param count (in billions) for a model name or local path."""
    if model_name_or_path in MODEL_PARAMS_BILLIONS:
        return MODEL_PARAMS_BILLIONS[model_name_or_path]
    # Local paths like ./models/namezz/...; match by basename.
    for key, val in MODEL_PARAMS_BILLIONS.items():
        if model_name_or_path.endswith(key) or model_name_or_path.endswith(key.split("/")[-1]):
            return val
    return None


def forward_flops_per_token(n_params_billions: float) -> float:
    """2 * N params per token (decode-only forward, matmul-dominated)."""
    return 2.0 * n_params_billions * 1e9


def baseline_run_flops(prompt_tokens: int, output_tokens: int, base_params_b: float) -> float:
    """Vanilla decoding: one base-model forward over every prompt + generated token."""
    return forward_flops_per_token(base_params_b) * (prompt_tokens + output_tokens)


def lvm_run_flops(
    prompt_tokens: int,
    output_tokens: int,
    base_params_b: float,
    lvm_params_b: float,
    k_candidates: int,
) -> float:
    """LenVM-guided decoding total FLOPs.

    Per generated token, LenVM adds:
    - 1 forward over the LenVM (extend KV by the just-accepted token)
    - k forwards over the LenVM (score the top-k candidates)

    Prefill on the LenVM happens lazily as candidates are scored, so we
    fold it into output_tokens for an upper-bound estimate.
    """
    base = baseline_run_flops(prompt_tokens, output_tokens, base_params_b)
    lvm_per_step = forward_flops_per_token(lvm_params_b) * (1 + k_candidates)
    return base + lvm_per_step * output_tokens
