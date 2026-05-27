"""Triton helpers for LenVM entropy-based guidance skips."""

from __future__ import annotations

import os
from typing import Optional

import torch


_DISABLE_TRITON_ENTROPY = os.getenv("LVM_DISABLE_TRITON_ENTROPY", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_TRITON_AVAILABLE: Optional[bool] = None


def _fallback_entropy_from_probs(
    probs: torch.Tensor,
    rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    p = probs.index_select(0, rows) if rows is not None else probs
    p = p.float()
    s = p.sum(dim=-1)
    p = p / s.clamp(min=1e-20).view(-1, 1)
    return torch.special.entr(p).sum(dim=-1)


def _next_power_of_2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def full_vocab_entropy_from_probs(
    probs: torch.Tensor,
    rows: Optional[torch.Tensor] = None,
    *,
    block_size: int = 4096,
) -> torch.Tensor:
    """Compute entropy over the full vocabulary distribution.

    The input is a probability tensor, not logits. To match the existing code
    exactly enough for thresholding, rows are renormalized by their full-vocab
    sum:

        H(p / sum(p)) = log(sum(p)) - sum(p * log(p)) / sum(p)

    Accumulation is FP32. This avoids materializing an extra selected
    [n_rows, vocab] tensor and the intermediate `entr` tensor.
    """

    if (
        _DISABLE_TRITON_ENTROPY
        or not probs.is_cuda
        or probs.dim() != 2
        or probs.shape[-1] == 0
    ):
        return _fallback_entropy_from_probs(probs, rows)

    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is False:
        return _fallback_entropy_from_probs(probs, rows)

    try:
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_AVAILABLE = False
        return _fallback_entropy_from_probs(probs, rows)

    _TRITON_AVAILABLE = True

    n_total_rows = int(probs.shape[0])
    vocab_size = int(probs.shape[1])
    if rows is None:
        n_rows = n_total_rows
        rows_t = torch.empty((0,), dtype=torch.int64, device=probs.device)
        has_rows = False
    else:
        rows_t = rows.to(device=probs.device, dtype=torch.int64)
        n_rows = int(rows_t.numel())
        has_rows = True
        if n_rows == 0:
            return torch.empty((0,), dtype=torch.float32, device=probs.device)

    block_size = int(block_size)
    if block_size <= 0:
        block_size = 4096
    n_blocks = triton.cdiv(vocab_size, block_size)
    reduce_block = _next_power_of_2(n_blocks)

    partial_sum = torch.empty((n_rows, n_blocks), dtype=torch.float32, device=probs.device)
    partial_p_log_p = torch.empty_like(partial_sum)
    out = torch.empty((n_rows,), dtype=torch.float32, device=probs.device)

    try:
        _entropy_partial_kernel[(n_rows, n_blocks)](
            probs,
            rows_t,
            partial_sum,
            partial_p_log_p,
            vocab_size,
            int(probs.stride(0)),
            n_blocks,
            HAS_ROWS=has_rows,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        _entropy_reduce_kernel[(n_rows,)](
            partial_sum,
            partial_p_log_p,
            out,
            n_blocks,
            BLOCK_N=reduce_block,
            num_warps=8,
        )
        return out
    except Exception:
        _TRITON_AVAILABLE = False
        return _fallback_entropy_from_probs(probs, rows)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _entropy_partial_kernel(
        probs,
        rows,
        partial_sum,
        partial_p_log_p,
        vocab_size: tl.constexpr,
        row_stride: tl.constexpr,
        n_blocks: tl.constexpr,
        HAS_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_pos = tl.program_id(0)
        block_id = tl.program_id(1)
        if HAS_ROWS:
            row = tl.load(rows + row_pos)
        else:
            row = row_pos
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        p = tl.load(probs + row * row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        p_log_p = tl.where(p > 0.0, p * tl.log(p), 0.0)
        out_offset = row_pos * n_blocks + block_id
        tl.store(partial_sum + out_offset, tl.sum(p, axis=0))
        tl.store(partial_p_log_p + out_offset, tl.sum(p_log_p, axis=0))

    @triton.jit
    def _entropy_reduce_kernel(
        partial_sum,
        partial_p_log_p,
        out,
        n_blocks: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < n_blocks
        s = tl.load(partial_sum + row * n_blocks + offsets, mask=mask, other=0.0)
        plogp = tl.load(partial_p_log_p + row * n_blocks + offsets, mask=mask, other=0.0)
        sum_p = tl.sum(s, axis=0)
        sum_p_log_p = tl.sum(plogp, axis=0)
        ent = tl.where(sum_p > 0.0, tl.log(sum_p) - sum_p_log_p / sum_p, 0.0)
        tl.store(out + row, ent)

except Exception:
    # Import-time fallback for CPU-only environments and Triton import issues.
    _TRITON_AVAILABLE = False
