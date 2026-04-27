from __future__ import annotations

from typing import Any, List, Set


def get_eos_token_ids(req: Any) -> Set[int]:
    """Best-effort extraction of EOS token ids for a request.

    SGLang may carry EOS in multiple places:
    - `req.eos_token_ids`: a Set[int] from `ModelConfig.hf_eos_token_id` (preferred)
    - `req.tokenizer.eos_token_id`: tokenizer-defined EOS id
    """
    eos: Set[int] = set()

    eos_ids = getattr(req, "eos_token_ids", None)
    if eos_ids:
        if isinstance(eos_ids, set):
            eos |= {int(x) for x in eos_ids}
        elif isinstance(eos_ids, (list, tuple)):
            eos |= {int(x) for x in eos_ids}
        else:
            try:
                eos.add(int(eos_ids))
            except Exception as exc:
                raise ValueError(f"Invalid LenVM eos_token_ids value: {eos_ids!r}") from exc

    tok = getattr(req, "tokenizer", None)
    tok_eos = getattr(tok, "eos_token_id", None) if tok is not None else None
    if tok_eos is not None:
        try:
            eos.add(int(tok_eos))
        except Exception as exc:
            raise ValueError(f"Invalid LenVM tokenizer eos_token_id: {tok_eos!r}") from exc

    return eos


def force_eos_value_zero(token_ids: List[int], token_values: List[float], req: Any) -> None:
    """Force EOS token's value to 0.0 in-place.

    Rationale: EOS represents "no further continuation", so its length/value should be 0
    for value-guided sampling.
    """
    if not token_ids or not token_values or len(token_ids) != len(token_values):
        return
    eos = get_eos_token_ids(req)
    if not eos:
        return
    for j, tid in enumerate(token_ids):
        try:
            if int(tid) in eos:
                token_values[j] = 0.0
        except Exception as exc:
            raise ValueError(f"Invalid LenVM token id while forcing EOS value to zero: {tid!r}") from exc

