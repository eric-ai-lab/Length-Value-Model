from __future__ import annotations

import math
import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

import requests
import torch

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils.common import dynamic_import
from sglang.srt.server_args import get_global_server_args
from sglang.srt.lvm.lvm_value_utils import force_eos_value_zero, get_eos_token_ids

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

logger = logging.getLogger(__name__)


def _get_req_custom_params(req: Any) -> dict[str, Any]:
    if req is None:
        return {}
    custom_params = getattr(getattr(req, "sampling_params", None), "custom_params", None)
    return custom_params if isinstance(custom_params, dict) else {}


def _get_kwarg_or_custom_param(kwargs: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = kwargs.get(name)
        if value is not None:
            return value

    custom_params = _get_req_custom_params(kwargs.get("req"))
    for name in names:
        value = custom_params.get(name)
        if value is not None:
            return value

    return None


def _default_guidance_fn(
    token_probs: List[float],
    token_values: List[float],
    token_ids: List[int],
    **kwargs: Any,
) -> List[float]:
    return token_probs


def _parse_required_float(raw_value: Any, field_name: str, *, default: Optional[float] = None) -> float:
    if raw_value is None:
        if default is None:
            raise ValueError(f"Missing required LenVM parameter: {field_name}")
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid LenVM parameter '{field_name}': expected a float, got {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Invalid LenVM parameter '{field_name}': expected a finite float, got {raw_value!r}"
        )
    return value


def _get_generated_len(req: Any) -> int:
    out = getattr(req, "output_ids", None) if req is not None else None
    if out is None:
        return 0
    try:
        return len(out)
    except Exception as exc:
        raise ValueError("Failed to read req.output_ids for LenVM guidance") from exc


def _extract_value_scale(kwargs: dict, default: float = 1.0) -> float:
    scale = _get_kwarg_or_custom_param(kwargs, "value_scale", "scale")
    return _parse_required_float(scale, "value_scale/scale", default=default)


def _extract_value_min(kwargs: dict, default: Optional[float] = None) -> Optional[float]:
    """Extract minimum expected-value threshold below which guidance is skipped.

    Reads from kwargs["value_min"] / custom_params["value_min"].
    Returns None if not set (no threshold → always apply guidance).
    """
    v = _get_kwarg_or_custom_param(kwargs, "value_min")
    if v is None:
        return default
    return _parse_required_float(v, "value_min")


def _extract_value_mode(kwargs: dict, default: str = "mul") -> str:
    mode = _get_kwarg_or_custom_param(kwargs, "mode", "value_mode")
    mode = (mode or default).strip().lower()
    valid_modes = {"mul", "linear", "exp", "length_mul", "centered_exp", "value_bias"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid LenVM parameter 'mode/value_mode': expected one of {sorted(valid_modes)}, got {mode!r}"
        )
    return mode


def _extract_length_gamma(kwargs: dict, default: float = 0.997) -> float:
    """Extract gamma used for value->length mapping.

    We follow repo README convention (default 0.997) and allow overriding via:
      - kwargs["gamma"] / kwargs["length_gamma"]
      - req.sampling_params.custom_params["gamma"] / ["length_gamma"]
    """
    gamma = _get_kwarg_or_custom_param(kwargs, "length_gamma", "gamma")
    gamma = _parse_required_float(gamma, "length_gamma/gamma", default=default)
    if not (0.0 < gamma < 1.0):
        raise ValueError(
            f"Invalid LenVM parameter 'length_gamma/gamma': expected a value in (0, 1), got {gamma!r}"
        )
    return gamma


def _extract_value_constraint(kwargs: dict, default: str = "eq") -> str:
    """Extract constraint type: eq/ge/le (equals / >= / <=)."""
    c = _get_kwarg_or_custom_param(kwargs, "value_constraint", "constraint", "cmp", "op")
    c = (c or default).strip().lower()
    if c in {"=", "==", "eq", "equal", "equals"}:
        return "eq"
    if c in {">", ">=", "ge", "gte", "greater_equal"}:
        return "ge"
    if c in {"<", "<=", "le", "lte", "less_equal"}:
        return "le"
    if c in {"eq_soft", "soft", "equal_soft"}:
        return "eq_soft"
    raise ValueError(
        f"Invalid LenVM parameter 'value_constraint/constraint/cmp/op': got {c!r}"
    )


def _extract_target_value(kwargs: dict) -> Optional[float]:
    """Extract target value v_tgt in [0,1], from either target_value or target_length.

    For target_length, we support two semantics (default: total):
      - remaining: target_length is remaining tokens, does not change with decoding
      - total: target_length is the *total* new-token budget; we convert to remaining via
          remaining = max(target_length - len(req.output_ids), 0)

    Length->value mapping follows README convention:
      v = 1 - gamma^length
    where gamma in (0,1).
    """
    tgt_v = _get_kwarg_or_custom_param(kwargs, "target_value")
    tgt_len = _get_kwarg_or_custom_param(kwargs, "target_length")
    tgt_len_mode = _get_kwarg_or_custom_param(kwargs, "target_length_mode", "length_mode")

    if tgt_v is not None:
        v = _parse_required_float(tgt_v, "target_value")
        return v

    if tgt_len is None:
        return None
    length = _parse_required_float(tgt_len, "target_length")

    # Optional: interpret target_length as total budget and convert to remaining.
    mode = (tgt_len_mode or "total").strip().lower()
    if mode in {"total", "budget"}:
        req = kwargs.get("req")
        out = getattr(req, "output_ids", None) if req is not None else None
        generated = _get_generated_len(req)
        length = max(length - float(generated), 0.0)

    gamma = _extract_length_gamma(kwargs, default=0.997)
    # v = 1 - gamma^length ; note gamma^length may overflow for huge length, guard with exp.
    try:
        v = 1.0 - math.exp(length * math.log(gamma))
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"Failed to convert target_length={length!r} into LenVM target_value"
        ) from exc
    if not math.isfinite(v):
        raise ValueError(
            f"Non-finite LenVM target_value computed from target_length={length!r}"
        )
    return v


def lvm_hard_value_constraint_guidance(
    token_probs: List[float],
    token_values: List[float],
    token_ids: List[int],
    **kwargs: Any,
) -> List[float]:
    """Hard-decoding guidance based on candidate values.

    Controlled via req.sampling_params.custom_params (or kwargs):
      - target_value: float in [0,1] (preferred), OR
      - target_length: float, with length->value v = 1 - gamma^length
      - gamma / length_gamma: optional for length conversion (default 0.997)
      - value_constraint / constraint: one of eq/ge/le

    Output is a one-hot distribution over the provided candidates.
    """
    if not token_probs or len(token_probs) != len(token_values):
        return token_probs

    v_tgt = _extract_target_value(kwargs)
    if v_tgt is None:
        return token_probs

    c = _extract_value_constraint(kwargs, default="eq")
    n = len(token_values)

    # For debugging / length semantics introspection.
    req = kwargs.get("req")
    generated_len = 0
    if req is not None:
        generated_len = _get_generated_len(req)

    # Pre-compute candidate min/max for edge handling.
    min_v = min(token_values)
    max_v = max(token_values)

    # Choose index according to constraint.
    chosen = 0

    if c == "eq":
        # Compare in length space: length = log(1 - v) / log(gamma)
        # This avoids the non-linear compression near v=1 in value space.
        gamma = _extract_length_gamma(kwargs, default=0.997)
        log_gamma = math.log(gamma)

        def _value_to_length(v: float) -> float:
            v = float(v)
            v = max(min(v, 1.0 - 1e-9), 1e-9)  # clamp to avoid log(0)
            return math.log(1.0 - v) / log_gamma

        tgt_len_space = _value_to_length(v_tgt)
        best_diff = float("inf")
        best_prob = -1.0
        for i in range(n):
            d = abs(_value_to_length(token_values[i]) - tgt_len_space)
            p = float(token_probs[i])
            if d < best_diff or (d == best_diff and p > best_prob):
                best_diff = d
                best_prob = p
                chosen = i
    elif c == "ge":
        # When v_tgt ≈ 0, v >= v_tgt holds for all candidates; pick minimum value to
        # prefer EOS / short continuation and finish quickly after hitting the budget.
        # Otherwise maximize value to grow toward the target.
        if abs(float(v_tgt)) <= 1e-12:
            best_v = float("inf")
            best_prob = -1.0
            for i in range(n):
                v = float(token_values[i])
                p = float(token_probs[i])
                if v < best_v or (v == best_v and p > best_prob):
                    best_v = v
                    best_prob = p
                    chosen = i
        else:
            best_v = float("-inf")
            best_prob = -1.0
            for i in range(n):
                v = float(token_values[i])
                p = float(token_probs[i])
                if v > best_v or (v == best_v and p > best_prob):
                    best_v = v
                    best_prob = p
                    chosen = i
    elif c == "eq_soft":
        # Soft eq: exponential tilting so E[v] = v_tgt.
        # Fallback to one-hot on boundary token if target is outside candidate range.
        if v_tgt <= min_v:
            # Clamp low: pick token with minimum value (tie-break by prob).
            best_v = float("inf")
            best_prob = -1.0
            for i in range(n):
                v = float(token_values[i])
                p = float(token_probs[i])
                if v < best_v or (v == best_v and p > best_prob):
                    best_v = v
                    best_prob = p
                    chosen = i
        elif v_tgt >= max_v:
            # Clamp high: pick token with maximum value (tie-break by prob).
            best_v = float("-inf")
            best_prob = -1.0
            for i in range(n):
                v = float(token_values[i])
                p = float(token_probs[i])
                if v > best_v or (v == best_v and p > best_prob):
                    best_v = v
                    best_prob = p
                    chosen = i
        else:
            # Exponential tilting: find lambda s.t. E_{p'}[v] = v_tgt.
            probs_t = torch.tensor(token_probs, dtype=torch.float64)
            values_t = torch.tensor(token_values, dtype=torch.float64)
            probs_t = torch.clamp(probs_t, min=0.0)
            prob_sum = probs_t.sum()
            if prob_sum > 0:
                probs_t = probs_t / prob_sum
            log_probs_t = torch.log(probs_t)
            v_ref = values_t.min()
            dv = values_t - v_ref
            cur_exp_t = (probs_t * values_t).sum().item()
            v_range = max_v - min_v
            tol = max(1e-10, v_range * 1e-6)
            target_eps = max(1e-12, v_range * 1e-6)
            tgt = min(max(float(v_tgt), min_v + target_eps), max_v - target_eps)

            def _tilted_stats_soft(lam: float):
                logits = log_probs_t + dv * lam
                m = torch.max(logits)
                if not torch.isfinite(m):
                    return cur_exp_t, 0.0
                w = torch.exp(logits - m)
                s = w.sum()
                if s <= 0:
                    return cur_exp_t, 0.0
                w = w / s
                mean = (w * values_t).sum()
                var = torch.clamp((w * values_t * values_t).sum() - mean * mean, min=0.0)
                return mean.item(), var.item()

            if tgt > cur_exp_t:
                lo_lam, hi_lam = 0.0, 1.0
                while _tilted_stats_soft(hi_lam)[0] < tgt and hi_lam < 100000:
                    hi_lam *= 2.0
            else:
                hi_lam, lo_lam = 0.0, -1.0
                while _tilted_stats_soft(lo_lam)[0] > tgt and lo_lam > -100000:
                    lo_lam *= 2.0

            lam = 0.0
            for _ in range(20):
                mid_exp, mid_var = _tilted_stats_soft(lam)
                if mid_exp < tgt:
                    lo_lam = lam
                else:
                    hi_lam = lam
                err = mid_exp - tgt
                if abs(err) <= tol:
                    break
                if mid_var > 1e-16:
                    lam_new = lam - err / mid_var
                    if lo_lam < lam_new < hi_lam:
                        lam = lam_new
                        continue
                lam = (lo_lam + hi_lam) / 2.0

            final_logits = log_probs_t + dv * lam
            m = torch.max(final_logits)
            if torch.isfinite(m):
                w = torch.exp(final_logits - m)
                s = w.sum()
                if s > 0:
                    soft_probs = (w / s).tolist()
                    logger.debug(
                        "[lvm_guided_sampling] soft constraint=%s generated_len=%d "
                        "v_tgt=%.6f cand_v=[%.6f,%.6f] lam=%.4f",
                        c,
                        generated_len,
                        float(v_tgt),
                        float(min_v),
                        float(max_v),
                        lam,
                    )
                    return soft_probs
            # Fallback: return original probs if tilting fails.
            return token_probs
    elif c == "le":  # c == "le"
        # Extreme policy: always minimize value (tie-break by prob).
        best_v = float("inf")
        best_prob = -1.0
        for i in range(n):
            v = float(token_values[i])
            p = float(token_probs[i])
            if v < best_v or (v == best_v and p > best_prob):
                best_v = v
                best_prob = p
                chosen = i
    else:
        raise ValueError(f"Invalid value constraint: {c}")

    logger.debug(
        "[lvm_guided_sampling] hard constraint=%s generated_len=%d "
        "v_tgt=%.6f cand_v=[%.6f,%.6f] chosen_i=%d chosen_v=%.6f chosen_p=%.6f",
        c, generated_len, float(v_tgt), float(min_v), float(max_v),
        chosen, float(token_values[chosen]), float(token_probs[chosen]),
    )

    out = [0.0] * n
    out[int(chosen)] = 1.0
    return out


def lvm_combined_guidance(
    token_probs: List[float],
    token_values: List[float],
    token_ids: List[int],
    **kwargs: Any,
) -> List[float]:
    """Combine hard constraint decoding + expectation guidance.

    Routing rule:
    - If the request provides a target threshold (target_value/target_length),
      apply hard decoding via `lvm_hard_value_constraint_guidance`.
    - Otherwise, fall back to `lvm_expectation_guidance`.

    This lets you use a single `--lvm-guided-fn` while supporting both behaviors
    on a per-request basis.
    """
    v_tgt = _extract_target_value(kwargs)
    if v_tgt is not None:
        logger.debug("Beginning hard decoding...")
        return lvm_hard_value_constraint_guidance(
            token_probs=token_probs,
            token_values=token_values,
            token_ids=token_ids,
            **kwargs,
        )
    logger.debug("Beginning expectation guidance...")
    return lvm_expectation_guidance(
        token_probs=token_probs,
        token_values=token_values,
        token_ids=token_ids,
        **kwargs,
    )


def lvm_expectation_guidance(
    token_probs: List[float],
    token_values: List[float],
    token_ids: List[int],
    **kwargs: Any,
) -> List[float]:
    """Adjust probs so E[v] matches scale * E[v] under original probs.

    Uses exponential tilting: p'(i) ∝ p(i) * exp(lambda * v(i)).
    """
    if not token_probs or len(token_probs) != len(token_values):
        return token_probs

    probs = torch.tensor(token_probs, dtype=torch.float64)
    values = torch.tensor(token_values, dtype=torch.float64)
    probs = torch.clamp(probs, min=0)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        return token_probs
    probs = probs / prob_sum

    cur_exp = (probs * values).sum().item()

    # If the current expected value is below the minimum threshold, skip guidance.
    value_min = _extract_value_min(kwargs)
    if value_min is not None and cur_exp < value_min:
        return probs.tolist()

    scale = _extract_value_scale(kwargs, default=1.0)
    mode = _extract_value_mode(kwargs, default="mul")

    min_v = values.min().item()
    max_v = values.max().item()

    if mode == "centered_exp":
        raise Exception("Not implemented, cpu should not be used")
        # Logit reweighting: p'(i) ∝ p(i) * exp(s * (g(i) - E[g])).
        # This preserves normalization while smoothly shifting mass toward higher g
        # for s>0 (and lower g for s<0).
        # eps = 1e-12
        # g = torch.clamp(values, min=eps, max=1.0 - eps)
        # mu = (probs * g).sum()
        weights = probs * torch.exp(values * float(scale))
        z = weights.sum()
        if z <= 0:
            return probs.tolist()
        target = (weights / z).tolist()
        logger.debug("LenVM centered_exp target=%s", target)
        return target
    if mode == "value_bias":
        raise Exception("Not implemented, cpu should not be used")
        # Convert sigmoid values back to logit space and use as bias.
        # logit(v) = log(v / (1 - v))
        # This allows stronger influence when v is near 0 or 1.
        eps = 1e-7  # Suitable for float32/float64 precision
        v_clamped = torch.clamp(values, min=eps, max=1.0 - eps)

        # Use torch.logit if available (PyTorch >= 1.7), otherwise manual computation
        if hasattr(torch, 'logit'):
            logit_values = torch.logit(v_clamped, eps=eps)
        else:
            logit_values = torch.log(v_clamped / (1.0 - v_clamped))

        log_probs = torch.log(probs)
        logits = log_probs + logit_values * float(scale)

        m = torch.max(logits)
        if not torch.isfinite(m):
            return probs.tolist()
        weights = torch.exp(logits - m)
        z = weights.sum()
        if z <= 0:
            return probs.tolist()
        target = (weights / z).tolist()
        return target
    if mode == "length_mul":
        # Length-space scaling:
        # 1) compute current expected value mu_v = E[v]
        # 2) map mu_v -> length via README mapping:
        #       l = ln(1 - v) / ln(gamma)   where gamma in (0,1)
        # 3) scale length: l' = scale * l
        # 4) map back to target expected value:
        #       v' = 1 - gamma^{l'} = 1 - exp(l' * ln(gamma))
        #
        # Then we find lambda for exponential tilting so that E_{p'}[v] == v'.
        gamma = _extract_length_gamma(kwargs, default=0.997)
        log_gamma = math.log(gamma)  # negative
        # Guard: keep mu_v in (0,1) for ln(1 - v).
        mu_v = float(cur_exp)
        mu_v = min(max(mu_v, 1e-15), 1.0 - 1e-15)
        # Convert expectation to length and scale.
        l_cur = math.log1p(-mu_v) / log_gamma
        l_tgt = float(scale) * l_cur
        # Convert back to value in (0,1).
        v_tgt = 1.0 - math.exp(l_tgt * log_gamma)
        target = min(max(v_tgt, min_v), max_v)
    elif mode == "linear":
        if scale >= 1:
            target = cur_exp + (scale - 1.0) * (max_v - cur_exp)
        else:
            target = cur_exp - (1.0 - scale) * (cur_exp - min_v)
    elif mode == "exp":
        if max_v - min_v < 1e-8:
            target = cur_exp
        else:
            cur_norm = (cur_exp - min_v) / (max_v - min_v)
            cur_norm = min(max(cur_norm, 0.0), 1.0)
            target_norm = 1.0 - (1.0 - cur_norm) ** max(scale, 0.0)
            target = min_v + target_norm * (max_v - min_v)
    else:
        if scale <= 0:
            return probs.tolist()
        target = cur_exp * scale
    target = min(max(target, min_v), max_v)

    if max_v - min_v < 1e-8 or abs(target - cur_exp) < 1e-8:
        return probs.tolist()

    # Numerics: exact boundary targets (min_v/max_v) require lambda -> +/- inf in general.
    # To avoid "no finite root" issues in bracketing, pull the target slightly into (min_v, max_v).
    v_range = max_v - min_v
    target_eps = max(1e-12, v_range * 1e-6)
    target = min(max(target, min_v + target_eps), max_v - target_eps)

    log_probs = torch.log(probs)  # zeros -> -inf, handled naturally in log-sum-exp
    v_ref = values.min()  # shift values to reduce exp magnitudes; cancels in normalization
    dv = values - v_ref

    def _tilted_stats(lam: float) -> tuple[float, float]:
        """Return (E[v], Var[v]) under tilted distribution at lam.

        For exponential tilting, d/dlam E[v] = Var[v] (always >= 0).
        """
        logits = log_probs + dv * lam
        m = torch.max(logits)
        if not torch.isfinite(m):
            return cur_exp, 0.0
        w = torch.exp(logits - m)
        s = w.sum()
        if s <= 0:
            return cur_exp, 0.0
        w = w / s
        mean = (w * values).sum()
        mean2 = (w * (values * values)).sum()
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        return mean.item(), var.item()

    if target > cur_exp:
        lo, hi = 0.0, 1.0
        while _tilted_stats(hi)[0] < target and hi < 100000:
            hi *= 2.0
    else:
        hi, lo = 0.0, -1.0
        while _tilted_stats(lo)[0] > target and lo > -100000:
            lo *= 2.0

    # Fast root finding: Newton step with bisection fallback.
    # Maintain invariant: E(lo) <= target <= E(hi).
    tol = max(1e-10, v_range * 1e-6)
    lam = 0.0
    for _ in range(20):
        mid = lam
        mid_exp, mid_var = _tilted_stats(mid)

        # tighten bracket
        if mid_exp < target:
            lo = mid
        else:
            hi = mid

        err = mid_exp - target
        if abs(err) <= tol:
            break

        # Newton step: lam_{t+1} = lam_t - (E[v]-target)/Var[v]
        if mid_var > 1e-16:
            lam_new = mid - err / mid_var
            # keep it in bracket to preserve monotonic root guarantees
            if lo < lam_new < hi:
                lam = lam_new
                continue

        # fallback: bisection
        lam = (lo + hi) / 2.0

    lam_star = lam

    final_logits = log_probs + (values - v_ref) * lam_star
    m = torch.max(final_logits)
    if not torch.isfinite(m):
        return probs.tolist()
    final_weights = torch.exp(final_logits - m)
    final_sum = final_weights.sum()
    if final_sum <= 0:
        return probs.tolist()
    final_probs = final_weights / final_sum
    logger.debug(
        "token_probs=%s token_values=%s E[v]=%.6f scale=%s target=%.6f mode=%s final_probs=%s",
        token_probs, token_values, cur_exp, scale, target, mode, final_probs.tolist(),
    )
    return final_probs.tolist()


def _extract_token_temperature_scale(req: Any) -> Optional[dict]:
    """Extract per-token temperature scale map from custom_params.

    Supports two forms:
      1. token_temperature_scale: {token_id: divisor, ...}
         - divisor > 1 → token more likely (temperature reduced)
         - divisor < 1 → token less likely (temperature increased)
      2. boosted_token_ids: [id, ...] + token_temp_divisor: float
         - convenience form; all listed tokens share the same divisor

    Returns a dict {int token_id: float divisor} or None if not set.
    Result is cached on the req object to avoid re-parsing every decode step.
    """
    # Fast path: cached result from a previous decode step.
    cached = getattr(req, "_lvm_token_temp_scale", None)
    if cached is not None:
        # Sentinel: False means "already checked, nothing to do".
        return None if cached is False else cached

    custom_params = _get_req_custom_params(req)
    if not custom_params:
        try:
            setattr(req, "_lvm_token_temp_scale", False)
        except Exception:
            pass
        return None

    result: Optional[dict] = None

    scale_map = custom_params.get("token_temperature_scale")
    if scale_map is not None:
        if not isinstance(scale_map, dict):
            raise ValueError("LenVM custom param 'token_temperature_scale' must be a dict")
        parsed = {}
        for k, v in scale_map.items():
            try:
                token_id = int(k)
                divisor = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid LenVM token_temperature_scale entry: {k!r} -> {v!r}"
                ) from exc
            parsed[token_id] = divisor
        result = parsed or None
    else:
        ids = custom_params.get("boosted_token_ids")
        divisor = custom_params.get("token_temp_divisor")
        if ids is not None or divisor is not None:
            if ids is None or divisor is None:
                raise ValueError(
                    "LenVM custom params 'boosted_token_ids' and 'token_temp_divisor' must be provided together"
                )
            d = _parse_required_float(divisor, "token_temp_divisor")
            parsed = {}
            for tid in ids:
                try:
                    parsed[int(tid)] = d
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid LenVM boosted_token_ids entry: {tid!r}"
                    ) from exc
            result = parsed or None

    try:
        setattr(req, "_lvm_token_temp_scale", result if result is not None else False)
    except Exception:
        pass
    return result


def _load_guidance_fn(spec: Optional[str]) -> Callable[..., List[float]]:
    if not spec:
        return _default_guidance_fn

    if ":" in spec:
        module_part, fn_name = spec.rsplit(":", 1)
        if os.path.isfile(module_part):
            spec_obj = importlib.util.spec_from_file_location("lvm_guided_fn", module_part)
            if spec_obj is None or spec_obj.loader is None:
                raise ValueError(f"Failed to load guidance function from file: {module_part}")
            module = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(module)
        else:
            module = importlib.import_module(module_part)
        fn = getattr(module, fn_name)
    else:
        fn = dynamic_import(spec)

    if not callable(fn):
        raise ValueError(f"Guidance function '{spec}' is not callable")
    return fn


@dataclass
class LvmGuidedConfig:
    url: Optional[str]
    timeout: float
    bypass_cache: bool
    fn_spec: Optional[str]


@dataclass
class PendingLvmResult:
    """Intermediate state produced by _build_pending() and consumed by apply().

    Carries filtered candidate lists, the cloned probs tensor (with deterministic
    rows already filled), and optional GPU tensors for the fast guidance path.
    """

    req_list: List[Any]
    device: torch.device
    # probs.clone() with deterministic (single-candidate) rows already zeroed/filled.
    # None means there is nothing to do (all rows were deterministic or skipped).
    guided: Optional[torch.Tensor]
    send_batch_indices: List[int]
    prefix_ids_send: List[List[int]]
    candidate_ids_send: List[List[int]]
    candidate_probs_send: List[List[float]]
    # GPU tensors for the fast path (only set when the guidance function can use the
    # expectation-guidance GPU path and all send indices come from the top-k path,
    # not top-k-all).
    # Tuple of (padded_probs [B_send, K_max] float32, padded_ids [B_send, K_max] int64,
    #           valid_mask [B_send, K_max] bool) — all on the main device.
    gpu_candidates: Optional[tuple] = None


class LvmGuidedSampler:
    def __init__(self, config: LvmGuidedConfig, *, model_runner=None):
        self.config = config
        self._session = requests.Session()
        self._fn = _load_guidance_fn(config.fn_spec)
        self._decode_model_runner = model_runner
        self._inproc = None

    @staticmethod
    def from_server_args(server_args, model_runner=None) -> Optional["LvmGuidedSampler"]:
        enable_lvm = getattr(server_args, "enable_lvm_guided_sampling", False)
        enable_tts = getattr(server_args, "enable_token_temp_scale", False)
        if not enable_lvm and not enable_tts:
            return None
        url = getattr(server_args, "lvm_guided_url", None)
        # url can be None when using in-proc provider or token-temp-scale-only mode.
        if enable_lvm and not url and not getattr(server_args, "lvm_guided_inproc", False):
            raise ValueError(
                "enable_lvm_guided_sampling requires --lvm-guided-url (or --lvm-guided-inproc)"
            )
        config = LvmGuidedConfig(
            url=url,
            timeout=float(getattr(server_args, "lvm_guided_timeout", 5.0)),
            bypass_cache=bool(getattr(server_args, "lvm_guided_bypass_cache", False)),
            fn_spec=getattr(server_args, "lvm_guided_fn", None),
        )
        return LvmGuidedSampler(config, model_runner=model_runner)

    def _get_inproc_provider(self):
        """Lazily initialize an in-proc tree_value provider (second model runner)."""
        if self._inproc is not None:
            return self._inproc

        if self._decode_model_runner is not None:
            server_args = (
                getattr(self._decode_model_runner, "server_args", None)
                or get_global_server_args()
            )
        else:
            server_args = get_global_server_args()
        if not getattr(server_args, "lvm_guided_inproc", False):
            self._inproc = False
            return self._inproc
        if self._decode_model_runner is None:
            raise RuntimeError(
                "LenVM in-proc guidance was requested but decode model_runner is not available"
            )

        lvm_path = getattr(server_args, "lvm_guided_inproc_model_path", None)
        if not lvm_path:
            raise ValueError("--lvm-guided-inproc requires --lvm-guided-inproc-model-path")

        # Build a separate ModelConfig for the LVM embedding model.
        lvm_revision = getattr(server_args, "lvm_guided_inproc_model_revision", None)
        lvm_override = getattr(server_args, "lvm_guided_inproc_json_model_override_args", "{}")

        # Create a shallow copy of server_args with embedding mode enabled.
        # NOTE: ModelRunner uses a global server args singleton; we keep server_args consistent
        # and only change ModelConfig.model_override_args by passing it explicitly here.
        # We also force is_embedding=True semantics via ModelConfig init.
        lvm_model_config = ModelConfig(
            model_path=lvm_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=lvm_revision or server_args.revision,
            context_length=server_args.context_length,
            model_override_args=lvm_override,
            is_embedding=True,
            enable_multimodal=None,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            model_impl=server_args.model_impl,
            sampling_defaults=server_args.sampling_defaults,
            quantize_and_serve=server_args.quantize_and_serve,
            override_config_file=server_args.decrypted_config_file,
            is_multi_layer_eagle=False,
            encoder_only=server_args.encoder_only,
            language_only=server_args.language_only,
            disable_hybrid_swa_memory=server_args.disable_hybrid_swa_memory,
        )

        # Instantiate a second model runner in-process.
        # Important: mark it as draft_worker to avoid reinitializing distributed groups.
        from sglang.srt.model_executor.model_runner import ModelRunner

        # Give it independent KV pools by leaving req_to_token_pool / allocator as None.
        mem_fraction_static = float(
            getattr(server_args, "lvm_guided_inproc_mem_fraction_static", None)
            or server_args.mem_fraction_static
        )
        lvm_runner = ModelRunner(
            model_config=lvm_model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=self._decode_model_runner.gpu_id,
            tp_rank=self._decode_model_runner.tp_rank,
            tp_size=self._decode_model_runner.tp_size,
            moe_ep_rank=self._decode_model_runner.moe_ep_rank,
            moe_ep_size=self._decode_model_runner.moe_ep_size,
            pp_rank=self._decode_model_runner.pp_rank,
            pp_size=self._decode_model_runner.pp_size,
            nccl_port=self._decode_model_runner.dist_port,
            server_args=server_args,
            dp_rank=None,
            is_draft_worker=True,
        )

        from sglang.srt.lvm.lvm_inproc_runner import LvmInprocRunner
        lvm_incremental_runner = LvmInprocRunner(lvm_runner)

        from sglang.srt.configs.model_config import is_multimodal_model

        lvm_architectures = getattr(lvm_model_config.hf_config, "architectures", []) or []
        lvm_is_vlm = is_multimodal_model(lvm_architectures)
        expected_vlm_arch = "Qwen2_5_VLForLengthValueModel"
        has_qwen2_5_vl_arch = any(arch.startswith("Qwen2_5_VL") for arch in lvm_architectures)
        if hasattr(lvm_model_config.hf_config, "vision_config") and not has_qwen2_5_vl_arch:
            raise RuntimeError(
                "LenVM in-proc model exposes vision_config but did not declare a supported Qwen2.5-VL LenVM architecture. "
                f"Architectures: {lvm_architectures!r}. Refusing to fall back to a non-VLM branch."
            )
        if lvm_is_vlm and not has_qwen2_5_vl_arch:
            raise RuntimeError(
                "LenVM in-proc model resolved to a multimodal architecture that is not supported by the local LenVM VLM integration. "
                f"Architectures: {lvm_architectures!r}."
            )
        if has_qwen2_5_vl_arch:
            if expected_vlm_arch not in lvm_architectures:
                raise RuntimeError(
                    "Qwen2.5-VL LenVM checkpoint did not select the LenVM VLM architecture. "
                    "Expected hf_config.architectures to include `Qwen2_5_VLForLengthValueModel`, "
                    f"got {lvm_architectures!r}. Refusing to fall back to another branch."
                )
            actual_model_arch = type(lvm_runner.model).__name__
            if actual_model_arch != expected_vlm_arch:
                raise RuntimeError(
                    "Qwen2.5-VL LenVM checkpoint resolved to an unexpected runtime model class. "
                    f"Expected `{expected_vlm_arch}`, got `{actual_model_arch}`."
                )

        class _Inproc:
            def __init__(self, runner, incremental_runner, is_vlm: bool):
                self.runner = runner
                self.incremental_runner = incremental_runner
                self.is_vlm = is_vlm
                self._mrope_deltas: dict[str, torch.Tensor] = {}
                # Dedicated CUDA stream for LVM forward passes so the main stream
                # can overlap with LVM GPU compute (e.g. during inter-step CPU work).
                self.lvm_stream = torch.cuda.Stream(device=runner.device)
                # Event signalling that the non-blocking GPU→CPU embedding transfer
                # on lvm_stream is complete and the CPU tensors are safe to read.
                self.embed_ready = torch.cuda.Event()
                # Event signalling that prefix extend is complete
                self.extend_ready = torch.cuda.Event()

            def clean_stale_requests(self, active_rids: set):
                self.incremental_runner.cleanup_stale_rids(active_rids)
                if self.is_vlm and self._mrope_deltas:
                    for rid in [rid for rid in self._mrope_deltas if rid not in active_rids]:
                        del self._mrope_deltas[rid]

            def tree_value_extend(
                self, rids: List[str], prefix_ids: List[List[int]], reqs: List[Req]
            ):
                """Phase A: Extend prefix KV cache.
                
                Calculates the delta between the requested prefix and the currently
                cached prefix length, and runs the EXTEND forward pass.
                """
                new_tokens_list = []
                valid_rids = []
                mm_inputs_list = [] if self.is_vlm else None

                for rid, p_ids, req in zip(rids, prefix_ids, reqs):
                    cached_len = self.incremental_runner.kv_mgr.kv_len(rid)
                    target_len = len(p_ids)

                    if target_len < cached_len:
                        # Retraction due to beam search rollback: free excess slots
                        self.incremental_runner.kv_mgr.retract(rid, target_len)
                    elif target_len > cached_len:
                        # Extract the un-cached suffix
                        valid_rids.append(rid)
                        new_tokens_list.append(p_ids[cached_len:])
                        if self.is_vlm:
                            if cached_len == 0:
                                mm_input = getattr(req, "multimodal_inputs", None)
                                mm_inputs_list.append(mm_input)
                                if mm_input is not None:
                                    delta = getattr(mm_input, "mrope_position_delta", None)
                                    if delta is not None:
                                        self._mrope_deltas[rid] = delta
                            else:
                                mm_inputs_list.append(None)

                # Check if there's actual work to do
                if not valid_rids:
                    return

                # Ensure lvm_stream waits for default stream
                self.lvm_stream.wait_stream(torch.cuda.current_stream())

                with torch.cuda.stream(self.lvm_stream):
                    self.incremental_runner.extend_prefix_batch(
                        valid_rids,
                        new_tokens_list,
                        mm_inputs_list=mm_inputs_list,
                        mrope_deltas=self._mrope_deltas if self.is_vlm else None,
                    )
                    self.extend_ready.record(self.lvm_stream)

            def tree_value_launch(
                self, rids: List[str], candidate_ids: List[List[int]], gpu_candidates: Optional[tuple] = None
            ):
                """Start the LVM forward pass on lvm_stream (non-blocking).

                Returns a CPU tensor (or list of tensors) that is still being
                transferred from GPU. Call tree_value_collect() to wait.
                """
                # Make lvm_stream wait for any pending default-stream ops
                self.lvm_stream.wait_stream(torch.cuda.current_stream())

                with torch.cuda.stream(self.lvm_stream):
                    # We wait for extend to finish first (this is a stream-level sync)
                    self.lvm_stream.wait_event(self.extend_ready)
                    
                    embeddings = self.incremental_runner.eval_candidates_batch_gpu(
                        rids,
                        candidate_ids,
                        gpu_candidates=gpu_candidates,
                        mrope_deltas=self._mrope_deltas if self.is_vlm else None,
                    )
                    
                    # Non-blocking transfer: PCIe copy runs while we do CPU work
                    if isinstance(embeddings, list):
                        cpu_embeddings = [
                            t.to("cpu", non_blocking=True) for t in embeddings
                        ]
                    else:
                        cpu_embeddings = embeddings.to("cpu", non_blocking=True)
                    # Record event *inside* the stream context
                    self.embed_ready.record(self.lvm_stream)

                return cpu_embeddings

            def tree_value_collect(self, cpu_embeddings) -> List[List[float]]:
                """Wait for the embedding transfer started by tree_value_launch() and
                convert the CPU tensors to Python lists.
                """
                self.embed_ready.synchronize()
                out: List[List[float]] = []
                if isinstance(cpu_embeddings, list):
                    for t in cpu_embeddings:
                        out.append(t.float().tolist())
                else:
                    for i in range(cpu_embeddings.shape[0]):
                        out.append(cpu_embeddings[i].float().tolist())
                return out

            def tree_value_launch_gpu(
                self, rids: List[str], candidate_ids: List[List[int]], gpu_candidates: Optional[tuple] = None
            ):
                """Like tree_value_launch() but keeps embeddings on GPU (no PCIe copy)."""
                self.lvm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.lvm_stream):
                    self.lvm_stream.wait_event(self.extend_ready)
                    embeddings = self.incremental_runner.eval_candidates_batch_gpu(
                        rids,
                        candidate_ids,
                        gpu_candidates=gpu_candidates,
                        mrope_deltas=self._mrope_deltas if self.is_vlm else None,
                    )
                    # embeddings stay on GPU — no PCIe copy.
                    self.embed_ready.record(self.lvm_stream)
                
                return embeddings  # GPU tensor(s), not yet safe from default stream

            def tree_value_collect_gpu(self, gpu_embeddings):
                """Insert a stream dependency so the default stream waits for lvm_stream."""
                torch.cuda.current_stream().wait_event(self.embed_ready)
                return gpu_embeddings

            def tree_value(
                self, rids: List[str], prefix_ids: List[List[int]], candidate_ids: List[List[int]], reqs: List[Req], gpu_candidates: Optional[tuple] = None
            ):
                """Synchronous wrapper: launch + collect in one call."""
                self.tree_value_extend(rids, prefix_ids, reqs)
                cpu_embeddings = self.tree_value_launch(rids, candidate_ids, gpu_candidates=gpu_candidates)
                return self.tree_value_collect(cpu_embeddings)

        self._inproc = _Inproc(lvm_runner, lvm_incremental_runner, lvm_is_vlm)
        return self._inproc

    def _post_tree_value(
        self, rids: List[str], prefix_ids: List[List[int]], candidate_ids: List[List[int]], reqs: List[Req]
    ) -> Optional[List[List[float]]]:
        inproc = self._get_inproc_provider()
        if inproc not in (None, False):
            try:
                return inproc.tree_value(rids, prefix_ids, candidate_ids, reqs)
            except Exception as exc:
                raise RuntimeError("LenVM in-proc tree_value failed") from exc

        payload = {
            "input_ids": prefix_ids,
            "candidate_ids": candidate_ids,
            "bypass_cache": self.config.bypass_cache,
        }
        try:
            # `requests.post(json=...)` uses Python's stdlib json (slower) and incurs extra copies.
            # Use orjson if available (SGLang server already depends on it), and parse from bytes.
            if orjson is not None:
                body = orjson.dumps(payload)
                resp = self._session.post(
                    f"{self.config.url}/tree_value",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=self.config.timeout,
                )
            else:
                resp = self._session.post(
                    f"{self.config.url}/tree_value",
                    json=payload,
                    timeout=self.config.timeout,
                )
            resp.raise_for_status()
            if orjson is not None:
                data = orjson.loads(resp.content)
            else:
                data = resp.json()
        except Exception as exc:
            raise RuntimeError("LenVM tree_value HTTP call failed") from exc

        if isinstance(data, str):
            raise RuntimeError(f"LenVM tree_value returned string payload: {data}")
        if isinstance(data, dict):
            if "embedding" in data:
                return [data.get("embedding", [])]
            if "data" in data and isinstance(data["data"], list):
                data = data["data"]
            else:
                raise RuntimeError(f"Unexpected LenVM tree_value dict payload: {data!r}")

        values: List[List[float]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    values.append(item.get("embedding", []))
                else:
                    raise RuntimeError(f"Unexpected LenVM tree_value item: {item!r}")
            return values

        raise RuntimeError(f"Unexpected LenVM tree_value payload type: {type(data)!r}")

    def _filter_probs(
        self, probs: torch.Tensor, top_k: int, top_p: float, min_p: float
    ) -> torch.Tensor:
        filtered = probs.clone()
        if top_k != TOP_K_ALL:
            k = min(int(top_k), filtered.numel())
            topk_vals, topk_idx = torch.topk(filtered, k)
            mask = torch.zeros_like(filtered, dtype=torch.bool)
            mask[topk_idx] = True
            filtered[~mask] = 0.0

        if top_p < 1.0:
            probs_sort, probs_idx = torch.sort(filtered, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            probs_sort[(probs_sum - probs_sort) > top_p] = 0.0
            filtered = torch.zeros_like(filtered).scatter_(-1, probs_idx, probs_sort)

        if min_p > 0.0:
            max_prob = filtered.max()
            threshold = max_prob * min_p
            filtered[filtered < threshold] = 0.0

        return filtered

    @staticmethod
    def _req_wants_value_guidance(req: Any) -> bool:
        """Return True iff the request explicitly specifies value-guidance params.

        We gate tree_value calls on this to avoid unnecessary network overhead when
        the user did not request value guidance (default scale/mode).
        """
        sampling_params = getattr(req, "sampling_params", None)
        custom_params = getattr(sampling_params, "custom_params", None)
        if not isinstance(custom_params, dict):
            return False
        keys = (
            "value_scale",
            "scale",
            "value_mode",
            "mode",
            # Hard constraint decoding:
            "target_value",
            "target_length",
            "value_constraint",
            "constraint",
            "cmp",
            "op",
        )
        return any(k in custom_params for k in keys)

    @staticmethod
    def _extract_entropy_threshold(req: Any) -> Optional[float]:
        """Optional entropy threshold to skip /tree_value when distribution is already confident.

        Reads from req.sampling_params.custom_params:
          - value_entropy_threshold
          - entropy_threshold

        Threshold is in *nats* (natural log). If not provided, returns None (disabled).
        """
        # Best-effort per-request cache to avoid dict lookups every token.
        cached = getattr(req, "_lvm_entropy_threshold", None)
        if cached is not None:
            try:
                cached_f = float(cached)
            except (TypeError, ValueError):
                cached_f = None
            if cached_f is None or not math.isfinite(cached_f):
                return None
            return cached_f

        sampling_params = getattr(req, "sampling_params", None)
        custom_params = getattr(sampling_params, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None
        thr = custom_params.get("value_entropy_threshold", custom_params.get("entropy_threshold"))
        if thr is None:
            return None
        try:
            thr_f = float(thr)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(thr_f):
            return None
        # Cache it on the req for subsequent tokens.
        try:
            setattr(req, "_lvm_entropy_threshold", thr_f)
        except Exception:
            pass
        return thr_f

    @staticmethod
    def _get_prefix_ids_incremental(req: Any) -> List[int]:
        """Get prefix_ids = origin_input_ids + output_ids with best-effort incremental caching.

        Building `list(origin)+list(output)` every token is expensive (Python alloc/copy).
        We cache the concatenated list on the req object and only append newly generated
        tokens on subsequent calls.

        Safety:
        - If output_ids shrinks (retract/reset) or cache becomes inconsistent, we rebuild.
        - If origin_input_ids length changes (should not), we rebuild.
        """
        origin = getattr(req, "origin_input_ids", None) or []
        out = getattr(req, "output_ids", None) or []

        # NOTE: In the common case, both are already Python lists (fast path).
        # Avoid converting `out` to a full list on every token; we only need the delta.
        if not isinstance(origin, list):
            origin = list(origin)

        cache = getattr(req, "_lvm_prefix_cache", None)
        if not isinstance(cache, dict):
            prefix = list(origin) + list(out)
            setattr(
                req,
                "_lvm_prefix_cache",
                {"origin_len": len(origin), "out_len": len(out), "prefix": prefix},
            )
            return prefix

        prefix = cache.get("prefix")
        origin_len = int(cache.get("origin_len", -1))
        out_len_cached = int(cache.get("out_len", -1))

        if not isinstance(prefix, list) or origin_len < 0 or out_len_cached < 0:
            prefix = list(origin) + list(out)
            cache.update({"origin_len": len(origin), "out_len": len(out), "prefix": prefix})
            return prefix

        # Rebuild if origin length changed or cache seems inconsistent.
        if len(origin) != origin_len or len(prefix) != origin_len + out_len_cached:
            prefix = list(origin) + list(out)
            cache.update({"origin_len": len(origin), "out_len": len(out), "prefix": prefix})
            return prefix

        # Retract/reset: output_ids got shorter.
        if len(out) < out_len_cached:
            prefix = list(origin) + list(out)
            cache.update({"origin_len": len(origin), "out_len": len(out), "prefix": prefix})
            return prefix

        # Incrementally append newly generated tokens.
        if len(out) > out_len_cached:
            try:
                delta = out[out_len_cached:]
            except Exception:
                # Fallback when slicing isn't supported: convert once and slice.
                delta = list(out)[out_len_cached:]
            # `delta` may be a list/tuple/tensor slice; extend can iterate it.
            prefix.extend(delta)
            cache["out_len"] = len(out)

        return prefix

    def _build_pending(
        self,
        probs: torch.Tensor,
        reqs: Iterable[Any],
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        min_ps: torch.Tensor,
    ) -> Optional["PendingLvmResult"]:
        """Filter candidates and build a PendingLvmResult without contacting the LVM.

        Returns None when there is nothing to do (no rows want guidance).
        """
        device = probs.device
        vocab_size = int(probs.shape[-1])

        # `reqs` is typically already a list, but may be any iterable.
        req_list = reqs if isinstance(reqs, list) else list(reqs)

        # Identify which rows actually want value guidance.
        guided_rows: List[int] = []
        for i, req in enumerate(req_list):
            if self._req_wants_value_guidance(req):
                guided_rows.append(i)

        # If nobody requested value guidance, do nothing and let normal sampling proceed.
        if not guided_rows:
            return None

        guided_rows_t = torch.tensor(guided_rows, device=device, dtype=torch.long)

        # Gather per-row sampling params for the guided subset (single gather, no per-row .item()).
        top_ks_sel = top_ks.index_select(0, guided_rows_t).to(torch.int64)
        top_ps_sel = top_ps.index_select(0, guided_rows_t).to(torch.float32)
        min_ps_sel = min_ps.index_select(0, guided_rows_t).to(torch.float32)

        prefix_ids_send: List[List[int]] = []
        candidate_ids_send: List[List[int]] = []
        send_batch_indices: List[int] = []
        candidate_probs_send: List[List[float]] = []
        deterministic_rows: List[tuple[int, int]] = []

        # Split rare slow-path (top_k == ALL) from the common (top_k is small).
        mask_all = top_ks_sel == TOP_K_ALL
        mask_topk = ~mask_all

        # GPU fast path is active when we are doing expectation-style guidance
        # without any hard target_value/target_length constraints, and all sequences
        # use top-k (not top-k-all), so we can keep tensors on GPU.
        has_hard_target = any(
            _extract_target_value({"req": req_list[ridx]}) is not None
            for ridx in range(len(req_list))
        )
        _use_gpu_path = self._fn in (
            lvm_expectation_guidance,
            lvm_combined_guidance,
        ) and not bool(mask_all.any().item()) and not has_hard_target
        # Will hold (vals_send_gpu, idx_send_gpu) for the top-k send rows, on GPU.
        _gpu_vals_chunks: List[torch.Tensor] = []
        _gpu_idx_chunks: List[torch.Tensor] = []

        # ---------------------------
        # Fast path: batched top-k -> top-p/min-p/entropy on the top-k subset.
        # ---------------------------
        if bool(mask_topk.any().item()):
            rows_topk_t = guided_rows_t[mask_topk]
            top_ks_k = top_ks_sel[mask_topk].clamp(min=1, max=vocab_size)
            top_ps_k = top_ps_sel[mask_topk]
            min_ps_k = min_ps_sel[mask_topk]

            # Single synchronization point to get k_max.
            k_max = int(top_ks_k.max().item())
            if k_max <= 0:
                return None

            probs_k = probs.index_select(0, rows_topk_t).float()
            topk_vals, topk_idx = torch.topk(probs_k, k_max, dim=-1)  # sorted desc

            # Apply per-row top-k (mask out positions >= top_k_i).
            ar = torch.arange(k_max, device=device, dtype=torch.int64).view(1, -1)
            keep_k = ar < top_ks_k.view(-1, 1)
            vals = torch.where(keep_k, topk_vals, torch.zeros_like(topk_vals))

            # Apply per-row top-p within the (masked) top-k list.
            # Keep token j if sum(vals[:j]) <= top_p (equivalently (cum - val) <= top_p).
            # Note: vals are already sorted descending before masking.
            keep_p = torch.ones_like(vals, dtype=torch.bool)
            if torch.any(top_ps_k < 1.0):
                cum = torch.cumsum(vals, dim=-1)
                keep_p = (cum - vals) <= top_ps_k.view(-1, 1)
                vals = torch.where(keep_p, vals, torch.zeros_like(vals))

            # Apply per-row min-p: keep tokens with prob >= max_prob * min_p.
            if torch.any(min_ps_k > 0.0):
                max_prob = vals.max(dim=-1).values
                thresh = max_prob * min_ps_k
                keep_min = vals >= thresh.view(-1, 1)
                vals = torch.where(keep_min, vals, torch.zeros_like(vals))

            # Ensure we always have at least one candidate (defensive, e.g. min_p > 1.0).
            mask_nz = vals > 0
            counts = mask_nz.sum(dim=-1)
            if torch.any(counts == 0):
                zero_rows = counts == 0
                vals = vals.clone()
                vals[zero_rows, 0] = topk_vals[zero_rows, 0]
                mask_nz = vals > 0
                counts = mask_nz.sum(dim=-1)

            # Deterministic rows: exactly 1 candidate.
            det_mask = counts == 1
            det_pos = torch.argmax(vals, dim=-1)
            det_token_ids = torch.gather(
                topk_idx, dim=1, index=det_pos.view(-1, 1)
            ).view(-1)

            # Optional entropy-based skip (per request, Python-sourced thresholds).
            rows_topk_list: List[int] = rows_topk_t.detach().cpu().tolist()
            thr_list: List[float] = []
            has_thr = torch.zeros(len(rows_topk_list), device=device, dtype=torch.bool)
            for j, ridx in enumerate(rows_topk_list):
                thr = self._extract_entropy_threshold(req_list[ridx])
                if thr is None:
                    thr_list.append(float("nan"))
                else:
                    thr_list.append(float(thr))
                    has_thr[j] = True

            # Use float64 for value-guidance gating to avoid precision loss in entropy comparisons.
            thr_t = torch.tensor(thr_list, device=device, dtype=torch.float64)
            skip_entropy = torch.zeros_like(has_thr, dtype=torch.bool)
            if bool(has_thr.any().item()):
                p = vals.to(torch.float64)
                s = p.sum(dim=-1)
                # Avoid division by 0; counts==0 already fixed.
                p = p / s.clamp(min=1e-20).view(-1, 1)
                ent = -(p * torch.log(p + 1e-20)).sum(dim=-1)
                skip_entropy = has_thr & (ent <= thr_t)

            # Rows to send to LVM: non-deterministic and not skipped by entropy.
            send_mask = (~det_mask) & (~skip_entropy)

            # Materialize deterministic rows in Python list.
            if bool(det_mask.any().item()):
                det_token_ids_cpu = det_token_ids.detach().cpu().tolist()
                det_mask_cpu = det_mask.detach().cpu().tolist()
                for j, is_det in enumerate(det_mask_cpu):
                    if is_det:
                        deterministic_rows.append((rows_topk_list[j], int(det_token_ids_cpu[j])))

            # Prepare candidate lists for rows that we will actually send.
            if bool(send_mask.any().item()):
                rows_send_t = rows_topk_t[send_mask]
                # Keep GPU slices before moving to CPU (used by GPU guidance fast path).
                vals_send_gpu = vals[send_mask]  # [B_topk_send, K_max], GPU
                idx_send_gpu = topk_idx[send_mask]  # [B_topk_send, K_max], GPU
                idx_send = idx_send_gpu.detach().cpu()
                # GPU fast path: only need bool mask (4x smaller than float32 transfer).
                # CPU path: need full float values for candidate_probs_send.
                if _use_gpu_path:
                    valid_mask_send = (vals_send_gpu > 0).detach().cpu()
                else:
                    vals_send = vals_send_gpu.detach().cpu()

                rows_send_list = rows_send_t.detach().cpu().tolist()
                for j, ridx in enumerate(rows_send_list):
                    # In practice (sorted desc + thresholding), non-zeros are a prefix. Still, use mask for safety.
                    if _use_gpu_path:
                        m = valid_mask_send[j]
                    else:
                        m = vals_send[j] > 0
                    cand_ids = idx_send[j][m].tolist()
                    if len(cand_ids) <= 1:
                        # Should have been caught by det_mask, but keep a safe fallback.
                        if len(cand_ids) == 1:
                            deterministic_rows.append((ridx, int(cand_ids[0])))
                        continue

                    prefix = self._get_prefix_ids_incremental(req_list[ridx])
                    prefix_ids_send.append(prefix)
                    candidate_ids_send.append(cand_ids)
                    if not _use_gpu_path:
                        candidate_probs_send.append(vals_send[j][m].tolist())
                    send_batch_indices.append(ridx)
                    if _use_gpu_path:
                        # Capture GPU row j for later gpu_candidates assembly.
                        _gpu_vals_chunks.append(vals_send_gpu[j].unsqueeze(0))
                        _gpu_idx_chunks.append(idx_send_gpu[j].unsqueeze(0))

        # ---------------------------
        # Slow path: top_k == ALL (full vocab filtering). Rare; keep correctness-oriented CPU behavior.
        # ---------------------------
        if bool(mask_all.any().item()):
            rows_all_list = guided_rows_t[mask_all].detach().cpu().tolist()
            top_ps_all = top_ps_sel[mask_all].detach().cpu().tolist()
            min_ps_all = min_ps_sel[mask_all].detach().cpu().tolist()
            for j, i in enumerate(rows_all_list):
                row = probs[i]
                top_p_i = float(top_ps_all[j])
                min_p_i = float(min_ps_all[j])

                row_cpu = row.float().cpu()
                filtered = self._filter_probs(row_cpu, TOP_K_ALL, top_p_i, min_p_i)
                cand_idx = filtered.nonzero(as_tuple=True)[0].tolist()
                if not cand_idx:
                    cand_idx = [int(torch.argmax(row_cpu).item())]
                    filtered = torch.zeros_like(filtered)
                    filtered[cand_idx[0]] = row_cpu[cand_idx[0]]
                cand_probs = filtered[cand_idx].tolist()

                if len(cand_idx) == 1:
                    deterministic_rows.append((i, int(cand_idx[0])))
                    continue

                thr = self._extract_entropy_threshold(req_list[i])
                if thr is not None:
                    # float64 for stable entropy computation/comparison
                    p = torch.tensor(cand_probs, dtype=torch.float64)
                    s = float(p.sum().item())
                    if s > 0:
                        p = p / s
                        ent = float(-(p * torch.log(p + 1e-20)).sum().item())
                    else:
                        ent = 0.0
                    if ent <= float(thr):
                        continue

                prefix = self._get_prefix_ids_incremental(req_list[i])
                prefix_ids_send.append(prefix)
                candidate_ids_send.append(cand_idx)
                candidate_probs_send.append(cand_probs)
                send_batch_indices.append(i)

        if not send_batch_indices and not deterministic_rows:
            return None

        # Build guided tensor and fill deterministic rows immediately.
        guided = probs.clone()

        # Fill deterministic rows (single candidate) without contacting LVM.
        for i, tok in deterministic_rows:
            guided[i].zero_()
            guided[i, tok] = 1.0

        # Assemble GPU candidate tensors for the fast guidance path.
        gpu_candidates = None
        if _use_gpu_path and _gpu_vals_chunks:
            gp = torch.cat(_gpu_vals_chunks, dim=0).float()  # [B_send, K_max]
            gi = torch.cat(_gpu_idx_chunks, dim=0)  # [B_send, K_max]
            gm = gp > 0  # [B_send, K_max] bool
            gpu_candidates = (gp, gi, gm)

        return PendingLvmResult(
            req_list=req_list,
            device=device,
            guided=guided,
            send_batch_indices=send_batch_indices,
            prefix_ids_send=prefix_ids_send,
            candidate_ids_send=candidate_ids_send,
            candidate_probs_send=candidate_probs_send,
            gpu_candidates=gpu_candidates,
        )

    def _apply_guidance_gpu(self, pending: "PendingLvmResult", gpu_embeddings) -> None:
        """GPU-native guidance path for lvm_expectation_guidance.

        Runs sigmoid + batched Newton's method entirely on GPU.  No CPU↔GPU
        data movement after this point — results are scattered directly into
        pending.guided.

        gpu_embeddings: GPU tensor(s) of raw LVM scalar outputs.
          - If a list of 1D tensors (variable-length per sequence): padded internally.
          - If a 2D tensor [B_send, K_max]: used directly.
        pending.gpu_candidates: (padded_probs [B,K], padded_ids [B,K], valid_mask [B,K])
        """
        padded_probs, padded_ids, valid_mask = pending.gpu_candidates
        B, K_max = padded_probs.shape
        device = padded_probs.device

        # -- Normalize gpu_embeddings to [B, K_max] float32 on device ---------------
        if isinstance(gpu_embeddings, list):
            # Variable-length per sequence: pad with zeros to K_max.
            emb = torch.zeros(B, K_max, device=device, dtype=torch.float32)
            for bi, t in enumerate(gpu_embeddings):
                n = min(int(t.shape[0]), K_max)
                emb[bi, :n] = t[:n].float()
        else:
            emb = gpu_embeddings.float()
            if emb.shape != (B, K_max):
                # Adjust dimensions if the model returns a different shape.
                if emb.dim() == 1:
                    emb = emb.view(B, -1)
                if emb.shape[1] > K_max:
                    emb = emb[:, :K_max]
                elif emb.shape[1] < K_max:
                    pad = torch.zeros(
                        B, K_max - emb.shape[1], device=device, dtype=torch.float32
                    )
                    emb = torch.cat([emb, pad], dim=1)

        # -- Per-sequence scale/mode (O(B) dict lookups, fast) ----------------------
        scales_list: List[float] = []
        modes_list: List[str] = []
        for ridx in pending.send_batch_indices:
            req = pending.req_list[ridx]
            scales_list.append(_extract_value_scale({"req": req}))
            modes_list.append(_extract_value_mode({"req": req}, default="mul"))

        # Use the most common mode; fall back to "mul" if mixed (rare).
        mode = modes_list[0] if len(set(modes_list)) == 1 else "mul"
        scale_t = torch.tensor(scales_list, device=device, dtype=torch.float32)  # [B]

        # -- Sigmoid of raw embeddings → values in [0, 1] ---------------------------
        values = torch.sigmoid(emb)  # [B, K_max]
        values = values.masked_fill(~valid_mask, 0.0)

        for bi, req_idx in enumerate(pending.send_batch_indices):
            eos_ids = get_eos_token_ids(pending.req_list[req_idx])
            if not eos_ids:
                continue
            eos_ids_t = torch.tensor(list(eos_ids), device=device, dtype=padded_ids.dtype)
            eos_mask = torch.isin(padded_ids[bi], eos_ids_t) & valid_mask[bi]
            if torch.any(eos_mask):
                values[bi] = values[bi].masked_fill(eos_mask, 0.0)

        # -- Normalised probs under filtered distribution ----------------------------
        p = padded_probs.float()  # [B, K_max]
        p_sum = p.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        p_norm = p / p_sum  # [B, K_max]

        # -- Compute target E[v] for exponential tilting ----------------------------
        cur_exp = (p_norm * values).sum(dim=-1)  # [B]
        min_v = values.masked_fill(~valid_mask, 1e9).min(dim=-1).values  # [B]
        max_v = values.masked_fill(~valid_mask, -1e9).max(dim=-1).values  # [B]
        v_range = (max_v - min_v).clamp(min=1e-8)  # [B]

        if mode == "centered_exp":
            # print("centered_exp")
            # p'(i) ∝ p(i) * exp(sigmoid(emb(i)) * scale); values = sigmoid(emb) ∈ [0, 1].
            # Subtract per-row max before exp to prevent overflow (cancels in normalization).
            logits = values * scale_t.unsqueeze(1)
            logits = logits.masked_fill(~valid_mask, -1e9)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            w = p_norm * torch.exp(logits)
            w = w.masked_fill(~valid_mask, 0.0)
            s = w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            final_probs = w / s  # [B, K_max]
            # print("p_norm, values, final_probs: ", p_norm.tolist(), values.tolist(), final_probs.tolist(),flush=True)
        elif mode == "value_bias":
            # print("value_bias")
            # p'(i) ∝ p(i) * exp(emb(i) * scale); emb is the raw LVM logit (unbounded).
            # Subtract per-row max before exp to prevent overflow (cancels in normalization).
            logits = emb * scale_t.unsqueeze(1)
            logits = logits.masked_fill(~valid_mask, -1e9)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            w = p_norm * torch.exp(logits)
            w = w.masked_fill(~valid_mask, 0.0)
            s = w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            final_probs = w / s  # [B, K_max]
        elif mode in ("exp", "linear", "length_mul", "mul"):
            logger.debug("LenVM GPU fast path uses mode=%s", mode)
            # Compute target E[v] then solve for lambda via Newton's method.
            if mode == "exp":
                cur_norm_t = ((cur_exp - min_v) / v_range).clamp(0.0, 1.0)
                target_norm_t = 1.0 - (1.0 - cur_norm_t) ** scale_t
                target = min_v + target_norm_t * v_range
            elif mode == "linear":
                above = scale_t >= 1.0
                target = torch.where(
                    above,
                    cur_exp + (scale_t - 1.0) * (max_v - cur_exp),
                    cur_exp - (1.0 - scale_t) * (cur_exp - min_v),
                )
            elif mode == "length_mul":
                gamma = 0.997  # default; per-seq gamma not available in GPU path
                log_gamma = math.log(gamma)
                mu_v = cur_exp.clamp(min=1e-15, max=1.0 - 1e-15)
                l_cur = torch.log1p(-mu_v) / log_gamma
                l_tgt = scale_t * l_cur
                target = 1.0 - torch.exp(l_tgt * log_gamma)
                target = target.clamp(min_v + 1e-12, max_v - 1e-12)
            else:  # "mul"
                target = cur_exp * scale_t

            eps_v = (v_range * 1e-6).clamp(min=1e-12)
            target = torch.max(torch.min(target, max_v - eps_v), min_v + eps_v)  # [B]

            # Skip sequences where target ≈ cur_exp (no tilting needed).
            needs_tilt = (v_range > 1e-8) & (torch.abs(target - cur_exp) > 1e-8)

            # -- Batched Newton's method for exponential tilting ------------------------
            log_p = torch.log(p_norm.clamp(min=1e-20))
            log_p = log_p.masked_fill(~valid_mask, -1e9)
            v_ref = min_v.unsqueeze(1)  # [B, 1]
            dv = (values - v_ref).masked_fill(~valid_mask, 0.0)  # [B, K_max]

            lam = torch.zeros(B, 1, device=device, dtype=torch.float32)
            for _ in range(20):
                logits = log_p + dv * lam  # [B, K_max]
                m = logits.max(dim=-1, keepdim=True).values
                w = torch.exp(logits - m).masked_fill(~valid_mask, 0.0)
                s = w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
                w = w / s
                mean = (w * values).sum(dim=-1, keepdim=True)  # [B, 1]
                mean2 = (w * values * values).sum(dim=-1, keepdim=True)
                var = (mean2 - mean * mean).clamp(min=0.0)
                err = mean - target.unsqueeze(1)
                step = err / var.clamp(min=1e-16)
                lam = (lam - step).clamp(-100.0, 100.0)

            # -- Compute final tilted probs and scatter into guided ----------------------
            final_logits = log_p + dv * lam
            final_logits = final_logits.masked_fill(~valid_mask, -1e9)
            m = final_logits.max(dim=-1, keepdim=True).values
            w = torch.exp(final_logits - m).masked_fill(~valid_mask, 0.0)
            s = w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            final_probs = w / s  # [B, K_max]

            # For sequences that don't need tilting, fall back to normalized p_norm.
            final_probs = torch.where(needs_tilt.unsqueeze(1), final_probs, p_norm)
        else:
            raise ValueError(f"[LVM GPU path] Unknown value_mode: {mode!r}")

        guided = pending.guided
        rows_t = torch.tensor(pending.send_batch_indices, device=device, dtype=torch.long)
        # Zero target rows in one kernel (avoids bs individual .zero_() calls).
        guided.index_fill_(0, rows_t, 0.0)
        # Batched scatter: no per-row boolean indexing → eliminates all GPU-CPU syncs.
        # padded_ids[invalid] == 0 and final_probs_typed[invalid] == 0 → safe no-op writes.
        row_indices = rows_t.unsqueeze(1).expand(-1, K_max)  # [B, K_max]
        final_probs_typed = final_probs.masked_fill(~valid_mask, 0.0).to(guided.dtype)
        guided[row_indices, padded_ids] = final_probs_typed

    def _apply_guidance(
        self, pending: "PendingLvmResult", lvm_values: List[List[float]]
    ) -> None:
        """Apply the guidance function and scatter results into pending.guided (in-place)."""
        guided = pending.guided
        device = pending.device
        for k, i in enumerate(pending.send_batch_indices):
            req = pending.req_list[i]
            raw_values = torch.tensor(lvm_values[k], dtype=torch.float64)
            token_values = torch.sigmoid(raw_values).tolist()
            token_probs = pending.candidate_probs_send[k]
            token_ids = pending.candidate_ids_send[k]

            force_eos_value_zero(token_ids, token_values, req)

            try:
                new_probs = self._fn(
                    token_probs=token_probs,
                    token_values=token_values,
                    token_ids=token_ids,
                    req=req,
                )
            except Exception as exc:
                raise RuntimeError("LenVM guidance function failed") from exc

            if not isinstance(new_probs, list) or len(new_probs) != len(token_probs):
                raise RuntimeError(
                    "LenVM guidance function returned an invalid probability vector"
                )

            new_probs_tensor = torch.tensor(
                new_probs, dtype=guided.dtype, device=device
            )
            new_probs_tensor = torch.clamp(new_probs_tensor, min=0)
            prob_sum = new_probs_tensor.sum()
            if prob_sum <= 0:
                raise RuntimeError("LenVM guidance function produced a non-positive probability sum")
            new_probs_tensor = new_probs_tensor / prob_sum

            guided[i].zero_()
            guided[i, torch.tensor(token_ids, device=device)] = new_probs_tensor

    def apply(
        self,
        probs: torch.Tensor,
        reqs: Iterable[Any],
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        min_ps: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Filter candidates, run LVM forward, and apply guidance in one call.

        Returns the modified probs tensor, or None when no guidance is needed
        (caller should use the original probs).
        """
        inproc = self._get_inproc_provider()
        if inproc not in (None, False):
            # Free KV cache for requests that have finished or aborted
            inproc.clean_stale_requests(set(r.rid for r in reqs))

        pending = self._build_pending(probs, reqs, temperatures, top_ps, top_ks, min_ps)
        if pending is None:
            return None

        if pending.send_batch_indices:
            reqs_send = [pending.req_list[i] for i in pending.send_batch_indices]
            if pending.gpu_candidates is not None:
                # GPU fast path (synchronous).
                if inproc not in (None, False):
                    try:
                        rids_send = [req.rid for req in reqs_send]
                        inproc.tree_value_extend(rids_send, pending.prefix_ids_send, reqs_send)
                        gpu_emb = inproc.tree_value_launch_gpu(
                            rids_send, pending.candidate_ids_send, gpu_candidates=pending.gpu_candidates
                        )
                        gpu_embeddings = inproc.tree_value_collect_gpu(gpu_emb)
                        self._apply_guidance_gpu(pending, gpu_embeddings)
                        return pending.guided
                    except Exception as exc:
                        raise RuntimeError("LenVM GPU guidance path failed") from exc

            lvm_values = self._post_tree_value(
                [req.rid for req in reqs_send], pending.prefix_ids_send, pending.candidate_ids_send, reqs_send
            )
            if lvm_values is None:
                return None
            self._apply_guidance(pending, lvm_values)

        return pending.guided
