"""Lightweight per-step Python wall-clock timer for LenVM-guided decoding.

Activated by environment variable SGLANG_LVM_TIMING_LOG=/path/to/timing.jsonl.
When unset, all timer operations are no-ops with negligible overhead.

Captures wall-clock (time.perf_counter) for sections within Sampler.forward and
LvmGuidedSampler.apply. Each scheduler step flushes one JSONL record with the
section durations and metadata. Records are flushed line-by-line so a tail-f
during a run is safe.

The collector lives in the GPU worker process. For DP/TP>1 it would need a
per-rank suffix on the log path; current scope is single-rank smoke testing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional


class _Timer:
    def __init__(self) -> None:
        log_path = os.environ.get("SGLANG_LVM_TIMING_LOG")
        self.enabled = bool(log_path)
        self._log_path: Optional[str] = log_path
        self._fh = None
        self._step_id = 0
        self._lock = threading.Lock()
        self._current_step: dict = {}
        if self.enabled:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            self._fh = open(self._log_path, "a", buffering=1)

    def section_start(self, name: str) -> Optional[float]:
        if not self.enabled:
            return None
        return time.perf_counter()

    def section_end(self, name: str, start: Optional[float]) -> None:
        if not self.enabled or start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Accumulate in case a section is entered multiple times per step.
        prev = self._current_step.get(name, 0.0)
        self._current_step[name] = prev + elapsed_ms

    def set_meta(self, **kwargs) -> None:
        if not self.enabled:
            return
        self._current_step.update(kwargs)

    def flush_step(self) -> None:
        """Write one JSONL line for the current step and reset accumulator."""
        if not self.enabled or self._fh is None:
            return
        if not self._current_step:
            return
        with self._lock:
            self._step_id += 1
            record = {"step": self._step_id, **self._current_step}
            self._fh.write(json.dumps(record) + "\n")
            self._current_step.clear()


_INSTANCE: Optional[_Timer] = None


def get_timer() -> _Timer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _Timer()
    return _INSTANCE
