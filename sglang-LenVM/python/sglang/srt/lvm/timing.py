"""Lightweight per-step timing for LenVM-guided decoding.

Set SGLANG_LVM_TIMING_LOG=/path/to/timing.jsonl to enable. The timer writes one
JSONL record per sampler step and also emits an aggregate summary next to it at
process exit. When unset, all methods are no-ops.
"""

from __future__ import annotations

import atexit
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
        self._seen_steps = 0
        self._lock = threading.Lock()
        self._current_step: dict[str, object] = {}
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._meta_counts: dict[str, int] = {}
        try:
            self._summary_interval = max(
                int(os.environ.get("SGLANG_LVM_TIMING_SUMMARY_INTERVAL", "500")), 1
            )
        except ValueError:
            self._summary_interval = 500
        try:
            self._skip_steps = max(
                int(os.environ.get("SGLANG_LVM_TIMING_SKIP_STEPS", "0")), 0
            )
        except ValueError:
            self._skip_steps = 0
        if self.enabled:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            self._fh = open(self._log_path, "a", buffering=1)
            atexit.register(self.close)

    def section_start(self, name: str) -> Optional[float]:
        if not self.enabled:
            return None
        return time.perf_counter()

    def section_end(self, name: str, start: Optional[float]) -> None:
        if not self.enabled or start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        prev = self._current_step.get(name, 0.0)
        if isinstance(prev, (int, float)):
            self._current_step[name] = float(prev) + elapsed_ms
        else:
            self._current_step[name] = elapsed_ms

    def set_meta(self, **kwargs) -> None:
        if not self.enabled:
            return
        self._current_step.update(kwargs)

    def flush_step(self) -> None:
        if not self.enabled or self._fh is None or not self._current_step:
            return
        with self._lock:
            self._seen_steps += 1
            if self._seen_steps <= self._skip_steps:
                self._current_step.clear()
                return
            self._step_id += 1
            record = {"step": self._step_id, **self._current_step}
            self._fh.write(json.dumps(record) + "\n")
            self._update_summary_from_record_locked(record)
            self._current_step.clear()
            if self._step_id % self._summary_interval == 0:
                self._write_summary_locked()

    def _update_summary_from_record_locked(self, record: dict[str, object]) -> None:
        for key, value in record.items():
            if key == "step":
                continue
            if isinstance(value, bool):
                self._meta_counts[key] = self._meta_counts.get(key, 0) + int(value)
            elif isinstance(value, (int, float)):
                self._totals[key] = self._totals.get(key, 0.0) + float(value)
                self._counts[key] = self._counts.get(key, 0) + 1

    def _write_summary_locked(self) -> None:
        if not self._log_path:
            return
        timing_keys = sorted(key for key in self._totals if key.endswith("_ms"))
        numeric_keys = sorted(key for key in self._totals if not key.endswith("_ms"))
        summary = {
            "steps": self._step_id,
            "means_ms": {
                key: self._totals[key] / max(self._counts.get(key, 1), 1)
                for key in timing_keys
            },
            "means": {
                key: self._totals[key] / max(self._counts.get(key, 1), 1)
                for key in numeric_keys
            },
            "totals": {key: self._totals[key] for key in sorted(self._totals)},
            "counts": dict(sorted(self._meta_counts.items())),
        }
        summary_path = f"{self._log_path}.summary.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._current_step:
                self._seen_steps += 1
                if self._seen_steps > self._skip_steps:
                    self._step_id += 1
                    record = {"step": self._step_id, **self._current_step}
                    if self._fh is not None:
                        self._fh.write(json.dumps(record) + "\n")
                    self._update_summary_from_record_locked(record)
                self._current_step.clear()

            self._write_summary_locked()

            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self.enabled = False


_INSTANCE: Optional[_Timer] = None


def get_timer() -> _Timer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _Timer()
    return _INSTANCE
