import asyncio
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 423, 425, 429, 499}

def calculate_backoff_delay(attempt: int, initial_delay: float, max_delay: float) -> float:
    """Calculates exponential backoff delay with jitter."""
    if attempt <= 0:
        return 0.0
    delay = initial_delay * (2 ** (attempt - 1))
    if max_delay:
        delay = min(delay, max_delay)
    return delay * random.uniform(0.8, 1.2)

def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get value from object or dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def load_existing_data(path: Path) -> Dict[str, int]:
    """Scans existing output file to count processed samples by lenvm_idx."""
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Extract index from deep structure
                    meta = safe_get(entry, "meta_info")
                    idx = safe_get(meta, "lenvm_idx", safe_get(meta, "index"))
                    
                    if idx is not None:
                        key = str(idx)
                        counts[key] = counts.get(key, 0) + 1
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning(f"Error reading existing data from {path}: {exc}")

    return counts

async def save_batch(path: Path, batch: List[Dict[str, Any]]):
    """Appends a batch of results to the output file."""
    if not batch:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Offload file I/O to a thread so it doesn't block the async event loop
        def _write_sync():
            with path.open("a", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_sync)
    except Exception as exc:
        logger.error(f"Failed to save batch to {path}: {exc}")

def group_shuffle_jsonl_by_index(
    input_path: Path,
    seed: int,
    output_path: Optional[Path] = None,
    inplace: bool = True,
) -> Path:
    """
    Reorder a jsonl file so that samples with the same meta_info.lenvm_idx are adjacent,
    while keeping randomness:
    - shuffle the order of index groups
    - shuffle items within each index group
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input jsonl not found: {input_path}")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    passthrough_lines: List[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Keep malformed lines (rare) at the end rather than dropping.
                passthrough_lines.append(raw)
                continue

            meta = safe_get(obj, "meta_info")
            idx = safe_get(meta, "lenvm_idx", safe_get(meta, "index"))
            if idx is None:
                passthrough_lines.append(raw)
                continue
            groups[str(idx)].append(obj)

    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])

    if output_path is None:
        output_path = input_path

    if inplace and output_path == input_path:
        tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        write_path = tmp_path
    else:
        write_path = output_path

    write_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("w", encoding="utf-8") as wf:
        for k in keys:
            for obj in groups[k]:
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for raw in passthrough_lines:
            wf.write(raw + "\n")

    if inplace and output_path == input_path:
        write_path.replace(input_path)
        return input_path
    return write_path
