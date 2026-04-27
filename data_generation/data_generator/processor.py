import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm.asyncio import tqdm_asyncio

from .generator import DataGenerator
from .utils import save_batch

logger = logging.getLogger(__name__)

async def process_dataset(
    generator: DataGenerator,
    samples: List[Dict[str, Any]],
    samples_per_question: int,
    existing_counts: Dict[str, int],
    save_path: Path,
    split_name: str,
    save_batch_size: int = 10,
):
    """Main processing loop: schedules tasks and saves results."""
    # Use existing_counts (already loaded) to compute accurate total without scanning samples
    already_done = sum(min(v, samples_per_question) for v in existing_counts.values())
    total_needed = max(0, len(samples) * samples_per_question - already_done)

    def task_generator():
        for sample in samples:
            idx = str(sample.get("lenvm_idx"))
            if not idx:
                continue

            current_count = existing_counts.get(idx, 0)
            needed = max(0, samples_per_question - current_count)

            if needed > 0:
                for _ in range(needed):
                    yield generator.process_sample(sample, split_name)
                    existing_counts[idx] = existing_counts.get(idx, 0) + 1

    batch = []

    pbar = tqdm_asyncio(
        total=total_needed,
        initial=0,
        desc="Generating",
        dynamic_ncols=True,
        mininterval=0.2,
        maxinterval=2.0,
        smoothing=0.2,
    )

    pending = set()
    task_gen = task_generator()

    # Saturate concurrency from the start instead of ramping up from 1
    max_buffer = max(1, generator.max_concurrency * 2)
    for _ in range(min(max_buffer, generator.max_concurrency)):
        try:
            pending.add(asyncio.create_task(next(task_gen)))
        except StopIteration:
            break

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = await task
            pbar.update(1)
            if result:
                batch.append(result)
                if save_batch_size > 0 and len(batch) >= save_batch_size:
                    await save_batch(save_path, batch)
                    batch = []

            refill_limit = min(100, max_buffer - len(pending))
            for _ in range(refill_limit):
                try:
                    pending.add(asyncio.create_task(next(task_gen)))
                except StopIteration:
                    break

    pbar.close()

    if batch and save_batch_size > 0:
        await save_batch(save_path, batch)
