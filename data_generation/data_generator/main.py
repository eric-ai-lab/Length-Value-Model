import argparse
import asyncio
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

import datasets

from .config import GenerationConfig
from .prompt_builder import PromptBuilder
from .generator import DataGenerator
from .processor import process_dataset
from .utils import load_existing_data, group_shuffle_jsonl_by_index

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0x2A700 <= cp <= 0x2B73F  # CJK Extension C
        or 0x2B740 <= cp <= 0x2B81F  # CJK Extension D
        or 0x2B820 <= cp <= 0x2CEAF  # CJK Extension E/F
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
    )


def _is_latin_letter(ch: str) -> bool:
    cp = ord(ch)
    # Basic Latin + Latin-1 Supplement + Latin Extended blocks.
    return (
        0x0041 <= cp <= 0x005A
        or 0x0061 <= cp <= 0x007A
        or 0x00C0 <= cp <= 0x00FF
        or 0x0100 <= cp <= 0x017F
        or 0x0180 <= cp <= 0x024F
        or 0x1E00 <= cp <= 0x1EFF
    )


def _wildchat_user_text(sample: dict) -> str:
    conv = sample.get("conversation")
    if isinstance(conv, list) and conv:
        first = conv[0]
        if isinstance(first, dict):
            role = str(first.get("role", "")).lower()
            if role == "user":
                return str(first.get("content", ""))
    return ""


def _is_math_greek_letter(ch: str) -> bool:
    # Strategy B: allow only a common math subset, not all Greek letters.
    # Lowercase + uppercase frequently seen in formulas.
    allowed = {
        "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
        "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
        # Common variant glyphs used in math text.
        "ϵ", "ϑ", "ϕ", "ϖ", "ς",
    }
    return ch in allowed


def _contains_only_zh_en_letters(text: str) -> bool:
    for ch in str(text):
        if ch.isalpha():
            if _is_cjk(ch) or _is_latin_letter(ch) or _is_math_greek_letter(ch):
                continue
            return False
        # Allow non-letters (digits, punctuation, symbols, whitespace, code tokens).
        _ = unicodedata.category(ch)
    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAPO Math Data Generation")

    # Dataset args
    parser.add_argument("--dataset-name", default="zwhe99/DeepMath-103K")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--load-from-local", action="store_true")
    parser.add_argument("--local-dataset-path", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=-1)
    parser.add_argument("--train-samples-per-question", type=int, default=3)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--test-samples-per-question", type=int, default=3)
    parser.add_argument("--train-raw-path", default=None)
    parser.add_argument("--test-raw-path", default=None)
    parser.add_argument("--train-group-path", default=None)
    parser.add_argument("--test-group-path", default=None)
    parser.add_argument("--train-indices-path", default=None)
    parser.add_argument("--test-indices-path", default=None)
    parser.add_argument("--keep-columns", default="")
    parser.add_argument("--image-dir", default=None, help="Directory to save images for multimodal datasets")
    parser.add_argument("--save-batch-size", type=int, default=10)
    parser.add_argument(
        "--reorder-only",
        action="store_true",
        help="Only reorder an existing jsonl specified by --raw-output-path (no dataset loading / generation).",
    )
    parser.add_argument(
        "--group-by-index",
        action="store_true",
        help='After generation, reorder jsonl so same meta_info.lenvm_idx are contiguous; '
             "shuffle group order + shuffle within each group.",
    )

    # Model/API args
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:10001/v1")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--max-concurrency", type=int, default=1000)
    parser.add_argument("--max-connections", type=int, default=100)
    parser.add_argument("--max-keepalive-connections", type=int, default=20)

    # Reliability args
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-delay", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)

    return parser.parse_args()

def parse_keep_columns(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]

def resolve_split_output_path(base_path: Path, split_name: str) -> Path:
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")
    return base_path / f"{split_name}.jsonl"

def load_dataset_source(args: argparse.Namespace) -> datasets.Dataset:
    if args.load_from_local:
        if not args.local_dataset_path:
            raise ValueError("local_dataset_path is required when load_from_local is set.")
        loaded = datasets.load_from_disk(args.local_dataset_path)
        if isinstance(loaded, datasets.DatasetDict):
            if args.dataset_split not in loaded:
                raise ValueError(f"Split '{args.dataset_split}' not found in local dataset.")
            return loaded[args.dataset_split]
        return loaded

    return datasets.load_dataset(args.dataset_name, split=args.dataset_split)

def compute_split_sizes(
    total: int,
    train_samples: int,
    test_samples: int,
) -> Tuple[int, int]:
    if train_samples < 0 and test_samples < 0:
        return total, 0
    if train_samples < 0:
        return max(0, total - max(test_samples, 0)), max(test_samples, 0)
    if test_samples < 0:
        return max(train_samples, 0), max(0, total - max(train_samples, 0))
    return max(train_samples, 0), max(test_samples, 0)

def main():
    args = parse_args()
    logger.warning("Generation args: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    keep_columns = parse_keep_columns(args.keep_columns)

    # 1. Setup paths
    default_base = Path(__file__).resolve().parent.parent / "data" / "dapo_math_17k"
    train_raw_path = Path(args.train_raw_path) if args.train_raw_path else resolve_split_output_path(default_base, "train")
    test_raw_path = Path(args.test_raw_path) if args.test_raw_path else resolve_split_output_path(default_base, "test")
    train_group_path = Path(args.train_group_path) if args.train_group_path else resolve_split_output_path(default_base, "train_group")
    test_group_path = Path(args.test_group_path) if args.test_group_path else resolve_split_output_path(default_base, "test_group")

    if args.train_indices_path:
        train_indices_path = Path(args.train_indices_path)
    else:
        train_indices_path = train_raw_path.with_name(f"{train_raw_path.stem}_indices.json")
    if args.test_indices_path:
        test_indices_path = Path(args.test_indices_path)
    else:
        test_indices_path = test_raw_path.with_name(f"{test_raw_path.stem}_indices.json")

    # Optional: only reorder an existing jsonl and exit.
    if args.reorder_only:
        for path_in, path_out in [(train_raw_path, train_group_path), (test_raw_path, test_group_path)]:
            if not path_in.exists():
                continue
            group_shuffle_jsonl_by_index(
                input_path=path_in,
                seed=args.random_seed,
                output_path=path_out,
                inplace=(path_out == path_in),
            )
            logger.warning(f"Grouped/shuffled jsonl saved to: {path_out}")
        return

    # 2. Load and Preprocess Dataset
    try:
        ds = load_dataset_source(args)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Filter WildChat to English before shuffling/splitting.
    if args.dataset_name == "allenai/WildChat":
        if "language" not in ds.column_names:
            raise ValueError("WildChat dataset missing 'language' column.")
        # Combined filter: language must be EN/ZH and user prompt must not contain other scripts (e.g. Cyrillic).
        ds = ds.filter(
            lambda row: (row["language"] == "English" or row["language"] == "Chinese")
            and _contains_only_zh_en_letters(_wildchat_user_text(row))
        )

    # Add 'lenvm_idx' if missing
    assert "lenvm_idx" not in ds.column_names, "lenvm_idx is found in dataset"
    ds = ds.add_column("lenvm_idx", list(range(len(ds))))

    # Shuffle and split samples
    ds = ds.shuffle(seed=args.random_seed)
    train_size, test_size = compute_split_sizes(len(ds), args.train_samples, args.test_samples)
    if train_size + test_size > len(ds):
        raise ValueError(
            f"Requested train+test samples ({train_size + test_size}) exceed dataset size ({len(ds)})."
        )

    total_len = len(ds)
    if test_size > 0:
        test_start = max(0, total_len - test_size)
        test_indices = list(range(test_start, total_len))
        test_ds = ds.select(test_indices)
    else:
        test_ds = None

    if train_size > 0:
        max_train = total_len - (len(test_ds) if test_ds is not None else 0)
        train_count = min(train_size, max_train)
        train_ds = ds.select(range(train_count))
    else:
        train_ds = None

    # 3. Save Indices
    train_indices_path.parent.mkdir(parents=True, exist_ok=True)
    test_indices_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if train_ds is not None:
            with train_indices_path.open("w") as f:
                json.dump(list(train_ds["lenvm_idx"]), f)
            logger.info(f"Saved train indices to {train_indices_path}")
        if test_ds is not None:
            with test_indices_path.open("w") as f:
                json.dump(list(test_ds["lenvm_idx"]), f)
            logger.info(f"Saved test indices to {test_indices_path}")
    except Exception as e:
        logger.error(f"Failed to save indices: {e}")

    # 4. Initialize Generation
    train_existing_counts = load_existing_data(train_raw_path)
    test_existing_counts = load_existing_data(test_raw_path)

    config = GenerationConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        max_retries=args.max_retries,
        retry_initial_delay=args.retry_initial_delay,
        retry_max_delay=args.retry_max_delay,
        request_timeout=args.request_timeout,
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
        keep_columns=keep_columns,
        image_dir=args.image_dir,
    )

    # 5. Run: use a single event loop for processing and cleanup
    try:
        async def run_generation():
            if train_ds is not None and args.train_samples_per_question > 0:
                train_prompt_builder = PromptBuilder(args.dataset_name, args.dataset_split)
                train_generator = DataGenerator(config, args.max_concurrency, train_prompt_builder)
                await process_dataset(
                    train_generator,
                    train_ds,
                    args.train_samples_per_question,
                    train_existing_counts,
                    train_raw_path,
                    "train",
                    args.save_batch_size,
                )
                await train_generator.close()
            if test_ds is not None and args.test_samples_per_question > 0:
                test_prompt_builder = PromptBuilder(args.dataset_name, args.dataset_split)
                test_generator = DataGenerator(config, args.max_concurrency, test_prompt_builder)
                await process_dataset(
                    test_generator,
                    test_ds,
                    args.test_samples_per_question,
                    test_existing_counts,
                    test_raw_path,
                    "test",
                    args.save_batch_size,
                )
                await test_generator.close()

        asyncio.run(run_generation())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        return

    for path_in, path_out in [(train_raw_path, train_group_path), (test_raw_path, test_group_path)]:
        if not path_in.exists():
            continue
        try:
            group_shuffle_jsonl_by_index(
                input_path=path_in,
                seed=args.random_seed,
                output_path=path_out,
                inplace=(path_out == path_in),
            )
            logger.warning(f"Grouped/shuffled jsonl saved to: {path_out}")
        except Exception as e:
            logger.error(f"Failed to group/shuffle jsonl: {e}")

if __name__ == "__main__":
    main()
