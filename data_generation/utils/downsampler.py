import argparse
import json
import os
import random
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a block-structured dataset of size (num_groups * group_size) to (m * n). "
                    "Assumes entries are ordered in contiguous blocks of length group_size."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSON/JSONL file. JSON: a list of records. JSONL: one JSON record per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output JSON/JSONL file. Format is inferred from output extension (.jsonl => JSONL, else JSON).",
    )
    parser.add_argument(
        "-m",
        "--num-questions",
        type=int,
        required=True,
        help="Number of questions (groups) to keep. Use a negative value (e.g. -1) to keep all questions.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples per question to keep. Must be <= group_size.",
    )
    parser.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=64,
        help="Number of records per group (default: 64).",
    )
    parser.add_argument(
        "--group-mode",
        choices=["head", "random"],
        default="head",
        help="How to choose which groups to keep: 'head' (first m) or 'random' (sample m groups).",
    )
    parser.add_argument(
        "--within-mode",
        choices=["head", "random"],
        default="head",
        help="How to choose which samples within each group to keep: 'head' (first n) or 'random' (sample n).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when mode is 'random'.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Indent for output JSON (default: None, compact).",
    )
    return parser.parse_args()


def choose_indices_head(count: int, k: int) -> List[int]:
    return list(range(min(k, count)))


def choose_indices_random(count: int, k: int, rng: random.Random) -> List[int]:
    k = min(k, count)
    if k <= 0:
        return []
    return rng.sample(range(count), k)


def load_records(path: str) -> Tuple[List[dict], str]:
    """
    Load records from JSON (a list) or JSONL (one object per line).
    Returns (records, detected_format) where detected_format is "json" or "jsonl".
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            # Likely JSONL.
            f.seek(0)
            records: List[dict] = []
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
                if not isinstance(rec, dict):
                    raise ValueError(f"JSONL records must be objects/dicts. Got {type(rec)} at {path}:{line_no}.")
                records.append(rec)
            return records, "jsonl"

    if not isinstance(obj, list):
        raise ValueError("Input JSON must be a list of records.")
    return obj, "json"


def dump_records(path: str, records: List[dict], indent: Optional[int]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=indent)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data, _detected_in_fmt = load_records(args.input)

    total_records = len(data)
    if total_records == 0:
        raise ValueError("Input dataset is empty.")

    if args.group_size <= 0:
        raise ValueError("group_size must be positive.")

    overall_num_questions = total_records // args.group_size
    remainder = total_records % args.group_size
    if remainder != 0:
        raise ValueError(
            f"Dataset length ({total_records}) is not divisible by group_size ({args.group_size}). "
            f"Remainder: {remainder}. Adjust --group-size or fix the input."
        )

    # Convention: negative num_questions (e.g. -1) means "keep all questions".
    if args.num_questions < 0:
        args.num_questions = overall_num_questions
    if args.num_questions == 0 or args.num_questions > overall_num_questions:
        raise ValueError(f"num_questions must be in [1, {overall_num_questions}] or <0 for all, got {args.num_questions}.")
    if args.num_samples <= 0 or args.num_samples > args.group_size:
        raise ValueError(f"num_samples must be in [1, {args.group_size}], got {args.num_samples}.")

    if args.group_mode == "head":
        selected_groups = choose_indices_head(overall_num_questions, args.num_questions)
    else:
        selected_groups = choose_indices_random(overall_num_questions, args.num_questions, rng)

    output_records: List[dict] = []
    for question_index in selected_groups:
        start = question_index * args.group_size
        end = start + args.group_size
        question_records = data[start:end]

        if args.within_mode == "head":
            within_indices = choose_indices_head(args.group_size, args.num_samples)
        else:
            within_indices = choose_indices_random(args.group_size, args.num_samples, rng)

        for idx in within_indices:
            output_records.append(question_records[idx])

    dump_records(args.output, output_records, indent=args.indent)

    print(
        f"Downsampled dataset written to: {args.output}\n"
        f"- Input records: {total_records} (groups: {overall_num_questions}, group_size: {args.group_size})\n"
        f"- Output records: {len(output_records)} (num_questions={args.num_questions}, num_samples={args.num_samples})\n"
        f"- group_mode: {args.group_mode}, within_mode: {args.within_mode}, seed: {args.seed}"
    )


if __name__ == "__main__":
    main()

