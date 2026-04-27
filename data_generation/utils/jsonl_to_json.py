"""Convert a JSONL file to a JSON file with list-of-rows format.

Usage example:

python jsonl_to_json.py \
  --input ../data/dapo_math_17k_7398_16.jsonl \
  --output ../data/dapo_math_17k_7398_16.json

The output JSON file will contain a list, where each element is
one parsed JSON object from each non-empty line of the JSONL file.
Invalid JSON lines will be skipped with a warning.
"""

import argparse
import json
from pathlib import Path
from typing import List, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL file to JSON list file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output JSON file. Default: same as input with .json suffix.")
    parser.add_argument("--max-lines", type=int, default=-1, help="Maximum number of lines to read (-1 means all).")
    return parser.parse_args()


def jsonl_to_list(path: Path, max_lines: int = -1) -> List[Any]:
    data: List[Any] = []
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if max_lines > 0 and i > max_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse line {i}: {e}")
                continue

            data.append(obj)

    return data


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if args.output is not None:
        output_path = Path(args.output)
    else:
        if input_path.suffix == ".jsonl":
            output_path = input_path.with_suffix(".json")
        else:
            output_path = input_path.with_name(input_path.name + ".json")

    data_list = jsonl_to_list(input_path, max_lines=args.max_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(data_list)} lines from {input_path} to JSON list file: {output_path}")


if __name__ == "__main__":
    main()
