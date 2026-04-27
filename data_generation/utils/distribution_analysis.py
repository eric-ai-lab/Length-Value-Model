"""Analyze and visualize the length distribution of generated DAPO data.

Each sample must contain a valid `meta_info.answer_token_length` field
(positive number); missing or invalid values raise an error.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

PERCENTILE_LINE_COLORS = [
    "#B279A2",  # mauve
    "#F58518",  # orange
    "#4C78A8",  # blue
    "#54A24B",  # green
    "#72B7B2",  # teal
]
PERCENTILE_LINESTYLE = (0, (6, 3))
MEAN_LINE_COLOR = "#E45756" # red
MEAN_LINESTYLE = (0, (3, 3))


def load_token_lengths(data_path: Path) -> List[int]:
    """Load answer token lengths from JSONL file."""
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    lengths = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

            meta_info = entry.get("meta_info")
            if meta_info is None or "answer_token_length" not in meta_info:
                raise ValueError(
                    f"Line {line_num}: missing required field 'meta_info.answer_token_length'."
                )

            token_length = meta_info["answer_token_length"]
            if not isinstance(token_length, (int, float)) or token_length <= 0:
                raise ValueError(
                    f"Line {line_num}: invalid 'answer_token_length' value: {token_length!r}"
                )

            lengths.append(int(token_length))

    return lengths


def _compute_stats(lengths_array: np.ndarray) -> dict:
    percentiles = [25, 50, 75, 90, 95, 99, 99.9, 99.99, 99.999]
    return {
        "mean": np.mean(lengths_array),
        "median": np.median(lengths_array),
        "std": np.std(lengths_array),
        "min": int(np.min(lengths_array)),
        "max": int(np.max(lengths_array)),
        "percentiles": list(zip(percentiles, np.percentile(lengths_array, percentiles))),
    }


def _print_stats(stats: dict, n: int) -> None:
    print("\n" + "=" * 50)
    print("Statistics Summary")
    print("=" * 50)
    print(f"Total Samples:     {n:,}")
    print(f"Mean:              {stats['mean']:.2f}")
    print(f"Median:            {stats['median']:.2f}")
    print(f"Std Dev:           {stats['std']:.2f}")
    print(f"Min:               {stats['min']:,}")
    print(f"Max:               {stats['max']:,}")
    print(f"Range:             {stats['max'] - stats['min']:,}")
    print("\nPercentiles:")
    for p, v in stats["percentiles"]:
        print(f"  {p:>7g}th: {v:>10.0f}")
    print("=" * 50 + "\n")


def plot_distribution(
    lengths: List[int],
    output_path: Path,
    source_name: str,
    title: str = "Answer Token Length Distribution",
    bins: int = 50,
    vlines: List[float] = [50, 90, 95, 99],
) -> None:
    """Create and save distribution plots as a vector PDF.

    Args:
        vlines: Percentile values (0-100) at which to draw vertical lines,
                e.g. [90, 95, 99].
    """
    if not lengths:
        print("No data to plot.")
        return

    lengths_array = np.array(lengths)
    stats = _compute_stats(lengths_array)
    vline_xvals = [(p, float(np.percentile(lengths_array, p))) for p in vlines]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{title}\n{source_name}", fontsize=16, fontweight="bold")

    # 1. Histogram
    ax1 = axes[0, 0]
    ax1.hist(lengths, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(stats["mean"], color=MEAN_LINE_COLOR, linestyle=MEAN_LINESTYLE, linewidth=2,
                label=f"Mean: {stats['mean']:.1f}")
    for idx, (p, x) in enumerate(vline_xvals):
        ax1.axvline(
            x,
            color=PERCENTILE_LINE_COLORS[idx % len(PERCENTILE_LINE_COLORS)],
            linestyle=PERCENTILE_LINESTYLE,
            linewidth=2,
            label=f"p{p:g}: {x:.0f}",
        )
    ax1.set_xlabel("Token Length", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Histogram", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(lengths, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][0].set_alpha(0.7)
    ax2.set_ylabel("Token Length", fontsize=12)
    ax2.set_title("Box Plot", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_lengths = np.sort(lengths_array)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax3.plot(sorted_lengths, cumulative, linewidth=2, color="steelblue")
    ax3.axvline(stats["mean"], color=MEAN_LINE_COLOR, linestyle=MEAN_LINESTYLE, linewidth=2.5,
                label=f"Mean: {stats['mean']:.1f}")
    for idx, (p, x) in enumerate(vline_xvals):
        ax3.axvline(
            x,
            color=PERCENTILE_LINE_COLORS[idx % len(PERCENTILE_LINE_COLORS)],
            linestyle=PERCENTILE_LINESTYLE,
            linewidth=2,
            label=f"p{p:g}: {x:.0f}",
        )
    ax3.set_xlabel("Token Length", fontsize=12)
    ax3.set_ylabel("Cumulative Percentage (%)", fontsize=12)
    ax3.set_title("Cumulative Distribution", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    stats_lines = [
        "Statistics Summary",
        "=" * 40,
        "",
        f"Total Samples:     {len(lengths):,}",
        "",
        f"Mean:              {stats['mean']:.2f}",
        f"Median:            {stats['median']:.2f}",
        f"Std Dev:           {stats['std']:.2f}",
        "",
        f"Min:               {stats['min']:,}",
        f"Max:               {stats['max']:,}",
        f"Range:             {stats['max'] - stats['min']:,}",
        "",
        "Percentiles:",
        *[f"  {p:>7g}th: {v:>10.0f}" for p, v in stats["percentiles"]],
    ]
    ax4.text(
        0.1, 0.5, "\n".join(stats_lines),
        fontsize=11, family="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Distribution plot saved to: {output_path}")

    _print_stats(stats, len(lengths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution from generated DAPO data"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the JSONL data file")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save the plot (default: same dir as data file, .pdf)")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of bins for histogram (default: 50)")
    parser.add_argument("--title", type=str, default="Answer Token Length Distribution",
                        help="Title for the plot")
    parser.add_argument("--vlines", type=float, nargs="+", default=[50, 90, 95, 99],
                        metavar="P", help="Draw vertical lines at these percentiles, e.g. 90 95 99")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path) if args.output_path else data_path.with_suffix(".pdf")

    print(f"Loading data from: {data_path}")
    lengths = load_token_lengths(data_path)

    if not lengths:
        print("Error: No valid token lengths found in the data file.")
        return

    print(f"Loaded {len(lengths):,} samples.")

    # source_name: last directory component + filename, e.g. "mydir/data.jsonl"
    source_name = str(Path(data_path.parent.name) / data_path.name)
    plot_distribution(lengths, output_path, source_name=source_name,
                      title=args.title, bins=args.bins, vlines=args.vlines)


if __name__ == "__main__":
    main()
