import argparse
import re
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def extract_nfound(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"nfound=(\d+)", text)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot nfound frequency histogram (per-integer bars).")
    parser.add_argument("--input", default="logs/nfound.log", help="Path to log file")
    parser.add_argument("--output", default="logs/nfound_hist.png", help="Path to output PNG")
    parser.add_argument(
        "--mode",
        choices=("frequency", "count"),
        default="frequency",
        help="Y-axis mode",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include missing integers between min/max as zero bars",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8", errors="replace")
    nfound = extract_nfound(text)
    if not nfound:
        raise SystemExit(f"No nfound found in {in_path}")

    counts = Counter(nfound)
    if args.include_missing:
        x_vals = np.arange(min(nfound), max(nfound) + 1, dtype=int)
    else:
        x_vals = np.array(sorted(counts.keys()), dtype=int)

    y_counts = np.array([counts.get(int(x), 0) for x in x_vals], dtype=int)
    total = int(y_counts.sum())
    y_freq = y_counts / total

    plt.figure(figsize=(14, 5))
    if args.mode == "frequency":
        plt.bar(x_vals, y_freq, width=0.9)
        plt.ylabel("Frequency")
    else:
        plt.bar(x_vals, y_counts, width=0.9)
        plt.ylabel("Count")

    plt.xlabel("nfound")
    plt.title(f"nfound distribution ({args.mode}, N={total})")
    if len(x_vals) <= 60:
        plt.xticks(x_vals, rotation=90)
    else:
        plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)

    # Print quick summary for CLI usage
    mode_val, mode_count = counts.most_common(1)[0]
    print(f"Saved: {out_path}")
    print(f"N={len(nfound)}; mode_nfound={mode_val}; mode_count={mode_count}; mode_freq={mode_count/len(nfound):.4f}")


if __name__ == "__main__":
    main()
