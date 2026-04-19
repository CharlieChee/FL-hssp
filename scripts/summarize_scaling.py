#!/usr/bin/env python3
"""
Summarize trial logs under logs/scaling/, grouped by (CNN_INT_SCALE, batch_size),
and report the average wall-clock time and average success rate.

Usage:
    python scripts/summarize_scaling.py logs/scaling
    python scripts/summarize_scaling.py logs/scaling --min_elapsed 0.5

Extracted from log contents:
    - CNN_INT_SCALE: "integer pool: floor(original x <scale>)..." or
      "integer pool: 0/1 binarized..." (scale=-1)
    - batch_size (n): filename n<N> or "BATCH_SIZE=<N>" in the log
    - NFound: "NFound(rows match X.T)= <nf> / <n>"
    - elapsed: "[total_elapsed] <sec>s"

Time rule: trials with elapsed <= min_elapsed are excluded from the time average.
"""
import argparse
import os
import re
import sys
from collections import defaultdict


def parse_log(filepath):
    """Return (scale, n, nfound, elapsed) or None."""
    scale = None
    n_total = None
    nfound = None
    elapsed = None

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            m_bin = re.search(r"integer pool: 0/1 binarized", line)
            if m_bin:
                scale = -1

            m_scale = re.search(r"integer pool: floor\(original x ([\d.]+)\)", line)
            if m_scale:
                scale = float(m_scale.group(1))
                if scale == int(scale):
                    scale = int(scale)

            m_nf = re.search(r"NFound\(rows match X\.T\)=\s*(\d+)\s*/\s*(\d+)", line)
            if m_nf:
                nfound = int(m_nf.group(1))
                n_total = int(m_nf.group(2))

            m_nf2 = re.search(r"NFound\(rows vs X\.T\)=\s*(\d+)\s*/\s*(\d+)", line)
            if m_nf2:
                nfound = int(m_nf2.group(1))
                n_total = int(m_nf2.group(2))

            m_el = re.search(r"\[total_elapsed\]\s*([\d.]+)s", line)
            if m_el:
                elapsed = float(m_el.group(1))

    if n_total is None:
        pat_fn = re.compile(r"_n(\d+)_m(\d+)_")
        fm = pat_fn.search(os.path.basename(filepath))
        if fm:
            n_total = int(fm.group(1))

    if n_total is None or elapsed is None:
        return None
    if nfound is None:
        nfound = 0
    if scale is None:
        scale = "?"

    return scale, n_total, nfound, elapsed


def main():
    parser = argparse.ArgumentParser(description="Summarize trial logs grouped by (CNN_INT_SCALE, batch_size)")
    parser.add_argument("logdir", help="log directory, e.g. logs/scaling")
    parser.add_argument(
        "--min_elapsed",
        type=float,
        default=0.5,
        help="trials with elapsed <= this value are excluded from the time average (default 0.5s)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        print("Error: directory does not exist: %s" % args.logdir, file=sys.stderr)
        sys.exit(1)

    # group key: (scale, n) -> list of (nfound, n_total, elapsed)
    groups = defaultdict(list)
    skipped = 0

    for fname in sorted(os.listdir(args.logdir)):
        if not fname.endswith(".log") or not fname.startswith("trial_"):
            continue
        result = parse_log(os.path.join(args.logdir, fname))
        if result is None:
            skipped += 1
            print("Warning: cannot parse %s; skipping" % fname, file=sys.stderr)
            continue

        scale, n_total, nfound, elapsed = result
        groups[(scale, n_total)].append((nfound, n_total, elapsed))

    if skipped:
        print("(skipped %d unparsable files)\n" % skipped, file=sys.stderr)

    if not groups:
        print("No parsable trial logs found.")
        return

    print("=" * 80)
    print(
        "%-12s %-6s %-8s %-14s %-14s %-10s"
        % ("scale", "n", "trials", "avg_success", "avg_time(s)", "time_trials")
    )
    print("-" * 80)

    _printed = 0
    for key in sorted(groups.keys(), key=lambda k: (str(k[0]), k[1])):
        scale, n_val = key
        records = groups[key]
        total_trials = len(records)

        success_rates = [
            nf / float(nt)
            for nf, nt, _ in records
            if nf > 0 and (nf / float(nt)) > 0
        ]
        if not success_rates:
            continue
        avg_success = sum(success_rates) / len(success_rates)

        valid_times = [el for _, _, el in records if el > args.min_elapsed]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            time_str = "%.1f" % avg_time
        else:
            time_str = "N/A"
        time_trials = len(valid_times)

        scale_label = "binary(-1)" if scale == -1 else str(scale)
        print(
            "%-12s %-6d %-8d %-14.2f %-14s %-10d"
            % (scale_label, n_val, total_trials, avg_success, time_str, time_trials)
        )
        _printed += 1

    if _printed == 0:
        print("(no group has success rate > 0)")

    print("=" * 80)
    print(
        "Note: scale=binary(-1) denotes 0/1 binarization; "
        "avg_success averages only trials with success rate > 0, all-zero groups omitted; "
        "avg_time counts only trials with elapsed > %.1fs" % args.min_elapsed
    )


if __name__ == "__main__":
    main()
