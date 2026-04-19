#!/usr/bin/env python3
"""
Summarize trial logs under logs/<subdir>/, grouped by m (TARGET_ROWS),
and report the average wall-clock time and average success rate.

Usage:
    python scripts/summarize_time_and_success.py logs/time_and_success
    python scripts/summarize_time_and_success.py logs/time_and_success --min_elapsed 0.5

Log filename format:
    trial_0001_20260406_151619_n10_m25_attack_ns_rowseed_xxx_attseed_yyy.log

n and m are taken from the filename; from the file contents:
    - NFound:        ">>> NS NFound(rows match X.T)= <nf> / <n>"
    - elapsed:       "[total_elapsed] <sec>s"

Rules:
    - success rate = nfound / n (one value per trial), averaged per m group
    - time: trials with elapsed <= min_elapsed are treated as "not really run"
      and excluded from the average
"""
import argparse
import os
import re
import sys
from collections import defaultdict


def parse_log(filepath, min_elapsed):
    """Return (nfound, n, elapsed); None on parse failure."""
    nfound = None
    n_total = None
    elapsed = None

    with open(filepath, "r", errors="replace") as f:
        for line in f:
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

    if n_total is None or elapsed is None:
        return None

    if nfound is None:
        nfound = 0

    return nfound, n_total, elapsed


def main():
    parser = argparse.ArgumentParser(description="Summarize wall-clock time and success rate from trial logs")
    parser.add_argument("logdir", help="log directory, e.g. logs/time_and_success")
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

    pat_filename = re.compile(
        r"trial_\d+_\d+_\d+_n(\d+)_m(\d+)_attack_(\w+)_rowseed_\d+_attseed_\d+"
        r"(?:_xsrc[a-z0-9]+)?(?:_bx\d+_xnz[0-9pm]+)?\.log"
    )

    # group key: (n, m, attack) -> list of (nfound, n_total, elapsed)
    groups = defaultdict(list)

    for fname in sorted(os.listdir(args.logdir)):
        fm = pat_filename.match(fname)
        if not fm:
            continue
        n_val = int(fm.group(1))
        m_val = int(fm.group(2))
        attack = fm.group(3)

        result = parse_log(os.path.join(args.logdir, fname), args.min_elapsed)
        if result is None:
            print("Warning: cannot parse %s; skipping" % fname, file=sys.stderr)
            continue

        groups[(n_val, m_val, attack)].append(result)

    if not groups:
        print("No parsable trial logs found.")
        return

    print("=" * 80)
    print(
        "%-6s %-6s %-12s %-8s %-14s %-14s %-10s"
        % ("n", "m", "attack", "trials", "avg_success", "avg_time(s)", "time_trials")
    )
    print("-" * 80)

    _printed = 0
    for key in sorted(groups.keys()):
        n_val, m_val, attack = key
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

        print(
            "%-6d %-6d %-12s %-8d %-14.2f %-14s %-10d"
            % (n_val, m_val, attack, total_trials, avg_success, time_str, time_trials)
        )
        _printed += 1

    if _printed == 0:
        print("(no group has success rate > 0)")

    print("=" * 80)
    print(
        "Note: avg_success averages only trials with success rate > 0, all-zero groups omitted; "
        "avg_time counts only trials with elapsed > %.1fs" % args.min_elapsed
    )


if __name__ == "__main__":
    main()
