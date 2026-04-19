#!/usr/bin/env python3
"""
Summarize logs/random_dense (or any trial directory), grouped by
(MHLCP_X_SOURCE, batch_size, X_non_zero), and report average success rate.

Usage:
    python scripts/summarize_random_dense.py logs/random_dense

Group keys:
    - MHLCP_X_SOURCE: from filename _xsrc<tag>; if absent, inferred from log
    - batch_size: n from the NFound log line (matches filename _n<N>_)
    - X_non_zero: from filename _xnz<token> (same convention as run_mhlcp:
      p=decimal point, m=minus sign); if absent, recorded as N/A (e.g. CNN)

Success rate: only trials with nfound/n > 0 contribute to the average;
trials with success rate 0 are excluded.
If a group has no trial with success rate > 0, that row is omitted.
Incomplete logs (missing [total_elapsed]) are skipped.
"""
import argparse
import os
import re
import sys
from collections import defaultdict


def parse_xsrc_from_filename(fname):
    m = re.search(r"_xsrc([a-z0-9]+)", fname)
    return m.group(1) if m else None


def parse_n_from_filename(fname):
    m = re.search(r"_n(\d+)_m(\d+)_", fname)
    return int(m.group(1)) if m else None


def decode_xnz_token(tok):
    """In filenames xnz0p7 -> 0.7; xnz1 -> 1.0"""
    if not tok:
        return None
    s = tok.replace("p", ".")
    if s.startswith("m") and len(s) > 1:
        s = "-" + s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def parse_xnz_from_filename(fname):
    m = re.search(r"_xnz([0-9pm]+)", fname)
    return decode_xnz_token(m.group(1)) if m else None


def infer_xsrc_from_log(text):
    if re.search(r"X source:\s*random", text):
        return "random"
    if re.search(r"loaded original X before FC1 ReLU", text):
        return "real"
    if re.search(r"CNN_USE_RELU_MASK_X", text) or re.search(r"\[mask mode\]", text):
        return "relu_mask"
    return "unknown"


def parse_log(filepath):
    """Return (nfound, n_total, elapsed) or None."""
    nfound = None
    n_total = None
    elapsed = None
    with open(filepath, "r", errors="replace") as f:
        text = f.read()

    for line in text.splitlines():
        m_nf = re.search(r"NFound\(rows match X\.T\)=\s*(\d+)\s*/\s*(\d+)", line)
        if m_nf:
            nfound, n_total = int(m_nf.group(1)), int(m_nf.group(2))
        m_nf2 = re.search(r"NFound\(rows vs X\.T\)=\s*(\d+)\s*/\s*(\d+)", line)
        if m_nf2:
            nfound, n_total = int(m_nf2.group(1)), int(m_nf2.group(2))
        m_el = re.search(r"\[total_elapsed\]\s*([\d.]+)s", line)
        if m_el:
            elapsed = float(m_el.group(1))

    if elapsed is None:
        return None
    if n_total is None:
        n_total = parse_n_from_filename(os.path.basename(filepath))
    if n_total is None:
        return None
    if nfound is None:
        nfound = 0
    return nfound, n_total, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Summarize trial success rate grouped by (MHLCP_X_SOURCE, batch_size, X_non_zero)"
    )
    parser.add_argument(
        "logdir",
        nargs="?",
        default="logs/random_dense",
        help="log directory (default logs/random_dense)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        print("Error: directory does not exist: %s" % args.logdir, file=sys.stderr)
        sys.exit(1)

    groups = defaultdict(list)
    skipped = 0

    for fname in sorted(os.listdir(args.logdir)):
        if not fname.endswith(".log") or not fname.startswith("trial_"):
            continue
        path = os.path.join(args.logdir, fname)
        result = parse_log(path)
        if result is None:
            skipped += 1
            print("Warning: skipped (incomplete or unparsable) %s" % fname, file=sys.stderr)
            continue

        nfound, n_total, _elapsed = result
        n_fn = parse_n_from_filename(fname)
        if n_fn is not None and n_fn != n_total:
            print(
                "Warning: %s filename n=%s disagrees with log n=%s; using log value"
                % (fname, n_fn, n_total),
                file=sys.stderr,
            )

        xsrc = parse_xsrc_from_filename(fname)
        if xsrc is None:
            with open(path, "r", errors="replace") as rf:
                xsrc = infer_xsrc_from_log(rf.read())

        xnz = parse_xnz_from_filename(fname)

        groups[(xsrc, n_total, xnz)].append((nfound, n_total))

    if skipped:
        print("(skipped %d files)\n" % skipped, file=sys.stderr)

    if not groups:
        print("No parsable trial logs found.")
        return

    print("=" * 88)
    print(
        "%-16s %-10s %-12s %-10s %-16s %-12s"
        % ("MHLCP_X_SOURCE", "batch_size", "X_non_zero", "trials", "avg_success", "nfound>0")
    )
    print("-" * 88)

    def _sort_key(k):
        xsrc, n_val, xnz = k
        xnz_s = float("nan") if xnz is None else float(xnz)
        return (xsrc, n_val, xnz_s)

    _printed = 0
    for key in sorted(groups.keys(), key=_sort_key):
        xsrc, n_val, xnz = key
        records = groups[key]
        total = len(records)
        rates = [
            nf / float(nt)
            for nf, nt in records
            if nf > 0 and (nf / float(nt)) > 0
        ]
        if not rates:
            continue
        avg = sum(rates) / len(rates)
        nz = len(rates)
        xnz_str = "N/A" if xnz is None else ("%g" % xnz)
        print(
            "%-16s %-10d %-12s %-10d %-16.4f %-12d"
            % (xsrc, n_val, xnz_str, total, avg, nz)
        )
        _printed += 1

    if _printed == 0:
        print("(no group has success rate > 0; no data rows emitted)")

    print("=" * 88)
    print(
        "Note: avg_success averages only trials with success rate > 0; "
        "groups that are all zero are omitted. "
        "X_non_zero is taken from filename _xnz...; N/A if absent."
    )


if __name__ == "__main__":
    main()
