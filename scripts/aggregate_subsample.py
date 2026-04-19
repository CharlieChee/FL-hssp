#!/usr/bin/env python3
"""Aggregate transpose-attack subsample sweep results.

The companion sage driver ``lattice/run_mhlcp_transpose.sage`` writes one
pipe-separated ``RESULT|...`` line per trial. This script reads many such
log files (one trial per file is fine, multiple per file also fine) and
emits a per-``(bs, c)`` summary as JSON.

Both row-selection modes share the same RESULT format, so a single aggregator
serves both. Pick the mode via ``--mode {random,bias}`` (it only affects the
output filename).

Usage::

    # Aggregate a sweep of random-row-selection runs:
    python scripts/aggregate_subsample.py \\
        --logs-dir logs/random_subsample \\
        --pattern 'bs*_c*_run*.log' \\
        --mode random \\
        --out logs/summary_random_subsample.json

    # Aggregate a sweep of bias-row-selection runs:
    python scripts/aggregate_subsample.py \\
        --logs-dir logs/bias_subsample \\
        --pattern 'bs*_c*_run*.log' \\
        --mode bias \\
        --out logs/summary_bias_subsample.json
"""
import argparse
import glob
import json
import os
import re


def parse_result(line):
    """Parse a single ``RESULT|k=v|k=v|...`` line into a dict, or return None."""
    if not line.startswith("RESULT|"):
        return None
    out = {}
    for kv in line.strip().split("|")[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def num(value, default=None):
    """Coerce ``value`` to ``float`` or fall back to ``default`` on ParseError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ``bs<BS>_c<C>_run<R>.log`` is the canonical filename used by the sample sweep
# in ``experiments/run_random_m7n_fill.sh``; we also accept ``bs<BS>_run<R>.txt``
# from the older bias sweep (where ``c`` was a directory level).
_RE_BS_C_RUN = re.compile(r"bs(\d+)_c(\d+)_run(\d+)")
_RE_BS_RUN   = re.compile(r"bs(\d+)_run(\d+)")


def derive_bs_c_run(path, parent_c=None):
    """Pull (bs, c, run) out of a trial-log filename. Falls back to ``parent_c``."""
    name = os.path.basename(path)
    m = _RE_BS_C_RUN.search(name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = _RE_BS_RUN.search(name)
    if m:
        return int(m.group(1)), parent_c, int(m.group(2))
    return None, parent_c, None


def aggregate(log_files, parent_c=None):
    """Walk ``log_files``; group by ``(bs, c)`` and produce mean / count metrics."""
    summary = {}
    for path in log_files:
        bs, c_from_name, _run = derive_bs_c_run(path, parent_c=parent_c)
        if bs is None:
            continue
        try:
            with open(path) as fh:
                lines = fh.read().splitlines()
        except (FileNotFoundError, PermissionError):
            continue

        for line in lines:
            r = parse_result(line)
            if r is None:
                continue
            c = int(r["c"]) if "c" in r else c_from_name
            if c is None:
                continue
            bucket = summary.setdefault(bs, {}).setdefault(c, {
                "N": 0, "n_succ": 0, "succ_t": [], "succ_step1": [], "succ_step2": [],
            })
            bucket["N"] += 1
            if r.get("st") == "S":
                bucket["n_succ"] += 1
                t       = num(r.get("ttotal"))
                t_step1 = num(r.get("tstep1"))
                if t is not None:
                    bucket["succ_t"].append(t)
                if t_step1 is not None:
                    bucket["succ_step1"].append(t_step1)
                    if t is not None:
                        bucket["succ_step2"].append(t - t_step1)
    return summary


def finalize(summary):
    """Reduce per-(bs, c) lists to ``(N, success_rate, avg_t, avg_step1, avg_step2)``."""
    out = {}
    for bs in sorted(summary):
        out[str(bs)] = {}
        for c in sorted(summary[bs]):
            b = summary[bs][c]

            def _mean(xs):
                return float(sum(xs)) / len(xs) if xs else None

            out[str(bs)][str(c)] = {
                "N":            b["N"],
                "n_succ":       b["n_succ"],
                "success_rate": b["n_succ"] / b["N"] if b["N"] else 0.0,
                "avg_ttotal":   _mean(b["succ_t"]),
                "avg_tstep1":   _mean(b["succ_step1"]),
                "avg_tstep2":   _mean(b["succ_step2"]),
            }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs-dir", required=True,
                    help="Directory containing trial log files.")
    ap.add_argument("--pattern", default="bs*_c*_run*.log",
                    help="Glob pattern, relative to --logs-dir.")
    ap.add_argument("--mode", required=True, choices=("random", "bias"),
                    help="Row-selection mode (only affects messaging / default output name).")
    ap.add_argument("--parent-c", type=int, default=None,
                    help="C value to attribute to filenames that lack `_c<C>_` "
                         "(used for the older bias sweep where C was a directory level).")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: logs/summary_<mode>_subsample.json).")
    args = ap.parse_args()

    out_path = args.out or os.path.join("logs", f"summary_{args.mode}_subsample.json")
    pattern = os.path.join(args.logs_dir, args.pattern)
    log_files = sorted(glob.glob(pattern))
    if not log_files:
        raise SystemExit(f"No files matched: {pattern}")

    print(f"[aggregate_subsample] mode={args.mode} files={len(log_files)} -> {out_path}")
    raw = aggregate(log_files, parent_c=args.parent_c)
    summary = finalize(raw)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[aggregate_subsample] wrote {out_path}")


if __name__ == "__main__":
    main()
