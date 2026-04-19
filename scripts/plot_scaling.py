#!/usr/bin/env python3
"""
From trial logs under logs/scaling/, grouped by (CNN_INT_SCALE, batch_size),
draw a dual-Y-axis figure: left axis = avg time (s), right axis = avg success
rate, with standard-deviation bands.

Usage:
    python scripts/plot_scaling.py logs/scaling
    python scripts/plot_scaling.py logs/scaling --min_elapsed 0.5 -o logs/scaling/plot.png
"""
import argparse
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_log(filepath):
    """Return (scale, n, nfound, elapsed) or None."""
    scale = None
    n_total = None
    nfound = None
    elapsed = None

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            if re.search(r"integer pool: 0/1 binarized", line):
                scale = -1
            m_scale = re.search(r"integer pool: floor\(original x ([\d.]+)\)", line)
            if m_scale:
                s = float(m_scale.group(1))
                scale = int(s) if s == int(s) else s

            m_nf = re.search(r"NFound\(rows match X\.T\)=\s*(\d+)\s*/\s*(\d+)", line)
            if m_nf:
                nfound, n_total = int(m_nf.group(1)), int(m_nf.group(2))
            m_nf2 = re.search(r"NFound\(rows vs X\.T\)=\s*(\d+)\s*/\s*(\d+)", line)
            if m_nf2:
                nfound, n_total = int(m_nf2.group(1)), int(m_nf2.group(2))

            m_el = re.search(r"\[total_elapsed\]\s*([\d.]+)s", line)
            if m_el:
                elapsed = float(m_el.group(1))

    if n_total is None:
        fm = re.search(r"_n(\d+)_m(\d+)_", os.path.basename(filepath))
        if fm:
            n_total = int(fm.group(1))

    if n_total is None or elapsed is None:
        return None
    if nfound is None:
        nfound = 0
    if scale is None:
        scale = "?"
    return scale, n_total, nfound, elapsed


def collect_data(logdir, min_elapsed):
    """Return {scale: {n: {'success_rates': [...], 'times': [...]}}}"""
    raw = defaultdict(list)
    for fname in sorted(os.listdir(logdir)):
        if not fname.endswith(".log") or not fname.startswith("trial_"):
            continue
        result = parse_log(os.path.join(logdir, fname))
        if result is None:
            continue
        scale, n_total, nfound, elapsed = result
        raw[(scale, n_total)].append((nfound, n_total, elapsed))

    data = defaultdict(dict)
    for (scale, n_val), records in raw.items():
        sr = [
            nf / float(nt)
            for nf, nt, _ in records
            if nf > 0 and (nf / float(nt)) > 0
        ]
        vt = [el for _, _, el in records if el > min_elapsed]
        data[scale][n_val] = {
            "success_rates": sr,
            "times": vt,
        }
    return data


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def main():
    parser = argparse.ArgumentParser(description="Plot the dual-axis time & success-rate figure for the scaling sweep")
    parser.add_argument("logdir", help="log directory, e.g. logs/scaling")
    parser.add_argument("--min_elapsed", type=float, default=0.5)
    parser.add_argument("-o", "--output", default="", help="output image path (default <logdir>/scaling_plot.png)")
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        print("Error: directory does not exist: %s" % args.logdir, file=sys.stderr)
        sys.exit(1)

    data = collect_data(args.logdir, args.min_elapsed)
    if not data:
        print("No parsable trial logs found.")
        return

    scale_order = sorted(data.keys(), key=lambda s: (0, s) if isinstance(s, (int, float)) else (1, str(s)))
    all_ns = sorted(set(n for sd in data.values() for n in sd))

    scale_labels = {s: ("binary" if s == -1 else str(s)) for s in scale_order}
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(scale_order), 1)))

    fig, ax_time = plt.subplots(figsize=(10, 6))
    ax_sr = ax_time.twinx()

    for idx, scale in enumerate(scale_order):
        sd = data[scale]
        ns = sorted(sd.keys())
        if not ns:
            continue

        color = colors[idx]
        label = scale_labels[scale]

        # --- time (left axis, solid line) ---
        t_ns, t_means, t_stds = [], [], []
        for n in ns:
            vals = sd[n]["times"]
            if vals:
                t_ns.append(n)
                t_means.append(sum(vals) / len(vals))
                t_stds.append(_std(vals))
        if t_ns:
            t_means_arr = np.array(t_means)
            t_stds_arr = np.array(t_stds)
            ax_time.plot(t_ns, t_means, "o-", color=color, label="time: c=%s" % label)
            ax_time.fill_between(
                t_ns,
                np.maximum(t_means_arr - t_stds_arr, 0),
                t_means_arr + t_stds_arr,
                color=color, alpha=0.12,
            )

        # --- success rate (right axis, dashed line) ---
        s_ns, s_means, s_stds = [], [], []
        for n in ns:
            vals = sd[n]["success_rates"]
            if not vals:
                continue
            s_ns.append(n)
            s_means.append(sum(vals) / len(vals))
            s_stds.append(_std(vals))
        if s_ns:
            s_means_arr = np.array(s_means)
            s_stds_arr = np.array(s_stds)
            ax_sr.plot(s_ns, s_means, "s--", color=color, label="success: c=%s" % label)
            ax_sr.fill_between(
                s_ns,
                np.clip(s_means_arr - s_stds_arr, 0, 1),
                np.clip(s_means_arr + s_stds_arr, 0, 1),
                color=color, alpha=0.08,
            )

    ax_time.set_xlabel("batch_size (n)", fontsize=13)
    ax_time.set_ylabel("Avg Time (s)", fontsize=13)
    ax_sr.set_ylabel("Avg Success Rate", fontsize=13)
    ax_sr.set_ylim(-0.05, 1.10)
    ax_time.set_xticks(all_ns)

    lines1, labels1 = ax_time.get_legend_handles_labels()
    lines2, labels2 = ax_sr.get_legend_handles_labels()
    ax_time.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", fontsize=9, ncol=2, framealpha=0.9,
    )

    plt.title("Scaling Experiment: Time & Success Rate vs batch_size", fontsize=14)
    fig.tight_layout()

    out_path = args.output or os.path.join(args.logdir, "scaling_plot.png")
    fig.savefig(out_path, dpi=150)
    print("Figure saved: %s" % out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
