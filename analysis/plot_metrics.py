#!/usr/bin/env python3
"""
Plot GIA sweep metrics from NPZ files in TPAMI publication style.

Usage:
    python plot_metrics.py <path_to_npz>
    python plot_metrics.py <path_to_npz> --out_dir ./figures
    python plot_metrics.py <path_to_npz> --format pdf
    python plot_metrics.py <path_to_npz> --separate
"""

import argparse
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# TPAMI style configuration
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.80",
    "legend.fancybox": False,
    "legend.handlelength": 2.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # TrueType — editable text in PDF
    "ps.fonttype": 42,
}

COLORS = {
    "naive": "#2166ac",
    "hssp": [
        "#d6604d",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
    ],
    "hlcp": "#000000",
}

MARKERS = ["o", "s", "^", "D", "v", "p", "h"]

IEEE_COL_W = 3.5   # single-column width in inches (IEEE)
IEEE_DBL_W = 7.16   # double-column width in inches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_sigma(sigma: float) -> str:
    if sigma == 0:
        return "HSSP + GIA"
    return rf"HSSP + GIA ($\sigma\!=\!{sigma:g}$)"


def _setup_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _legend_auto_place(ax, candidates=None, point_margin_px: float = 4.0):
    """
    Automatically pick a legend location that minimizes overlap with plotted data points.

    Heuristic:
      - Try a set of candidate `loc` values.
      - For each candidate, create a temporary legend, compute its bbox (display coords),
        then count how many line-marker points fall inside the bbox.
      - Keep the candidate with minimum count (ties broken by first minimal).
    """
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    if candidates is None:
        candidates = [
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "center right",
            "center left",
            "upper center",
            "lower center",
        ]

    # Ensure renderer exists
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Pre-compute all line data points in display coordinates
    point_xy_disp = []
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=np.float64)
        y = np.asarray(line.get_ydata(), dtype=np.float64)
        if x.shape != y.shape:
            continue
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        pts = ax.transData.transform(np.column_stack([x[mask], y[mask]]))
        point_xy_disp.append(pts)
    if point_xy_disp:
        all_pts = np.concatenate(point_xy_disp, axis=0)
    else:
        all_pts = np.zeros((0, 2), dtype=np.float64)

    best_loc = candidates[0]
    best_score = None

    for loc in candidates:
        leg = ax.legend(handles, labels, loc=loc)
        fig.canvas.draw()
        bbox = leg.get_window_extent(renderer=renderer)  # display coords
        leg.remove()

        xmin, ymin, w, h = bbox.bounds
        xmax = xmin + w
        ymax = ymin + h

        if all_pts.shape[0] == 0:
            score = 0
        else:
            # Add a small margin so we treat near-misses as overlap.
            xmin2 = xmin - point_margin_px
            ymin2 = ymin - point_margin_px
            xmax2 = xmax + point_margin_px
            ymax2 = ymax + point_margin_px
            inside = (
                (all_pts[:, 0] >= xmin2)
                & (all_pts[:, 0] <= xmax2)
                & (all_pts[:, 1] >= ymin2)
                & (all_pts[:, 1] <= ymax2)
            )
            score = int(np.count_nonzero(inside))

        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc

    return ax.legend(handles, labels, loc=best_loc)


def _add_curve(
    ax,
    x,
    y_mean,
    y_std,
    label,
    color,
    marker,
    zorder=2,
    band_clip: Optional[Tuple[float, float]] = None,
):
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    if band_clip is not None:
        lo_c, hi_c = float(band_clip[0]), float(band_clip[1])
        y_mean = np.clip(y_mean, lo_c, hi_c)
        y_std = np.maximum(y_std, 0.0)
        band_lo = np.clip(y_mean - y_std, lo_c, hi_c)
        band_hi = np.clip(y_mean + y_std, lo_c, hi_c)
    else:
        band_lo = y_mean - y_std
        band_hi = y_mean + y_std
    ax.plot(
        x, y_mean,
        marker=marker,
        markersize=3.5,
        markeredgewidth=0.5,
        markeredgecolor="white",
        linewidth=1.2,
        label=label,
        color=color,
        zorder=zorder,
    )
    ax.fill_between(
        x,
        band_lo,
        band_hi,
        color=color,
        alpha=0.15,
        edgecolor="none",
        zorder=zorder - 1,
    )


def _add_line(ax, x, y, label, color, marker, zorder=2, linestyle="-"):
    ax.plot(
        x,
        y,
        marker=marker,
        markersize=3.5,
        markeredgewidth=0.5,
        markeredgecolor="white",
        linewidth=1.2,
        label=label,
        color=color,
        zorder=zorder,
        linestyle=linestyle,
    )


def _add_envelope_between(ax, x, y1, y2, color="0.75", alpha=0.22, zorder=1):
    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    lo = np.minimum(y1, y2)
    hi = np.maximum(y1, y2)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, edgecolor="none", zorder=zorder)


def _save_figure(fig, out_path, stem, fmt):
    """Save requested format and always save PNG."""
    saved = []
    primary = os.path.join(out_path, f"{stem}.{fmt}")
    fig.savefig(primary)
    saved.append(primary)
    if fmt != "png":
        png_path = os.path.join(out_path, f"{stem}.png")
        fig.savefig(png_path)
        saved.append(png_path)
    return saved


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _unique_legend_items(handles, labels):
    seen = set()
    out_h, out_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        out_h.append(h)
        out_l.append(l)
    return out_h, out_l


def plot_combined(data, out_path, fmt="pdf", stem_suffix: str = ""):
    """Side-by-side PSNR + SSIM figure (double-column width), with a shared legend."""
    bs = data["batch_sizes"]
    sigmas = data["noise_on_fc"]

    fig, (ax_psnr, ax_ssim) = plt.subplots(
        1, 2,
        figsize=(IEEE_DBL_W, 2.55),
        constrained_layout=False,
    )

    # --- PSNR ---
    _add_curve(ax_psnr, bs,
               data["psnr_naive_mean"], data["psnr_naive_std"],
               "Naive GIA", COLORS["naive"], MARKERS[0], zorder=3)
    if "psnr_hlcp_mean" in data and "psnr_hlcp_std" in data:
        _add_line(ax_psnr, bs, data["psnr_hlcp_mean"], "HLCP + GIA", COLORS["hlcp"], "X", zorder=4)
    for i, sigma in enumerate(sigmas):
        _add_curve(ax_psnr, bs,
                   data["psnr_feat_mean"][i], data["psnr_feat_std"][i],
                   _label_sigma(sigma),
                   COLORS["hssp"][i % len(COLORS["hssp"])],
                   MARKERS[(i + 1) % len(MARKERS)],
                   )
    _setup_ax(ax_psnr, "Batch size", "PSNR (dB)")
    ax_psnr.set_title("(a)", fontsize=9, pad=4)

    # --- SSIM ---
    _add_curve(ax_ssim, bs,
               data["ssim_naive_mean"], data["ssim_naive_std"],
               "Naive GIA", COLORS["naive"], MARKERS[0], zorder=3)
    if "ssim_hlcp_mean" in data and "ssim_hlcp_std" in data:
        _add_line(ax_ssim, bs, data["ssim_hlcp_mean"], "HLCP + GIA", COLORS["hlcp"], "X", zorder=4)
    for i, sigma in enumerate(sigmas):
        _add_curve(ax_ssim, bs,
                   data["ssim_feat_mean"][i], data["ssim_feat_std"][i],
                   _label_sigma(sigma),
                   COLORS["hssp"][i % len(COLORS["hssp"])],
                   MARKERS[(i + 1) % len(MARKERS)],
                   )
    _setup_ax(ax_ssim, "Batch size", "SSIM")
    ax_ssim.set_title("(b)", fontsize=9, pad=4)
    ax_ssim.set_ylim(0.0, 1.0)

    # Shared legend (paper style): place in a dedicated top strip, outside axes.
    h1, l1 = ax_psnr.get_legend_handles_labels()
    h2, l2 = ax_ssim.get_legend_handles_labels()
    handles, labels = _unique_legend_items(h1 + h2, l1 + l2)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(3, max(1, len(labels))),
        frameon=True,
        borderaxespad=0.2,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    # Reserve space for the shared legend (top) and keep margins tight.
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.83, wspace=0.24)

    saved_files = _save_figure(fig, out_path, f"metrics_vs_batchsize{stem_suffix}", fmt)
    plt.close(fig)
    for fname in saved_files:
        print(f"  -> {fname}")
    return saved_files


def plot_single(data, metric, out_path, fmt="pdf", stem_suffix: str = ""):
    """Single-column figure for one metric."""
    bs = data["batch_sizes"]
    sigmas = data["noise_on_fc"]

    key_naive_m = f"{metric}_naive_mean"
    key_naive_s = f"{metric}_naive_std"
    key_feat_m = f"{metric}_feat_mean"
    key_feat_s = f"{metric}_feat_std"

    ylabel = {"psnr": "PSNR (dB)", "ssim": "SSIM", "fid": "FID"}[metric]

    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4), constrained_layout=True)

    _add_curve(ax, bs,
               data[key_naive_m], data[key_naive_s],
               "Naive GIA", COLORS["naive"], MARKERS[0], zorder=3)
    if f"{metric}_hlcp_mean" in data and f"{metric}_hlcp_std" in data:
        _add_line(ax, bs, data[f"{metric}_hlcp_mean"], "HLCP + GIA", COLORS["hlcp"], "X", zorder=4)
    for i, sigma in enumerate(sigmas):
        _add_curve(ax, bs,
                   data[key_feat_m][i], data[key_feat_s][i],
                   _label_sigma(sigma),
                   COLORS["hssp"][i % len(COLORS["hssp"])],
                   MARKERS[(i + 1) % len(MARKERS)],
                   )

    _setup_ax(ax, "Batch size", ylabel)
    _legend_auto_place(ax)
    if metric == "ssim":
        ax.set_ylim(0.0, 1.0)

    saved_files = _save_figure(fig, out_path, f"{metric}_vs_batchsize{stem_suffix}", fmt)
    plt.close(fig)
    for fname in saved_files:
        print(f"  -> {fname}")
    return saved_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-plot GIA sweep metrics in TPAMI style from NPZ file."
    )
    parser.add_argument("npz", help="Path to metrics_sweep_summary.npz")
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: same as NPZ file)",
    )
    parser.add_argument(
        "--format", default="pdf", choices=["pdf", "png", "svg", "eps"],
        help="Output figure format (default: pdf)",
    )
    parser.add_argument(
        "--separate", action="store_true",
        help="Also save separate single-column figures for each metric",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        sys.exit(f"Error: file not found — {args.npz}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update(_STYLE)

    print(f"Loading {args.npz} ...")
    data = dict(np.load(args.npz))

    for k in ("psnr_feat_mean", "psnr_feat_std",
              "ssim_feat_mean", "ssim_feat_std",
              "fid_feat_mean", "fid_feat_std"):
        if k in data and data[k].ndim == 1:
            data[k] = data[k][np.newaxis, :]

    print(f"  batch_sizes : {data['batch_sizes']}")
    print(f"  sigma values: {data['noise_on_fc']}")
    print(f"  #points     : {len(data['batch_sizes'])}")

    has_hlcp = any(k.endswith("_hlcp_mean") for k in data.keys())
    stem_suffix = "_hlcp" if has_hlcp else ""

    plot_combined(data, out_dir, args.format, stem_suffix=stem_suffix)

    if args.separate:
        for metric in ("psnr", "ssim", "fid"):
            key = f"{metric}_naive_mean"
            if key in data:
                plot_single(data, metric, out_dir, args.format, stem_suffix=stem_suffix)
            else:
                print(f"  [skip] {metric} — not in NPZ")

    print("Done.")


if __name__ == "__main__":
    main()
