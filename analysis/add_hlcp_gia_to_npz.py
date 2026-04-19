#!/usr/bin/env python3
"""
Add "HLCP + GIA" mixed metrics into an existing GIA sweep NPZ.

HLCP + GIA is defined as a per-batch-size mixture of:
  - HSSP + GIA (i.e., the HSSP curve with sigma == 0)
  - Naive GIA

The mixture weight w(n) comes from:
  logs/smooth_success_rate_n1_to_128.txt
where each line is:
  n <tab> success_rate
and means "w(n) is the HSSP + GIA weight".

The resulting NPZ adds these keys (when present in input):
  psnr_hlcp_mean, psnr_hlcp_std
  ssim_hlcp_mean, ssim_hlcp_std
  fid_hlcp_mean, fid_hlcp_std  (only if fid_* exist)

Optionally runs `plot_metrics.py` to generate the beautified figures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict

import numpy as np


def _load_success_weights_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#", dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected success-rate txt format: {path}")
    n = arr[:, 0].astype(np.int64)
    w = arr[:, 1].astype(np.float64)
    return n, w


def _pick_sigma0_index(sigmas: np.ndarray, sigma0: float = 0.0, tol: float = 1e-8) -> int:
    sigmas = np.asarray(sigmas, dtype=np.float64)
    idx = int(np.argmin(np.abs(sigmas - sigma0)))
    if abs(sigmas[idx] - sigma0) > tol:
        raise ValueError(
            f"Cannot find sigma=={sigma0:g} in noise_on_fc; nearest is {sigmas[idx]:.12g} (idx={idx})"
        )
    return idx


def _mix_mean_std(
    w: np.ndarray, mean_h: np.ndarray, std_h: np.ndarray, mean_n: np.ndarray, std_n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mixture-of-distributions variance:
      Var = w*(std_h^2 + (mean_h - mean_mix)^2) + (1-w)*(std_n^2 + (mean_n - mean_mix)^2)
    """
    w = np.asarray(w, dtype=np.float64)
    mean_h = np.asarray(mean_h, dtype=np.float64)
    std_h = np.asarray(std_h, dtype=np.float64)
    mean_n = np.asarray(mean_n, dtype=np.float64)
    std_n = np.asarray(std_n, dtype=np.float64)

    mean_mix = w * mean_h + (1.0 - w) * mean_n
    var_mix = w * (std_h**2 + (mean_h - mean_mix) ** 2) + (1.0 - w) * (
        std_n**2 + (mean_n - mean_mix) ** 2
    )
    std_mix = np.sqrt(np.maximum(var_mix, 0.0))
    return mean_mix, std_mix


def _compute_hlcp_arrays(data: Dict[str, np.ndarray], weights_n: np.ndarray, weights_w: np.ndarray) -> Dict[str, np.ndarray]:
    bs = np.asarray(data["batch_sizes"], dtype=np.float64)
    w_bs = np.interp(bs, weights_n.astype(np.float64), weights_w.astype(np.float64))

    sigmas = np.asarray(data["noise_on_fc"], dtype=np.float64)
    idx_sigma0 = _pick_sigma0_index(sigmas, sigma0=0.0)

    out: Dict[str, np.ndarray] = {"hlcp_weight": w_bs.astype(np.float64)}

    # PSNR
    psnr_h_mean = np.asarray(data["psnr_feat_mean"][idx_sigma0], dtype=np.float64)
    psnr_h_std = np.asarray(data["psnr_feat_std"][idx_sigma0], dtype=np.float64)
    psnr_n_mean = np.asarray(data["psnr_naive_mean"], dtype=np.float64)
    psnr_n_std = np.asarray(data["psnr_naive_std"], dtype=np.float64)
    psnr_hlcp_mean, psnr_hlcp_std = _mix_mean_std(w_bs, psnr_h_mean, psnr_h_std, psnr_n_mean, psnr_n_std)
    out["psnr_hlcp_mean"] = psnr_hlcp_mean
    out["psnr_hlcp_std"] = psnr_hlcp_std

    # SSIM
    ssim_h_mean = np.asarray(data["ssim_feat_mean"][idx_sigma0], dtype=np.float64)
    ssim_h_std = np.asarray(data["ssim_feat_std"][idx_sigma0], dtype=np.float64)
    ssim_n_mean = np.asarray(data["ssim_naive_mean"], dtype=np.float64)
    ssim_n_std = np.asarray(data["ssim_naive_std"], dtype=np.float64)
    ssim_hlcp_mean, ssim_hlcp_std = _mix_mean_std(w_bs, ssim_h_mean, ssim_h_std, ssim_n_mean, ssim_n_std)
    out["ssim_hlcp_mean"] = ssim_hlcp_mean
    out["ssim_hlcp_std"] = ssim_hlcp_std

    # FID (optional)
    if "fid_naive_mean" in data and "fid_feat_mean" in data:
        fid_h_mean = np.asarray(data["fid_feat_mean"][idx_sigma0], dtype=np.float64)
        fid_h_std = np.asarray(data["fid_feat_std"][idx_sigma0], dtype=np.float64)
        fid_n_mean = np.asarray(data["fid_naive_mean"], dtype=np.float64)
        fid_n_std = np.asarray(data["fid_naive_std"], dtype=np.float64)
        fid_hlcp_mean, fid_hlcp_std = _mix_mean_std(w_bs, fid_h_mean, fid_h_std, fid_n_mean, fid_n_std)
        out["fid_hlcp_mean"] = fid_hlcp_mean
        out["fid_hlcp_std"] = fid_hlcp_std

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add HLCP + GIA mixed metrics into metrics_sweep_summary.npz"
    )
    parser.add_argument(
        "sweep_dir",
        help="Sweep folder containing metrics_sweep_summary.npz (e.g. expdata/gia_sweep/.../pool1)",
    )
    parser.add_argument(
        "--success_txt",
        "--weight_curve",
        default=None,
        help=(
            "Path to a curve txt (n<TAB>w) used as HLCP weight w(n). "
            "Alias: --weight_curve. "
            "Default: repo_root/logs/smooth_success_rate_n1_to_128.txt"
        ),
    )
    parser.add_argument(
        "--out_npz",
        default=None,
        help="Output NPZ path (default: <sweep_dir>/metrics_sweep_summary_hlcp.npz)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip running plot_metrics.py after writing NPZ",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output figure format for plot_metrics.py (default: pdf)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also save separate single-column figures for each metric",
    )
    args = parser.parse_args()

    sweep_dir = os.path.abspath(args.sweep_dir)
    npz_path = os.path.join(sweep_dir, "metrics_sweep_summary.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"metrics_sweep_summary.npz not found: {npz_path}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if args.success_txt is None:
        success_txt = os.path.join(repo_root, "logs", "smooth_success_rate_n1_to_128.txt")
    else:
        success_txt = os.path.abspath(args.success_txt)
    if not os.path.isfile(success_txt):
        raise FileNotFoundError(f"success-rate txt not found: {success_txt}")

    out_npz = args.out_npz or os.path.join(sweep_dir, "metrics_sweep_summary_hlcp.npz")

    print(f"Loading NPZ: {npz_path}")
    with np.load(npz_path) as z:
        data = {k: z[k] for k in z.files}

    weights_n, weights_w = _load_success_weights_txt(success_txt)
    print(f"Loaded success weights: n=[{weights_n.min()}..{weights_n.max()}]")

    hlcp_arrays = _compute_hlcp_arrays(data, weights_n, weights_w)

    # Save: keep original arrays, just add HLCP keys.
    to_save: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        to_save[k] = v
    for k, v in hlcp_arrays.items():
        to_save[k] = v

    np.savez(out_npz, **to_save)
    print(f"Wrote: {out_npz}")

    if not args.no_plot:
        plot_script = os.path.join(repo_root, "plot_metrics.py")
        if not os.path.isfile(plot_script):
            raise FileNotFoundError(f"plot_metrics.py not found: {plot_script}")
        out_dir = os.path.dirname(os.path.abspath(out_npz))
        cmd = [sys.executable, plot_script, out_npz, "--out_dir", out_dir, "--format", args.format]
        if args.separate:
            cmd.append("--separate")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

