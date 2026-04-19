#!/usr/bin/env python3
"""Parse logs/nfound.log and plot max success_rate per Batch_size(n)."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def parse_max_success_by_batch(log_path: Path) -> Tuple[List[int], List[float]]:
    current_n: Optional[int] = None
    max_rate: dict[int, float] = {}

    header_re = re.compile(r"^===== Batch_size\(n\)=(\d+) =====$")
    sr_re = re.compile(r"success_rate=([\d.]+)")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        hm = header_re.match(line)
        if hm:
            current_n = int(hm.group(1))
            continue
        sm = sr_re.search(line)
        if sm is not None and current_n is not None:
            v = float(sm.group(1))
            prev = max_rate.get(current_n)
            max_rate[current_n] = v if prev is None else max(prev, v)

    ns = sorted(max_rate.keys())
    ys = [max_rate[n] for n in ns]
    return ns, ys


def suffix_max_non_increasing(ys: List[float]) -> List[float]:
    """For each i, y'[i] = max(ys[i:]); then y'[i] >= y'[j] for all j > i (non-increasing in n)."""
    if not ys:
        return []
    out = ys[:]
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1])
    return out


def _enforce_nonincreasing(y: np.ndarray) -> None:
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            y[i] = y[i - 1]


def monotone_smooth_uniform(
    xs: List[float],
    ys: List[float],
    n_points: int = 1200,
    gaussian_sigma: float = 88.0,
    post_smooth_sigma: float = 14.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Heavy Gaussian smoothing to kill stair-steps; light post-blur + re-clamp keeps curve round."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 2:
        return x, y
    x_f = np.linspace(float(x[0]), float(x[-1]), int(n_points))
    y_f = np.interp(x_f, x, y)
    y_f = gaussian_filter1d(y_f, sigma=float(gaussian_sigma), mode="reflect")
    y_f = gaussian_filter1d(y_f, sigma=float(gaussian_sigma) * 0.42, mode="reflect")
    y_f = np.clip(y_f, 0.0, 1.0)
    _enforce_nonincreasing(y_f)
    if post_smooth_sigma > 0:
        y_f = gaussian_filter1d(y_f, sigma=float(post_smooth_sigma), mode="reflect")
        y_f = np.clip(y_f, 0.0, 1.0)
        _enforce_nonincreasing(y_f)
    return x_f, y_f


def write_smooth_rates_txt(
    n_fine: np.ndarray,
    y_fine: np.ndarray,
    path: Path,
    n_max: int = 128,
) -> None:
    """Sample the smooth curve at n=1..n_max; left/right extrapolation uses curve endpoints."""
    left_v = float(y_fine[0])
    right_v = float(y_fine[-1])
    lines = ["# n\tsuccess_rate (monotone smoothed curve; np.interp on n_fine,y_fine)\n"]
    for n in range(1, n_max + 1):
        r = float(np.interp(n, n_fine, y_fine, left=left_v, right=right_v))
        lines.append("%d\t%.6f\n" % (n, r))
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    log_path = root / "logs" / "nfound.log"
    out_path = root / "logs" / "nfound_max_success_rate.png"
    rates_txt = root / "logs" / "smooth_success_rate_n1_to_128.txt"

    ns, ys_raw = parse_max_success_by_batch(log_path)
    ys = suffix_max_non_increasing(ys_raw)
    n_fine, y_fine = monotone_smooth_uniform([float(v) for v in ns], ys)
    write_smooth_rates_txt(n_fine, y_fine, rates_txt)

    plt.figure(figsize=(9, 5))
    plt.plot(
        ns,
        ys_raw,
        marker="s",
        linewidth=1,
        markersize=4,
        color="#aaaaaa",
        linestyle="--",
        label="per-n max (raw)",
        alpha=0.85,
    )
    plt.plot(
        n_fine,
        y_fine,
        linewidth=2.2,
        color="#1f77b4",
        label="monotone smooth (uniform/Gaussian)",
    )
    plt.plot(
        ns,
        ys,
        marker="o",
        linestyle="none",
        markersize=4,
        color="#0d47a1",
        alpha=0.9,
        label="suffix-max knots",
    )
    plt.xlabel("Batch size (n)")
    plt.ylabel("Success rate")
    plt.title(
        "Max success rate vs batch size — monotone decreasing, smoothed (nfound.log)"
    )
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(min(ns) - 2, max(ns) + 2)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("Wrote", out_path)
    print("Wrote", rates_txt)
    for n, a, b in zip(ns, ys_raw, ys):
        print(f"  n={n:3d}  raw_max={a:.4f}  monotone={b:.4f}")


if __name__ == "__main__":
    main()
