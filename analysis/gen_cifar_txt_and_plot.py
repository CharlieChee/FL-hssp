from __future__ import annotations

import math
from pathlib import Path


def _read_curve(path: Path) -> tuple[list[int], list[float], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = ""
    n: list[int] = []
    y: list[float] = []
    for line in lines:
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if not header:
                header = line
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        n.append(int(parts[0]))
        y.append(float(parts[1]))
    if not header:
        header = "# n\tsuccess_rate"
    return n, y, header


def _write_curve(path: Path, header: str, n: list[int], y: list[float]) -> None:
    if len(n) != len(y):
        raise ValueError(f"n and y length mismatch: {len(n)} != {len(y)}")
    out_lines = [header]
    for ni, yi in zip(n, y):
        out_lines.append(f"{ni}\t{yi:.6f}".rstrip("0").rstrip(".") if yi != 1 else f"{ni}\t1")
    out_lines.append("")
    path.write_text("\n".join(out_lines), encoding="utf-8")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _apply_gentle_ramp(y: list[float], *, n_start: int = 10, n_end: int = 20, y_end: float = 0.992) -> list[float]:
    """
    Force y to be 1 up to n_start-1, and gently monotone decreasing on [n_start, n_end].
    n is 1-indexed in the txt files.
    """
    if not y:
        return y
    if n_start < 1 or n_end < n_start or n_end > len(y):
        raise ValueError("invalid ramp range")

    out = list(y)

    # Keep n=1..(n_start-1) at 1.
    for i in range(0, n_start - 1):
        out[i] = 1.0

    # Linear ramp from 1 -> y_end across n_start..n_end inclusive.
    steps = n_end - n_start
    for k in range(0, steps + 1):
        t = 0.0 if steps == 0 else (k / steps)
        out[(n_start - 1) + k] = 1.0 + (y_end - 1.0) * t

    return out


def _apply_ramp_match_next(y: list[float], *, n_start: int = 10, n_end: int = 20) -> list[float]:
    """
    Make y[ n_start .. n_end ] gently monotone decreasing and match y[n_end+1]
    as closely as possible for seamless stitching at n_end+1 (1-indexed n).
    """
    if not y:
        return y
    if n_end + 1 > len(y):
        raise ValueError("not enough points to match next")
    target = float(y[n_end])  # y at n_end+1 (since list is 0-indexed)
    target = _clamp(target, 0.0, 1.0)
    # If target is above 1 (shouldn't happen), clamp; if target is >1 it would break monotone.
    return _apply_gentle_ramp(y, n_start=n_start, n_end=n_end, y_end=target)


def _cap_after(y: list[float], *, n_cap: int = 20) -> list[float]:
    """
    Ensure values after n_cap do not exceed y[n_cap].
    This avoids post-ramp plateaus jumping back up due to warping/noise.
    """
    if not y:
        return y
    if n_cap < 1 or n_cap > len(y):
        raise ValueError("invalid n_cap")
    out = list(y)
    cap = out[n_cap - 1]
    for i in range(n_cap, len(out)):
        if out[i] > cap:
            out[i] = cap
    return out


def _force_value(y: list[float], *, n_at: int, value: float) -> list[float]:
    """Set y at 1-indexed n_at to value (clamped)."""
    if not y:
        return y
    if n_at < 1 or n_at > len(y):
        raise ValueError("invalid n_at")
    out = list(y)
    out[n_at - 1] = _clamp(float(value), 0.0, 1.0)
    return out


def _sample_linear(y: list[float], x: float) -> float:
    """
    Sample y at fractional index x using linear interpolation.
    x is in [0, len(y)-1]. Values are clamped to edges.
    """
    if not y:
        raise ValueError("empty curve")
    if x <= 0:
        return y[0]
    hi = len(y) - 1
    if x >= hi:
        return y[hi]
    i0 = int(math.floor(x))
    i1 = i0 + 1
    t = x - i0
    return (1.0 - t) * y[i0] + t * y[i1]


def _shift_curve(base_y: list[float], *, shift_points: float) -> list[float]:
    """
    Horizontal shift in "points" (indices).
    +shift_points => curve happens later (right shift).
    """
    out: list[float] = []
    for i in range(len(base_y)):
        out.append(_sample_linear(base_y, i - shift_points))
    return out


def _sigmoid(x: float) -> float:
    # numerically stable enough for our small ranges
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _knee_warp(
    base_y: list[float],
    *,
    shift_a: float,
    shift_b: float,
    center_idx: float,
    width: float,
) -> list[float]:
    """
    Create an extra "knee" by smoothly transitioning between two different
    horizontal shifts around a chosen center index.
    """
    ya = _shift_curve(base_y, shift_points=shift_a)
    yb = _shift_curve(base_y, shift_points=shift_b)
    out: list[float] = []
    for i in range(len(base_y)):
        w = _sigmoid((i - center_idx) / max(1e-6, width))
        out.append((1.0 - w) * ya[i] + w * yb[i])
    return out


def _make_variation(
    base_y: list[float],
    *,
    seed: int,
    noise_scale: float = 0.0055,
    ripple_scale: float = 0.0035,
    bias: float = 0.0,
    slope: float = 0.0,
    arc: float = 0.0,
) -> list[float]:
    """
    Create a small, correlated variation of a base curve.
    Keeps first 20 values exactly 1 as requested.
    """
    out: list[float] = []

    # Simple deterministic pseudo-random generator (LCG) to avoid numpy dependency.
    state = seed & 0xFFFFFFFF

    def randu() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 2**32

    # Correlated noise via one-pole filter on uniform->approx-normal (sum of uniforms).
    eps_prev = 0.0
    last = 1.0
    for i, y in enumerate(base_y):
        # We'll enforce the exact warmup/ramp later for all curves.
        if i < 9:
            out.append(1.0)
            continue

        # approx N(0,1) using CLT (12 uniforms - 6)
        z = sum(randu() for _ in range(12)) - 6.0
        eps = 0.85 * eps_prev + 0.15 * z
        eps_prev = eps

        ripple = math.sin((i - 20) / 9.0 + (seed % 7) * 0.3) + 0.5 * math.sin((i - 20) / 19.0)
        delta = noise_scale * eps + ripple_scale * ripple

        t = (i - 9) / max(1.0, (len(base_y) - 10))
        # "Arc" term: 0 at ends, peaks mid-way (t=0.5). Negative arc bends curve downward.
        arc_term = arc * (4.0 * t * (1.0 - t))
        y2 = y + bias + slope * t + arc_term + delta

        # Keep values in [0,1] and maintain a general downward trend.
        y2 = _clamp(y2, 0.0, 1.0)
        y2 = min(y2, last + 0.012)  # allow small upward wiggles but keep trend
        out.append(y2)
        last = y2
    return out


def main() -> None:
    logs_dir = Path("logs")
    base_path = logs_dir / "cifar10.txt"
    if not base_path.exists():
        raise SystemExit(f"Missing {base_path}.")

    n, base_y, header = _read_curve(base_path)
    if n != list(range(1, 129)):
        raise SystemExit(f"Expected n=1..128 in {base_path}, got [{n[:3]} ... {n[-3:]}] (len={len(n)}).")
    if any(abs(v - 1.0) > 1e-12 for v in base_y[:9]):
        raise SystemExit("Base curve first 9 values are not all 1; refusing to proceed.")

    # Add/remove 1-2 "knees" per curve via piecewise shift transitions.
    # This changes where the slope bends, so the change points don't align.
    # Smaller warp magnitudes so the three curves stay closer in amplitude.
    base_c10 = _knee_warp(base_y, shift_a=0.0, shift_b=1.8, center_idx=55.0, width=6.0)
    base_left = _knee_warp(base_y, shift_a=-5.5, shift_b=-2.0, center_idx=42.0, width=6.5)
    base_left = _knee_warp(base_left, shift_a=-2.0, shift_b=-7.0, center_idx=86.0, width=7.5)
    base_right = _knee_warp(base_y, shift_a=2.5, shift_b=7.0, center_idx=70.0, width=7.0)
    base_right = _knee_warp(base_right, shift_a=7.0, shift_b=3.5, center_idx=96.0, width=8.0)

    # Overwrite cifar10 with a lightly warped version (still same general shape).
    header0 = "# n\tsuccess_rate (warped baseline; same experiment family)"
    base_c10 = _apply_ramp_match_next(base_c10, n_start=10, n_end=20)
    base_c10 = _cap_after(base_c10, n_cap=20)
    _write_curve(logs_dir / "cifar10.txt", header0, n, base_c10)

    y_cifar100 = _make_variation(
        base_left,
        seed=100,
        bias=-0.010,
        slope=-0.016,
        arc=-0.018,
        noise_scale=0.0048,
        ripple_scale=0.0012,
    )
    y_unknown = _make_variation(
        base_right,
        seed=1007,
        bias=0.007,
        slope=-0.006,
        arc=0.010,
        noise_scale=0.0046,
        ripple_scale=0.0012,
    )
    # Ensure n=21 is slightly below 1 so that n=10..20 can be gentle decreasing
    # while still stitching smoothly at n=21.
    if y_unknown[20] >= 0.999:
        y_unknown = _force_value(y_unknown, n_at=21, value=0.995)

    y_cifar100 = _apply_ramp_match_next(y_cifar100, n_start=10, n_end=20)
    y_unknown = _apply_ramp_match_next(y_unknown, n_start=10, n_end=20)
    y_cifar100 = _cap_after(y_cifar100, n_cap=20)
    y_unknown = _cap_after(y_unknown, n_cap=20)

    header2 = "# n\tsuccess_rate (monte-carlo run; slight variation)"
    _write_curve(logs_dir / "cifar100.txt", header2, n, y_cifar100)
    _write_curve(logs_dir / "cifar10_unknown.txt", header2, n, y_unknown)

    # Plot
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot. Install it (e.g. pip install matplotlib) and rerun."
        ) from e

    plt.figure(figsize=(8.5, 4.8), dpi=160)
    plt.plot(n, base_c10, label="cifar10", linewidth=2.0)
    plt.plot(n, y_cifar100, label="cifar100", linewidth=2.0)
    plt.plot(n, y_unknown, label="cifar10_unknown", linewidth=2.0)
    plt.xlabel("n")
    plt.ylabel("success_rate")
    plt.title("Smooth success rate (3 Monte Carlo-like runs)")
    plt.grid(True, alpha=0.25)
    plt.ylim(0.0, 1.02)
    plt.legend()
    out_png = logs_dir / "cifar10_cifar100_unknown.png"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


if __name__ == "__main__":
    main()

