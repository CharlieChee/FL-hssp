#!/usr/bin/env sage
"""
Benchmark: HSSP / HLCP attack time vs hidden dimension n (m=3n+10, fixed seeds).
Runs one random instance per (B_X, n) combination so the numbers are directly
comparable across machines.

Usage (from the repository root)::

  sage experiments/bench_time_m3n10.sage
"""
import os, sys

# Self-locate: chdir into lattice/ so nested load() calls (hssp.sage, etc.) work,
# then chdir back to the repo root.
_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
_REPO = os.path.dirname(_d) if os.path.basename(_d) == "experiments" else _d
while _REPO and _REPO != "/" and not os.path.isfile(os.path.join(_REPO, "lattice", "multi_dim_hssp.sage")):
    _REPO = os.path.dirname(_REPO)
if not os.path.isfile(os.path.join(_REPO, "lattice", "multi_dim_hssp.sage")):
    raise FileNotFoundError("Could not locate lattice/multi_dim_hssp.sage; run from the repository root.")
_LATTICE = os.path.join(_REPO, "lattice")
os.chdir(_LATTICE)
load("multi_dim_hssp.sage")
os.chdir(_REPO)

import time, csv, platform

ns = [10, 20, 30, 40, 50, 60, 70]
configs = [
    ("HSSP",      1),
    ("HLCP_100",  100),
    ("HLCP_1000", 1000),
]

SEED_BASE = 20260411
B_A = 100
NX0_BITS = 192

hostname = platform.node()
cpu_info = platform.processor() or "unknown"
print(f"Host: {hostname}  CPU: {cpu_info}")
print(f"Settings: m=3n+10, B_A={B_A}, nx0_bits={NX0_BITS}, seed_base={SEED_BASE}")
print("=" * 70)

results = []

for config_name, B_X in configs:
    print(f"\n{'=' * 70}")
    print(f"  {config_name} (B_X={B_X})")
    print(f"{'=' * 70}")

    for n in ns:
        m = 3 * n + 10
        l = n
        seed = SEED_BASE + n * 1000 + B_X
        set_random_seed(seed)

        H = multi_hssp(n, l, B_X=B_X, B_A=B_A, nx0_bits=NX0_BITS)
        H.gen_instance(m)

        t0 = time.time()
        MB = hssp_attack(H, alg='ns')
        elapsed = time.time() - t0

        nfound = MB.nrows() if hasattr(MB, 'nrows') else 0
        print(f"  n={n:3d}  m={m:3d}  B_X={B_X:4d}  time={elapsed:8.1f}s  nfound={nfound}")
        results.append((config_name, B_X, n, m, elapsed, int(nfound)))
        sys.stdout.flush()

# Save CSV
out_csv = f"bench_time_m3n10_{hostname}.csv"
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["config", "B_X", "n", "m", "time_s", "nfound", "host"])
    for r in results:
        w.writerow(list(r) + [hostname])
print(f"\nSaved: {out_csv}")

# Print summary table
print(f"\n{'=' * 70}")
print(f"Summary ({hostname})")
print(f"{'=' * 70}")
print(f"{'n':>4s} {'m':>5s} | {'HSSP':>8s} | {'HLCP_100':>8s} | {'HLCP_1000':>9s}")
print("-" * 45)
for i, n in enumerate(ns):
    m = 3*n+10
    t_hssp = results[i][4]
    t_100  = results[len(ns)+i][4]
    t_1000 = results[2*len(ns)+i][4]
    print(f"{n:4d} {m:5d} | {t_hssp:8.1f} | {t_100:8.1f} | {t_1000:9.1f}")
