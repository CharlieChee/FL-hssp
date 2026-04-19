#!/usr/bin/env sage
"""
Benchmark with REAL CNN data.
Loads pre-ReLU npy files, quantizes with INT_SCALE=1000, selects rows with a
fixed seed, and runs the NS attack. Produces identical instances on any machine.

Usage (from the repository root)::

  sage experiments/bench_real_cnn_v2.sage

Prerequisite: the npy files under ``expdata/fc_hssp/cifar10/bs<n>/`` must exist,
e.g. via::

  for n in 10 20 30 40 50 60 70; do
      python src/cnn.py --mode fc_hssp --batch_size $n --dataset cifar10
  done
"""
import os, sys

# Self-locate: chdir into lattice/ so nested load() calls (hssp.sage, etc.) work,
# then chdir back to the repo root so expdata/... resolves.
_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
_REPO = os.path.dirname(_d) if os.path.basename(_d) == "experiments" else _d
# Fallback: walk up until we find lattice/multi_dim_hssp.sage
while _REPO and _REPO != "/" and not os.path.isfile(os.path.join(_REPO, "lattice", "multi_dim_hssp.sage")):
    _REPO = os.path.dirname(_REPO)
if not os.path.isfile(os.path.join(_REPO, "lattice", "multi_dim_hssp.sage")):
    raise FileNotFoundError("Could not locate lattice/multi_dim_hssp.sage; run from the repository root.")
_LATTICE = os.path.join(_REPO, "lattice")
os.chdir(_LATTICE)
load("multi_dim_hssp.sage")
os.chdir(_REPO)

import time, numpy as np, platform, csv

hostname = platform.node()
print(f"Host: {hostname}")
print(f"Sage: {version()}")

INT_SCALE = 1000
B_A = 100
NX0_BITS = 192
SEED = 42  # Fixed seed for row selection and A generation

results = []

for n in [10, 20, 30, 40, 50, 60, 70]:
    m = 3 * n + 10
    l = n

    # Load real CNN pre-relu data
    npy_path = f"expdata/fc_hssp/cifar10/bs{n}/fc1_pre_relu_fc.npy"
    pre = np.load(npy_path).astype(np.float64)
    print(f"\nn={n} m={m}: loaded {npy_path} shape={pre.shape}")

    # Quantize: X = floor(pre * INT_SCALE)
    X_pool = np.floor(pre * INT_SCALE).astype(np.int64)

    # Select rows: fixed seed, random + full rank
    rng = np.random.RandomState(SEED + n)
    num_rows = X_pool.shape[0]
    perm = rng.permutation(num_rows)

    # Greedy rank selection
    selected = []
    rank_so_far = 0
    for idx in perm:
        candidate = list(selected) + [idx]
        test_mat = X_pool[candidate, :]
        r = np.linalg.matrix_rank(test_mat)
        if r > rank_so_far:
            selected.append(idx)
            rank_so_far = r
        if rank_so_far == n:
            break
    # Fill remaining
    remaining = [i for i in perm if i not in selected]
    while len(selected) < m:
        selected.append(remaining.pop(0))
    sel = np.array(selected[:m])

    X_sel = X_pool[sel, :]
    B_X = int(np.max(np.abs(X_sel)))
    print(f"  X: {X_sel.shape}, B_X={B_X}, rank={np.linalg.matrix_rank(X_sel)}")

    # Create instance
    X_sage = matrix(ZZ, X_sel.tolist())

    # recover_box_max: same formula as run_mhlcp_cnn_quant_x.sage
    rbox_base = 220000
    recover_box_max = max(rbox_base, int(800 * n))
    if B_X > 10:
        recover_box_max = max(recover_box_max, int(B_X * n * 100))

    H = multi_hssp(n, l, B_X=B_X, B_A=B_A, nx0_bits=NX0_BITS,
                   recover_box_max=recover_box_max)

    # Generate A with fixed seed
    set_random_seed(SEED + n + 10000)
    H.gen_instance_fixed_X(X_sage)

    print(f"  B_X={B_X}, recover_box_max={recover_box_max}")

    # Run attack
    t0 = time.time()
    MB = hssp_attack(H, alg='ns')
    elapsed = time.time() - t0

    nfound = MB.nrows() if hasattr(MB, 'nrows') else 0
    print(f"  time={elapsed:.1f}s  nfound={nfound}/{n}")
    results.append((n, m, B_X, recover_box_max, elapsed, int(nfound)))
    sys.stdout.flush()

# Summary
print(f"\n{'='*60}")
print(f"Summary ({hostname}, INT_SCALE={INT_SCALE}, m=3n+10)")
print(f"{'='*60}")
print(f"{'n':>4s} {'m':>5s} {'B_X':>5s} {'rbox_max':>10s} {'time':>8s} {'nfound':>8s}")
for n, m, bx, rbox, t, nf in results:
    print(f"{n:4d} {m:5d} {bx:5d} {rbox:10d} {t:8.1f} {nf:8d}")

# Save CSV
csv_path = f"bench_real_cnn_v2_{hostname}.csv"
with open(csv_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(["n", "m", "B_X", "recover_box_max", "time_s", "nfound", "host"])
    for r in results:
        w.writerow([int(x) if isinstance(x, (int, Integer)) else x for x in r] + [hostname])
print(f"\nSaved: {csv_path}")
