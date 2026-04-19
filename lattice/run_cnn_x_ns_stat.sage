#!/usr/bin/env sage
r"""
Real X_sec exported from the CNN (n x n, fc-feature quantization) plus the same
A as in cnn.ipynb (m x n ReLU rows) is used to build an HLCP:
H.X = P (row-permuted A), secret a = column 0 of X_sec, b = P*a (mod x0).

Main flow (stat experiments):
  - Quantization defaults to floor(fc*1000) (override via CNN_FC_SCALES).
  - When building X prefer "few zeros": greedily pick n columns by the per-column
    nonzero count (while keeping full rank); also sample random full-rank column
    sets and keep the one with the most nonzeros. This primary X is used to
    search for a row permutation (falling back to pivot X if that fails); with
    the permutation fixed we then run statistical on each column set.
  - Column sets include: low_zero_greedy, low_zero_random_best, pivot, random
    columns, low-correlation columns, etc.; print each set's X and metrics
    (including zero_fraction).
  - NS (if not skipped) uses the same columns as the primary X.

Tunables: environment variables CNN_STAT_LIGHT=1 (fewer column trials, skip NS)
and CNN_STAT_SKIP_NS=1. CNN_FC_SCALES: comma-separated quantization factors,
default "1000". NUM_RANDOM, NUM_GREEDY_POOL inside main() (non-LIGHT only).

Dependencies: fc_hssp npy files under expdata; sklearn.

About Step1(..., BKZ=...): in this repo's ortho_attack.sage the flag is NOT the
NTL BKZ lattice-basis reduction. BKZ=True means MO is built over ZZ; BKZ=False
puts MO over GF(3), which cannot be fed to MO.LLL() inside statistical_1. The
statistical attack path must therefore pass BKZ=True (meaning "integer MO").
"""
import os
import sys
import math
import signal
import subprocess
import shutil
from io import StringIO
from time import time as wall_time

# See above: must be True so that the statistical second step can run integer LLL on MO
STEP1_INTEGER_MO = True

import numpy as np
from sage.misc.prandom import shuffle


def _find_repo_root():
    """Locate the repo root (contains ``lattice/multi_dim_hssp.sage``)."""
    seeds = []
    if sys.argv and sys.argv[0]:
        seeds.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    seeds.append(os.path.abspath(os.getcwd()))
    for s in seeds:
        cur = s
        for _ in range(8):
            if os.path.isfile(os.path.join(cur, "lattice", "multi_dim_hssp.sage")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return os.path.abspath(os.getcwd())


_REPO = _find_repo_root()
os.makedirs(os.path.join(_REPO, "ICA"), exist_ok=True)

try:
    from sklearn.decomposition import FastICA
except ImportError as e:
    raise RuntimeError("requires sklearn: pip install scikit-learn") from e

# Sage's load() chain needs cwd=lattice/ for nested loads to resolve;
# we chdir back to repo root afterwards so subsequent expdata/ I/O works.
_LATTICE = os.path.join(_REPO, "lattice")
os.chdir(_LATTICE)
load("hlcp.sage")
load("statistical.sage")
os.chdir(_REPO)

globals()["FastICA"] = FastICA
globals()["math"] = math
globals()["time"] = wall_time


def ensure_fc_hssp_npys(batch_size=10, dataset="cifar10"):
    """If no npy exists at the standard path, invoke local python3 to run src/cnn.py --mode fc_hssp (requires torch and the dataset)."""
    relu = os.path.join("expdata", "fc_hssp", dataset, "bs%s" % batch_size, "relu_mask_fc.npy")
    fc = os.path.join("expdata", "fc_hssp", dataset, "bs%s" % batch_size, "fc_input_batch.npy")
    if os.path.isfile(relu) and os.path.isfile(fc):
        return
    py = shutil.which("python3") or shutil.which("python")
    if py is None:
        raise FileNotFoundError(
            "did not find %s / %s and no python3 in PATH. First run: python3 src/cnn.py --mode fc_hssp --dataset %s --batch_size %s"
            % (relu, fc, dataset, batch_size)
        )
    root = os.getcwd()
    cmd = [
        py,
        os.path.join(root, "src", "cnn.py"),
        "--mode",
        "fc_hssp",
        "--dataset",
        dataset,
        "--batch_size",
        str(batch_size),
    ]
    print("generating FC HSSP data:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=root)
    if r.returncode != 0:
        raise RuntimeError("src/cnn.py fc_hssp failed, exit=%s" % r.returncode)
    if not (os.path.isfile(relu) and os.path.isfile(fc)):
        raise FileNotFoundError("src/cnn.py finished but %s / %s still missing" % (relu, fc))


def parse_fc_scales():
    raw = os.environ.get("CNN_FC_SCALES", "1000").strip()
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else [int(1000)]


def load_cnn_AX(batch_size=10, attack_m_rows=100, fc_scale=1000):
    relu_path = f"expdata/fc_hssp/relu_mask_fc_bs{batch_size}.npy"
    fc_path = f"expdata/fc_hssp/fc_input_batch_bs{batch_size}.npy"
    alt_relu = f"expdata/fc_hssp/cifar10/bs{batch_size}/relu_mask_fc.npy"
    alt_fc = f"expdata/fc_hssp/cifar10/bs{batch_size}/fc_input_batch.npy"
    alt_legacy_relu = "expdata/fc_hssp/cifar10/relu_mask_fc.npy"
    alt_legacy_fc = "expdata/fc_hssp/cifar10/fc_input_batch.npy"

    if os.path.exists(relu_path) and os.path.exists(fc_path):
        pass
    elif os.path.exists(alt_relu) and os.path.exists(alt_fc):
        relu_path, fc_path = alt_relu, alt_fc
    elif batch_size == 10 and os.path.exists(alt_legacy_relu) and os.path.exists(alt_legacy_fc):
        relu_path, fc_path = alt_legacy_relu, alt_legacy_fc
    else:
        raise FileNotFoundError(
            "relu_mask_fc / fc_input_batch not found. First run: python src/cnn.py --mode fc_hssp --batch_size 10"
        )

    relu_mask = np.load(relu_path)
    fc_input = np.load(fc_path).astype(float)
    relu_int = relu_mask.astype(int)
    num_rows, n = relu_int.shape
    fc_dim = int(fc_input.shape[1])
    target_rows = min(attack_m_rows, num_rows)

    ones_counts = relu_int.sum(axis=1)
    balanced_indices = [i for i, c in enumerate(ones_counts) if c in (4, 5, 6)]
    secondary_indices = [i for i, c in enumerate(ones_counts) if c in (3, 7)]
    other_indices = [i for i in range(num_rows) if i not in balanced_indices and i not in secondary_indices]
    np.random.shuffle(balanced_indices)
    np.random.shuffle(secondary_indices)
    np.random.shuffle(other_indices)
    selected = []
    current_matrix = None

    def fill_from(pool):
        nonlocal selected, current_matrix
        for idx in pool:
            if idx in selected:
                continue
            row = relu_int[idx : idx + 1, :]
            if len(selected) < n:
                if current_matrix is None:
                    current_matrix = row
                    selected.append(idx)
                else:
                    candidate = np.vstack([current_matrix, row])
                    if np.linalg.matrix_rank(candidate.astype(float)) > np.linalg.matrix_rank(
                        current_matrix.astype(float)
                    ):
                        current_matrix = candidate
                        selected.append(idx)
            else:
                selected.append(idx)
            if len(selected) == target_rows:
                break

    for pool in (balanced_indices, secondary_indices, other_indices):
        if len(selected) < target_rows:
            fill_from(pool)

    A_np = relu_int[selected, :]
    sf = float(fc_scale)
    scaled = np.floor(fc_input * sf).astype(np.int64)
    M_fc = matrix(ZZ, scaled.tolist())
    try:
        pivots = list(M_fc.pivots())
    except AttributeError:
        pivots = list(M_fc.echelon_form().pivots())
    pivot_cols = tuple(int(p) for p in pivots[:n])
    X_np = scaled[:, pivot_cols]
    return A_np, X_np, n, A_np.shape[0], scaled, fc_dim, pivot_cols


def build_cnn_instances(A_np, X_np, n):
    """Same CNN A, X_sec, and row permutation P; run Step1 with a large / small modulus to feed ns and statistical respectively."""
    A_base = matrix(ZZ, A_np.tolist())
    X_sec = matrix(ZZ, X_np.tolist())
    m = A_base.nrows()
    B_bound = max(ZZ(100), max(abs(ZZ(x)) for x in X_sec.list()))
    iota = 0.035
    nx0_big = int(2 * iota * n^2 + n * log(n, 2) + 2 * n * log(B_bound, 2))
    a = vector(ZZ, X_sec.column(0))
    m_stat = min(m, 10 * n)

    for attempt in range(8000):
        rows = list(range(m))
        shuffle(rows)
        P = matrix(ZZ, [list(A_base[i]) for i in rows])
        if P.rank() < n:
            continue
        b_raw = P * a
        x0_big = genpseudoprime(nx0_big)
        b_big = vector(ZZ, [Integer(mod(b_raw[i], x0_big)) for i in range(m)])
        if gcd(b_big[0], x0_big) != 1:
            continue
        MO_big, _, _, _ = Step1(n, -1, x0_big, a, P, b_big, m, BKZ=STEP1_INTEGER_MO)
        if MO_big == -1:
            continue

        mx = max(abs(Integer(x)) for x in b_raw)
        x0_sm = next_prime(Integer(mx + 1))
        b_sm_full = vector(ZZ, [Integer(mod(b_raw[i], x0_sm)) for i in range(m)])
        if gcd(b_sm_full[0], x0_sm) != 1:
            continue
        P_stat = P[:m_stat, :]
        b_stat = vector(ZZ, [b_sm_full[i] for i in range(m_stat)])
        MO_sm, _, _, _ = Step1(n, -1, x0_sm, a, P_stat, b_stat, m_stat, BKZ=STEP1_INTEGER_MO)
        if MO_sm == -1:
            continue

        H_ns = {
            "n": n,
            "m": m,
            "B": B_bound,
            "kappa": -1,
            "x0": x0_big,
            "a": a,
            "X": P,
            "b": b_big,
        }
        H_st = {
            "n": n,
            "m": m_stat,
            "B": B_bound,
            "kappa": -1,
            "x0": x0_sm,
            "a": a,
            "X": P_stat,
            "b": b_stat,
        }
        return H_ns, MO_big, H_st, MO_sm

    return None, None, None, None


class HLCPInst:
    pass


def H_to_inst(H):
    o = HLCPInst()
    for k, v in H.items():
        setattr(o, k, v)
    return o


class NS_Timeout(Exception):
    pass


def _ns_alarm(*_args):
    raise NS_Timeout("ns Step2 exceeded time limit (recoverBounded/BKZ is often very slow on CNN data)")


def run_ns(H, MO, timeout_sec=300):
    print("\n========== NS (B>1, kappa=-1) ==========")
    print("(will abort if Step2 exceeds %s seconds to avoid a long recoverBounded closure.)" % timeout_sec)
    Hi = H_to_inst(H)
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        signal.signal(signal.SIGALRM, _ns_alarm)
        signal.alarm(int(timeout_sec))
        try:
            beta, tt2, nrafound, textra = ns(Hi, MO, H["B"])
        finally:
            signal.alarm(0)
        sys.stdout = old
        print("ns returned: nrafound =", nrafound, "/", H["n"], " beta=", beta)
    except NS_Timeout as e:
        sys.stdout = old
        print("ns:", e)
    except Exception as e:
        sys.stdout = old
        print("ns exception:", type(e).__name__, e)


def run_stat(H, MO, B_ica=None):
    """B_ica: the B (scale) passed to ICA; default H['B']. Returns a dict containing nfound / nrafound."""
    print("\n========== Statistical ==========")
    B_use = H["B"] if B_ica is None else B_ica
    print("  ICA parameter B =", B_use, "(H['B']=", H["B"], ")")
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        MOn, MO_lll = statistical_1(
            MO, H["n"], H["m"], H["x0"], H["X"], H["a"], H["b"], H["kappa"], B_use
        )
        tica, tt2, nrafound, nfound = statistical_2(
            MOn, MO_lll, H["n"], H["m"], H["x0"], H["X"], H["a"], H["b"], H["kappa"], B_use
        )
        buf = sys.stdout.getvalue()
        sys.stdout = old
        print(buf, end="")
        print("statistical returned: NFound=%s nrafound=%s / %s" % (nfound, nrafound, H["n"]))
        return {
            "nfound": int(nfound),
            "nrafound": int(nrafound),
            "tica": tica,
            "tt2": tt2,
            "ok": True,
            "log": buf,
        }
    except Exception as e:
        buf = sys.stdout.getvalue()
        sys.stdout = old
        if buf:
            print(buf, end="")
        print("statistical exception:", type(e).__name__, e)
        return {"ok": False, "err": str(e), "log": buf}


def x_submatrix_metrics(X_np):
    """X_np: (batch, n) quantized sub-matrix; columns = selected fc dimensions."""
    Xf = X_np.astype(float)
    n = X_np.shape[1]
    r = int(np.linalg.matrix_rank(Xf))
    if r < n:
        cond = float("inf")
    else:
        cond = float(np.linalg.cond(Xf))
    if n < 2:
        mac = 0.0
    else:
        C = np.corrcoef(X_np.T)
        mac = float(np.mean(np.abs(C[np.triu_indices(n, 1)])))
    n_cells = int(X_np.size)
    n_zeros = int(np.sum(X_np == 0))
    zf = float(n_zeros) / float(n_cells) if n_cells else 0.0
    return {
        "rank": r,
        "cond": cond,
        "mean_abs_corr": mac,
        "max_entry": int(np.max(X_np)),
        "n_zeros": n_zeros,
        "zero_fraction": zf,
    }


def _random_column_set(dim, n, rng):
    return tuple(sorted(rng.choice(dim, size=n, replace=False)))


def sample_full_rank_columns(scaled, dim, n, rng, max_tries=500):
    for _ in range(max_tries):
        cols = _random_column_set(dim, n, rng)
        Xp = scaled[:, cols]
        if int(np.linalg.matrix_rank(Xp.astype(float))) >= n:
            return cols
    return None


def best_low_corr_columns(scaled, dim, n, rng, num_samples=100):
    best = None
    best_cols = None
    for _ in range(num_samples):
        cols = sample_full_rank_columns(scaled, dim, n, rng)
        if cols is None:
            continue
        m = x_submatrix_metrics(scaled[:, cols])
        if best is None or m["mean_abs_corr"] < best:
            best = m["mean_abs_corr"]
            best_cols = cols
    return best_cols, best


def greedy_low_zero_columns(scaled, n, fc_dim):
    """Greedily add columns in decreasing order of "nonzero count across the batch", keeping a column only when it increases the rank, until n columns are chosen."""
    nnz = np.count_nonzero(scaled, axis=0)
    order = [int(j) for j in np.argsort(-nnz)]
    cols = []
    used = set()
    for j in order:
        if len(cols) >= n:
            break
        if j in used:
            continue
        cand = cols + [j]
        sub = scaled[:, cand]
        if int(np.linalg.matrix_rank(sub.astype(float))) == len(cand):
            cols.append(j)
            used.add(j)
    if len(cols) < n:
        for j in range(fc_dim):
            if len(cols) >= n:
                break
            if j in used:
                continue
            cand = cols + [j]
            if int(np.linalg.matrix_rank(scaled[:, cand].astype(float))) == len(cand):
                cols.append(j)
                used.add(j)
    if len(cols) < n:
        return None
    return tuple(cols[:n])


def best_low_zero_random_columns(scaled, dim, n, rng, num_samples):
    """Among random full-rank n-column sets, pick the one whose sub-matrix has the most nonzeros (fewest zeros)."""
    best_nz = None
    best_cols = None
    for _ in range(int(num_samples)):
        cols = sample_full_rank_columns(scaled, dim, n, rng)
        if cols is None:
            continue
        Xp = scaled[:, cols]
        nz = int(np.count_nonzero(Xp))
        if best_nz is None or nz > best_nz:
            best_nz = nz
            best_cols = cols
    return best_cols, best_nz


def find_rows_perm_for_stat(A_base, X_np, n, m_stat, max_tries=5000):
    """Use the given X sub-matrix to find a row permutation that lets small-modulus Step1 succeed (same logic as build_cnn_instances)."""
    X_sec = matrix(ZZ, X_np.tolist())
    if X_sec.rank() < n:
        return None
    a = vector(ZZ, X_sec.column(0))
    m = A_base.nrows()
    for _ in range(max_tries):
        rows = list(range(m))
        shuffle(rows)
        P = matrix(ZZ, [list(A_base[i]) for i in rows])
        if P.rank() < n:
            continue
        b_raw = P * a
        x0_sm = next_prime(Integer(max(abs(Integer(x)) for x in b_raw) + 1))
        b_sm_full = vector(ZZ, [Integer(mod(b_raw[i], x0_sm)) for i in range(m)])
        if gcd(b_sm_full[0], x0_sm) != 1:
            continue
        P_stat = P[:m_stat, :]
        b_stat = vector(ZZ, [b_sm_full[i] for i in range(m_stat)])
        MO_sm, _, _, _ = Step1(n, -1, x0_sm, a, P_stat, b_stat, m_stat, BKZ=STEP1_INTEGER_MO)
        if MO_sm != -1:
            return {
                "rows": rows,
                "x0_sm": x0_sm,
                "P_stat": P_stat,
                "b_stat": b_stat,
                "a": a,
                "MO_sm": MO_sm,
            }
    return None


def build_stat_from_fixed_perm(A_base, rows_perm, scaled, cols, n, m_stat):
    """Fix the row permutation and swap only fc columns; used to compare different X sub-matrices."""
    cols = tuple(int(c) for c in cols)
    X_np = scaled[:, cols]
    met = x_submatrix_metrics(X_np)
    if met["rank"] < n:
        return None, "rank_X", met
    X_sec = matrix(ZZ, X_np.tolist())
    B_bound = max(ZZ(100), max(abs(ZZ(x)) for x in X_sec.list()))
    a = vector(ZZ, X_sec.column(0))
    P = matrix(ZZ, [list(A_base[i]) for i in rows_perm])
    if P.rank() < n:
        return None, "rank_P", met
    b_raw = P * a
    x0_sm = next_prime(Integer(max(abs(Integer(x)) for x in b_raw) + 1))
    b_sm_full = vector(ZZ, [Integer(mod(b_raw[i], x0_sm)) for i in range(P.nrows())])
    if gcd(b_sm_full[0], x0_sm) != 1:
        return None, "gcd", met
    P_stat = P[:m_stat, :]
    b_stat = vector(ZZ, [b_sm_full[i] for i in range(m_stat)])
    MO_sm, _, _, _ = Step1(n, -1, x0_sm, a, P_stat, b_stat, m_stat, BKZ=STEP1_INTEGER_MO)
    if MO_sm == -1:
        return None, "Step1", met
    H_st = {
        "n": n,
        "m": m_stat,
        "B": B_bound,
        "kappa": -1,
        "x0": x0_sm,
        "a": a,
        "X": P_stat,
        "b": b_stat,
    }
    return (H_st, MO_sm), None, met


def _print_matrix_full(X_arr, title):
    _pt = np.get_printoptions()
    np.set_printoptions(threshold=sys.maxsize, linewidth=110, edgeitems=20)
    print(title)
    print(X_arr)
    np.set_printoptions(**_pt)


def run_experiment_for_fc_scale(fc_scale, light, skip_ns, scale_index, do_ns_tail):
    import warnings

    rng = np.random.default_rng(int(scale_index) + int(1))
    if light:
        NUM_RANDOM = 2
        NUM_GREEDY_POOL = 24
        NZ_RANDOM_SAMPLES = 80
    else:
        NUM_RANDOM = 15
        NUM_GREEDY_POOL = 120
        NZ_RANDOM_SAMPLES = 500

    A_np, X_pivot, n, m, scaled, fc_dim, pivot_cols = load_cnn_AX(
        10, 100, fc_scale=fc_scale
    )
    print(
        "CNN data [fc_scale=%s]: A" % fc_scale,
        A_np.shape,
        " fc_dim=",
        fc_dim,
        " m=",
        m,
        " n=",
        n,
    )
    print("pivot column indices (reference):", pivot_cols)
    met_p = x_submatrix_metrics(X_pivot)
    print(
        "pivot X metrics: rank=%s zero_frac=%.2f cond=%.4g mean|corr|=%.4f max_entry=%s"
        % (
            met_p["rank"],
            met_p["zero_fraction"],
            met_p["cond"],
            met_p["mean_abs_corr"],
            met_p["max_entry"],
        )
    )
    _print_matrix_full(
        X_pivot,
        "\n========== Reference: pivot-column CNN X (floor(fc*%s)) shape=%s =========="
        % (fc_scale, X_pivot.shape),
    )

    cols_greedy = greedy_low_zero_columns(scaled, n, fc_dim)
    cols_nz_rand, nz_score = best_low_zero_random_columns(
        scaled, fc_dim, n, rng, NZ_RANDOM_SAMPLES
    )
    if cols_greedy is not None:
        cols_primary = cols_greedy
        primary_tag = "low_zero_greedy"
    elif cols_nz_rand is not None:
        cols_primary = cols_nz_rand
        primary_tag = "low_zero_random_best"
    else:
        cols_primary = pivot_cols
        primary_tag = "pivot_fallback"

    X_primary = scaled[:, cols_primary]
    met_m = x_submatrix_metrics(X_primary)
    print(
        "\n--- primary X column strategy: %s cols=%s ---" % (primary_tag, cols_primary)
    )
    print(
        "primary X metrics: rank=%s zero_frac=%.2f (zeros=%s/%s) cond=%.4g mean|corr|=%.4f max_entry=%s"
        % (
            met_m["rank"],
            met_m["zero_fraction"],
            met_m["n_zeros"],
            int(X_primary.size),
            met_m["cond"],
            met_m["mean_abs_corr"],
            met_m["max_entry"],
        )
    )
    if cols_nz_rand is not None:
        print(
            "  low-zero random baseline: sampled %s sets, best nonzero count=%s / %s"
            % (NZ_RANDOM_SAMPLES, nz_score, int(n * n))
        )

    _print_matrix_full(
        X_primary,
        "\n========== Primary CNN X (floor(fc*%s), prefer few-zero columns) shape=%s =========="
        % (fc_scale, X_primary.shape),
    )
    mx_ent = max(int(met_m["max_entry"]), 1)
    rng_show = np.random.default_rng(int(42) + int(scale_index))
    X_rand = rng_show.integers(0, mx_ent + 1, size=X_primary.shape, dtype=np.int64)
    _print_matrix_full(
        X_rand,
        "\n========== Random baseline X (same shape, U{0..%s}, unrelated to CNN) ==========" % mx_ent,
    )
    print(
        "random X: rank=%s mean|corr|=%.4f"
        % (
            int(np.linalg.matrix_rank(X_rand.astype(float))),
            float(
                np.mean(
                    np.abs(
                        np.corrcoef(X_rand.T)[np.triu_indices(X_rand.shape[1], 1)]
                    )
                )
            )
            if X_rand.shape[1] > 1
            else 0.0,
        )
    )

    A_base = matrix(ZZ, A_np.tolist())
    m_stat = min(m, 10 * n)

    print("\n--- [fc_scale=%s] searching for a fixed row permutation with primary X ---" % fc_scale)
    found = find_rows_perm_for_stat(A_base, X_primary, n, m_stat, max_tries=5000)
    if found is None and tuple(cols_primary) != tuple(pivot_cols):
        print("primary X failed to yield a row permutation; falling back to pivot X...")
        found = find_rows_perm_for_stat(A_base, X_pivot, n, m_stat, max_tries=5000)
    if found is None:
        print("no suitable row permutation found (Step1/gcd); trying a single build.")
        H_ns, MO_ns, H_st, MO_st = build_cnn_instances(A_np, X_primary, n)
        if H_ns is None:
            return
        run_ns(H_ns, MO_ns)
        run_stat(H_st, MO_st)
        return

    rows_perm = found["rows"]
    print("row permutation found, m_stat=%s x0_sm nbits=%s" % (m_stat, found["x0_sm"].nbits()))

    column_trials = []
    seen = set()

    def _add_trial(name, cols):
        if cols is None:
            return
        if cols in seen:
            return
        seen.add(cols)
        column_trials.append((name, cols))

    _add_trial("low_zero_greedy", cols_greedy)
    _add_trial("low_zero_random_best", cols_nz_rand)
    _add_trial("pivot", pivot_cols)
    for k in range(NUM_RANDOM):
        cols = sample_full_rank_columns(scaled, fc_dim, n, rng)
        if cols is None or cols in seen:
            continue
        seen.add(cols)
        column_trials.append(("random_%s" % (k + 1), cols))

    greedy_corr_cols, greedy_mac = best_low_corr_columns(
        scaled, fc_dim, n, rng, NUM_GREEDY_POOL
    )
    if greedy_corr_cols is not None and greedy_corr_cols not in seen:
        column_trials.append(("low_corr_best", greedy_corr_cols))
        print(
            "low-correlation sampling over %s full-rank column sets, best mean|corr|=%.4f cols=%s"
            % (NUM_GREEDY_POOL, greedy_mac, greedy_corr_cols)
        )

    print(
        "\n========== [fc_scale=%s] multiple column selections x two ICA scales B ==========" % fc_scale
    )
    print(
        "Note: NFound = number of S2 rows that exactly match columns of P; nrafound = number of recovered coefficients that hit the secret a."
        " B=1 is closer to the 0/1 P; B=B_bound comes from the X quantization upper bound (often inconsistent with P).\n"
    )

    for name, cols in column_trials:
        cols_str = str(cols) if len(cols) <= 14 else (str(cols)[:90] + "...")
        print("\n>>> [fc_scale=%s] column set [%s] cols=%s" % (fc_scale, name, cols_str))
        X_sub = scaled[:, cols]
        _print_matrix_full(
            X_sub,
            "    --- X selected in this group (floor(fc*%s)) shape=%s ---" % (fc_scale, X_sub.shape),
        )
        out, err, met = build_stat_from_fixed_perm(A_base, rows_perm, scaled, cols, n, m_stat)
        print(
            "    X metrics: rank=%s zero_frac=%.2f cond=%.4g mean|corr|=%.4f max_entry=%s"
            % (
                met["rank"],
                met["zero_fraction"],
                met["cond"],
                met["mean_abs_corr"],
                met["max_entry"],
            )
        )
        if out is None:
            print("    Step1/algebra failed:", err)
            continue
        H_st, MO_sm = out
        B_bound = H_st["B"]
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            r1 = run_stat(H_st, MO_sm, B_ica=1)
            r2 = run_stat(H_st, MO_sm, B_ica=B_bound)
        wica = [str(w.message) for w in wrec]
        if wica:
            print("    sklearn warnings (excerpt):", wica[-3:])

        def summarize(tag, r):
            if not r.get("ok"):
                return "%s: exception %s" % (tag, r.get("err", "?"))
            return "%s: NFound=%s nrafound=%s" % (tag, r["nfound"], r["nrafound"])

        print("    ", summarize("B=1", r1))
        print("    ", summarize("B=B_bound(%s)" % B_bound, r2))

    if (not skip_ns) and do_ns_tail:
        print(
            "\n========== NS (same columns as primary X, fc_scale=%s) =========="
            % fc_scale
        )
        H_ns, MO_ns, _, _ = build_cnn_instances(A_np, X_primary, n)
        if H_ns is not None:
            run_ns(H_ns, MO_ns)
        else:
            print("NS branch did not find an instance, skipping.")


def main():
    light = os.environ.get("CNN_STAT_LIGHT", "").strip() in ("1", "true", "yes")
    skip_ns = os.environ.get("CNN_STAT_SKIP_NS", "").strip() in ("1", "true", "yes") or light

    ensure_fc_hssp_npys(batch_size=10, dataset="cifar10")

    scales = parse_fc_scales()
    if light:
        print("(CNN_STAT_LIGHT=1: fewer column trials, skip NS)")
    print("fc quantization factors for this run CNN_FC_SCALES =", scales)

    nsc = len(scales)
    for scale_index, fc_scale in enumerate(scales):
        print("\n" + "#" * 72)
        print("###  fc_scale = %s  (%s / %s)  ###" % (fc_scale, scale_index + 1, nsc))
        print("#" * 72)
        do_ns = scale_index == nsc - 1
        run_experiment_for_fc_scale(fc_scale, light, skip_ns, scale_index, do_ns)

    if skip_ns:
        print("\n========== NS skipped (CNN_STAT_SKIP_NS or LIGHT) ==========")


main()
