#!/usr/bin/env sage
r"""
Transpose attack: build the mHLCP instance from the FC1 weight-gradient backprop
factor instead of the forward FC1 pre-ReLU output.

The forward attack (``run_mhlcp_cnn_quant_x.sage``) recovers ``X = fc1_pre_relu``.
This script attacks the *transpose* factor instead: ``X = backprop_masked_fc``
(i.e. ``(softmax-onehot) @ W_fc2 * relu_mask``), so the recovered rows are the
per-sample upstream gradient signals. The companion factor ``A = fc_input_batch``
is the same in both attacks.

The script is the bridge for the signed-X / transpose research line. The two
row-selection strategies ``MHLCP_ROW_MODE=random|bias`` are exposed as siblings:

  - ``random`` (default, **the main method**) — pure ``np.random.permutation``,
    seeded by ``ROWSEED``. Each row is treated equally.
  - ``bias`` (**exploration**) — sort rows by ``|sum_j X_int[i, j]| = |∇b_1[i]|``.
    The FC1 bias gradient is observable to the attacker, so this is a way to
    bias selection towards "informative" rows. Use this only as an ablation
    alternative; ``random`` is the canonical baseline.

Data: ``python src/cnn.py --mode fc_hssp --batch_size <BS> --dataset <DATASET>``
exports ``backprop_masked_fc.npy`` and ``fc_input_batch.npy`` under
``expdata/fc_hssp/<DATASET>/bs<BS>/`` (or ``bs<BS>/run_<RUN>/`` when
``HSSP_RUN_ID=<RUN>`` is also set).

Usage (from the repo root)::

    BS=20 sage lattice/run_mhlcp_transpose.sage                   # random, default
    BS=20 MHLCP_ROW_MODE=bias sage lattice/run_mhlcp_transpose.sage   # bias, exploration

Environment variables:
  BS              Batch size (= hidden dimension n).
  RUN             Multi-run id; reads ``bs<BS>/run_<RUN>/`` if present.
  C               Quantization scale (X_int = round(bp * C)).
  MR              Sample-row multiplier; ``m = min(MR*n + 10, fc_hidden_dim)``.
  M_TARGET        Override m directly (otherwise computed from MR).
  NX0             Bit-length of the modulus ``x0``.
  ROWSEED         Seed for the random row/column permutations.
  RBOX            Floor for the ``recoverBox`` cap (the cap is ``max(RBOX, 800*n)``).
  MHLCP_ROW_MODE  ``random`` (default) or ``bias``.
  MHLCP_DATASET   Dataset directory (default ``cifar10``).
"""
import os
import sys

import numpy as np
from numpy.linalg import matrix_rank
from time import time as walltime_now

# --------------- self-locate + load core lattice library ---------------
_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
_REPO = os.path.dirname(_d) if os.path.basename(_d) == "lattice" else _d
while _REPO and _REPO != "/" and not os.path.isfile(os.path.join(_REPO, "lattice", "multi_dim_hssp.sage")):
    _REPO = os.path.dirname(_REPO)
if not os.path.isfile(os.path.join(_REPO, "lattice", "multi_dim_hssp.sage")):
    raise FileNotFoundError("Cannot locate lattice/multi_dim_hssp.sage; run from the repo root or lattice/.")
_LATTICE = os.path.join(_REPO, "lattice")
os.chdir(_LATTICE)
load("multi_dim_hssp.sage")
os.chdir(_REPO)


# --------------- env-var configuration ---------------
BS       = int(os.environ.get("BS", "20"))
RUN      = int(os.environ.get("RUN", "0"))
C        = int(os.environ.get("C", "50"))
MR       = int(os.environ.get("MR", "7"))
NX0      = int(os.environ.get("NX0", "20"))
ROWSEED  = int(os.environ.get("ROWSEED", "42"))
RBOX     = int(os.environ.get("RBOX", "100000"))
ROW_MODE = os.environ.get("MHLCP_ROW_MODE", "random").strip().lower()
DATASET  = os.environ.get("MHLCP_DATASET", "cifar10").strip()
M_TARGET = int(os.environ.get("M_TARGET", "0"))

if ROW_MODE not in ("random", "bias"):
    raise ValueError("MHLCP_ROW_MODE must be 'random' or 'bias', got %r" % ROW_MODE)


# --------------- locate the FC tensors exported by cnn.py --mode fc_hssp ---------------
_candidates = [
    "expdata/fc_hssp/%s/bs%d/run_%04d" % (DATASET, BS, RUN),
    "expdata/fc_hssp/%s/bs%d" % (DATASET, BS),
]
data_dir = None
for c in _candidates:
    if os.path.isfile(os.path.join(c, "backprop_masked_fc.npy")):
        data_dir = c
        break
if data_dir is None:
    print("NODATA bs=%d run=%d  (run `python src/cnn.py --mode fc_hssp --batch_size %d --dataset %s` first)"
          % (BS, RUN, BS, DATASET))
    sys.stdout.flush()
    sys.exit(1)

print("[transpose] data_dir=%s  ROW_MODE=%s" % (data_dir, ROW_MODE))
sys.stdout.flush()

bp = np.load(os.path.join(data_dir, "backprop_masked_fc.npy")).astype(np.float64)
Z  = np.load(os.path.join(data_dir, "fc_input_batch.npy")).astype(np.float64)

n     = BS
X_int = np.round(bp * C).astype(np.int64)
A_int = np.round(Z * C).astype(np.int64)


# --------------- row selection ---------------
if ROW_MODE == "bias":
    # |∇b1| = |sum over batch of backprop_masked|; attacker-observable.
    bias_grad = bp.sum(axis=1)
    order = np.argsort(-np.abs(bias_grad))
else:  # random (default)
    np.random.seed(ROWSEED)
    order = np.random.permutation(X_int.shape[0])

m = M_TARGET if M_TARGET > 0 else (MR * n + 10)
m = min(m, X_int.shape[0])

sel = []
for idx in order:
    sel.append(int(idx))
    rk = matrix_rank(X_int[sel, :])
    if rk == min(len(sel), n):
        if len(sel) >= m:
            break
    elif rk < len(sel):
        sel.pop()
X_np = X_int[sel[:m], :]


# --------------- A column selection (random, fixed seed offset) ---------------
l = n
np.random.seed(ROWSEED + 1)
cperm = np.random.permutation(A_int.shape[1])
acols = []
for idx in cperm:
    acols.append(int(idx))
    rk = matrix_rank(A_int[:, acols])
    if rk == min(len(acols), n):
        if len(acols) >= l:
            break
    elif rk < len(acols):
        acols.pop()
A_np = A_int[:, acols[:l]]


# --------------- build mHLCP instance and solve ---------------
B_X       = int(np.max(np.abs(X_np)))
B_X_step2 = max(B_X, 2)
neg_pct   = 100.0 * float((X_np < 0).sum()) / X_np.size
zero_pct  = 100.0 * float((X_np == 0).sum()) / X_np.size
actual_m  = X_np.shape[0]

X_sage = matrix(ZZ, X_np.tolist())
A_sage = matrix(ZZ, A_np.tolist())
H = X_sage * A_sage
set_random_seed(ROWSEED + 999)
x0 = genpseudoprime(NX0)
H_mod = H % x0

try:
    Matrix(Integers(x0), H_mod[:l, :l]).inverse()
except Exception:
    print("NOTINV bs=%d run=%d  (leading l x l block of B is not invertible mod x0)" % (BS, RUN))
    sys.stdout.flush()
    sys.exit(1)

t0 = walltime_now()
MO = orthoLattice_mat(H_mod, x0)
M2 = MO.LLL()
ke = kernelLLL(M2[:actual_m - n])
t_step1 = walltime_now() - t0

rbox = max(Integer(RBOX), Integer(800) * n)
MB, beta = Step2_BK_mat(ke, n, actual_m, X_sage, kappa=-1, B=B_X_step2, recover_box_max=rbox)
t_total = walltime_now() - t0


# --------------- score: NFound (signed match) and unique-column count ---------------
nf = 0
unique_cols = 0
if MB != 0 and hasattr(MB, "nrows") and MB.nrows() > 0:
    Xset = set()
    for j in range(n):
        Xset.add(tuple(int(x) for x in X_sage.column(j)))
        Xset.add(tuple(-int(x) for x in X_sage.column(j)))
    nf = sum(1 for r in MB.rows() if tuple(int(x) for x in r) in Xset)

    found_cols = set()
    for r in MB.rows():
        t = tuple(int(x) for x in r)
        tneg = tuple(-x for x in t)
        for j in range(n):
            cj = tuple(int(x) for x in X_sage.column(j))
            if t == cj or tneg == cj:
                found_cols.add(j)
                break
    unique_cols = len(found_cols)

st = "S" if unique_cols == n else ("P" if unique_cols > 0 else "F")
mb_rows = int(MB.nrows()) if MB != 0 and hasattr(MB, "nrows") else 0


# --------------- pipe-separated RESULT line (consumed by scripts/aggregate_subsample.py) ---------------
print("RESULT|sel=%s|bs=%d|run=%d|c=%d|nf=%d|unique=%d|n=%d|st=%s|bx=%d|beta=%d"
      "|neg=%.1f|zero=%.1f|m=%d|tstep1=%.1f|ttotal=%.1f|mb=%d"
      % (ROW_MODE, BS, RUN, C, nf, unique_cols, n, st, B_X, int(beta),
         float(neg_pct), float(zero_pct), actual_m,
         float(t_step1), float(t_total), mb_rows))
sys.stdout.flush()
