#!/usr/bin/env sage
r"""
CNN raw FC1 output -> mHLCP (matches the pipeline described in the paper).

1. **Raw X**: ``fc1_pre_relu_fc.npy``, shape **1000xn** (pre-ReLU floats).
2. **Integer pool**: ``X_int = floor(raw X * 1000)`` (fixed factor 1000; the environment variable ``CNN_INT_SCALE`` can override this).
3. **Sampled X (``TARGET_ROWS``xn)**:
   - Shuffle row indices via ``ROW_SELECT_SEED``, greedily reach **column rank = n**, then fill up to **TARGET_ROWS** rows at random (``m`` may exceed 100).
   - Guarantees **rank(X)=n** over the reals; the row count **m** only needs to be **>= n**.
4. **A**: quantized at the same scale as **X**, ``A_ij = floor(U * CNN_INT_SCALE)`` with ``U ~ Uniform[0, B_A+1)`` (defaults ``B_A=100`` and ``CNN_INT_SCALE=1000``, matching ``floor(raw X * 1000)``). For the legacy discrete ``{0..B_A}`` distribution, set the environment variable ``MHLCP_A_DISCRETE=1``.
5. **H** (called ``B`` in the code): ``H == X*A (mod x0)``; solved via ``hssp_attack(H,'ns')``.

Optional ``MHLCP_ROW_MODE=relu_stratified``: restores the legacy row selection "stratified by ReLU Hamming weight + rank" (requires ``relu_mask_fc.npy``).

Data: ``python src/cnn.py --mode fc_hssp --batch_size <BATCH_SIZE>`` (must match the script's ``BATCH_SIZE=n``).

Environment variable ``MHLCP_ATTACK``: ``ns`` (default), ``statistical``, ``both``.
``MHLCP_TARGET_ROWS``: number of rows **m** sampled from **X** (default 150).
With the current default **n=20**, the npy files under ``expdata/.../bs20/`` must exist beforehand (``python src/cnn.py --mode fc_hssp --batch_size 20``).

Usage (**run from the repository root**, otherwise the npy files cannot be found)::

  cd /path/to/gradient-lattice-attack
  python3 src/cnn.py --mode fc_hssp --batch_size 20
  MHLCP_ATTACK=ns sage lattice/run_mhlcp_cnn_quant_x.sage

Note that ``python3 -u`` for unbuffered output does not apply to sage; this script already uses ``flush=True`` wherever practical.

**On servers / when Sage compiles the script under /tmp**: ``sys.argv[0]`` often points to a temporary directory, so the project root cannot be located automatically. Pick one of the following:

- ``export MHLCP_REPO_ROOT=/path/to/gradient-lattice-attack`` and then run; or
- ``cd`` to the repo root first (``ls lattice/multi_dim_hssp.sage`` should show the file) and then run.

If the terminal still produces no output, inspect ``/tmp/run_mhlcp_cnn_quant_x.boot.log`` (the very first lines of the script write to it).

For parallel / multi-instance experiments (multi-process is recommended over BKZ-internal threading):
- ``MHLCP_ROW_SELECT_SEED``: controls the row sampling of X.
- ``MHLCP_ATTACK_SEED``: controls the column sampling of A (and subsequent randomness inside the attack).
"""
import os
import sys
import re
import contextlib
import io
import fcntl


def _boot_log(msg):
    """stdout may be redirected or fully buffered; using stderr plus a fixed-path log file makes it easy to confirm that the script actually reached Python."""
    line = "[mhlcp] %s\n" % msg
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        with open("/tmp/run_mhlcp_cnn_quant_x.boot.log", "a") as f:
            f.write(line)
    except Exception:
        pass


_boot_log(
    "BOOT argv0=%r cwd=%r MHLCP_REPO_ROOT=%r"
    % (
        sys.argv[0] if sys.argv else None,
        os.getcwd(),
        os.environ.get("MHLCP_REPO_ROOT", ""),
    )
)

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _find_hssp_repo_root():
    """Locate the repository root (containing ``lattice/multi_dim_hssp.sage``). On servers, Sage often uses a ``.sage.py`` under /tmp, so we have to rely on cwd or MHLCP_REPO_ROOT."""

    def _has(d):
        return os.path.isfile(os.path.join(d, "lattice", "multi_dim_hssp.sage"))

    tried = []

    for key in ("MHLCP_REPO_ROOT", "HSSP_ROOT"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        d = os.path.abspath(os.path.expanduser(raw))
        tried.append("%s=%s" % (key, d))
        if _has(d):
            return d, "env:%s" % key
        print(
            "[warning] %s is set but the directory does not contain lattice/multi_dim_hssp.sage: %s" % (key, d),
            flush=True,
        )

    seeds = []
    if len(sys.argv) > 0 and sys.argv[0]:
        seeds.append(os.path.abspath(os.path.dirname(sys.argv[0])))
    seeds.append(os.path.abspath(os.getcwd()))
    for s in seeds:
        if not s or s in (".", "/"):
            continue
        cur = s
        for _ in range(16):
            if _has(cur):
                return cur, "walk-from:%s" % s
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    print("Path hints tried:", "; ".join(tried) if tried else "(no MHLCP_REPO_ROOT/HSSP_ROOT)", flush=True)
    print("  argv[0]:", sys.argv[0] if sys.argv else "", flush=True)
    print("  cwd:", os.getcwd(), flush=True)
    return None, None


_repo, _repo_via = _find_hssp_repo_root()
if _repo is None:
    print(
        "error: cannot find lattice/multi_dim_hssp.sage.",
        flush=True,
    )
    print(
        "  fix: export MHLCP_REPO_ROOT=/path/to/gradient-lattice-attack",
        flush=True,
    )
    print("  then: cd \"$MHLCP_REPO_ROOT\" && MHLCP_ATTACK=ns sage lattice/run_mhlcp_cnn_quant_x.sage", flush=True)
    sys.exit(1)
print("[path] project root: %s  (source: %s)" % (_repo, _repo_via), flush=True)

# Sage's load() resolves nested load("foo.sage") relative to cwd, so we cd into
# lattice/ for the load chain (multi_dim_hssp -> hssp -> building/...) to work,
# then cd back to the repo root so subsequent expdata/... I/O resolves correctly.
_LATTICE = os.path.join(_repo, "lattice")
os.chdir(_LATTICE)
load("multi_dim_hssp.sage")
os.chdir(_repo)

# Pull in the path resolvers and row/column selection strategies.
load("lattice/_data_paths.sage")
load("lattice/_row_select.sage")

BATCH_SIZE = 20
# Number of sampled rows m (can be relaxed); making it too large may change the shape of the orthogonal lattice, and NFound is not necessarily monotonically improved. Override via environment variable.
TARGET_ROWS = int(os.environ.get("MHLCP_TARGET_ROWS", "150"))
L = 20
INT_SCALE = 1000
B_A = int(os.environ.get("MHLCP_B_A", "100"))
B_X_RANDOM = int(os.environ.get("MHLCP_B_X", "1"))
NX0_BITS = 192
RECOVER_BOX_MAX_BASE = int(os.environ.get("MHLCP_RECOVER_BOX_MAX", "220000"))
# Allow the random choices used by X (row sampling) and A (column sampling) to be
# driven by externally specified seeds, so that running many instances in parallel
# can boost the probability of "recovering more columns".
ROW_SELECT_SEED = int(os.environ.get("MHLCP_ROW_SELECT_SEED", "0"))
ATTACK_SEED = int(os.environ.get("MHLCP_ATTACK_SEED", "1"))


def main():
    print("", flush=True)
    print("=" * 72, flush=True)
    print("run_mhlcp_cnn_quant_x", flush=True)
    print("  working directory cwd:", os.getcwd(), flush=True)
    print(
        "  parameter meanings: n = number of columns of X = number of rows of A = secret dimension (determined by the npy column count, must match BATCH_SIZE)",
        flush=True,
    )
    print(
        "                      m = number of rows of X = number of sampled rows (TARGET_ROWS in the script, env var MHLCP_TARGET_ROWS)",
        flush=True,
    )
    print(
        "                      l = number of columns of A (script constant L; typically l = n, so A is n x n)",
        flush=True,
    )
    print(
        "  current script: BATCH_SIZE=%s  L=%s  TARGET_ROWS(target m)=%s  INT_SCALE=%s  NX0_BITS=%s"
        % (BATCH_SIZE, L, TARGET_ROWS, INT_SCALE, NX0_BITS),
        flush=True,
    )
    print("=" * 72, flush=True)

    x_source = os.environ.get("MHLCP_X_SOURCE", "real").strip().lower()
    row_mode = os.environ.get("MHLCP_ROW_MODE", "random").strip().lower()
    int_scale = float(os.environ.get("CNN_INT_SCALE", str(INT_SCALE)))

    use_relu_only = False
    scale_100 = False

    if x_source == "random":
        n = int(BATCH_SIZE)
        b_x = int(B_X_RANDOM)
        x_nz = float(X_NON_ZERO)
        m_target = int(TARGET_ROWS)
        X_int_sel = gen_random_sparse_matrix(
            m_target, n, b_x, x_nz, seed=ROW_SELECT_SEED, require_rank=n,
        )
        pre_sel = X_int_sel.astype(np.float64)
        print(
            "X source: randomly generated  m=%d n=%d B_X=%d non_zero=%.2f  seed=%d"
            % (m_target, n, b_x, x_nz, ROW_SELECT_SEED)
        )
    else:
        use_relu_only = os.environ.get("CNN_USE_RELU_MASK_X", "").strip() in (
            "1",
            "true",
            "True",
            "yes",
        )

        if use_relu_only:
            relu_path, relu_int, num_rows, n = load_relu_mask(BATCH_SIZE)
            if num_rows < TARGET_ROWS:
                raise RuntimeError("relu has %s rows < %s" % (num_rows, TARGET_ROWS))
            print("[mode] CNN_USE_RELU_MASK_X=1: integer pool is a 0/1 mask (x1000 does not apply).")
            scale_100 = os.environ.get("CNN_RELU_SCALE_100", "").strip() in ("1", "true", "True", "yes")
            X_pool = relu_int.copy()
            if scale_100:
                X_pool = X_pool * 100
            if row_mode == "relu_stratified":
                sel = select_rows_relu_stratified(relu_int, X_pool, int(n), TARGET_ROWS, seed=ROW_SELECT_SEED)
            else:
                sel = select_rows_random_full_rank(X_pool, int(n), TARGET_ROWS, seed=ROW_SELECT_SEED)
            X_int_sel = X_pool[sel, :]
            pre = X_pool.astype(np.float64)
            pre_sel = pre[sel, :]
            b_x = int(max(1, int(np.max(np.abs(X_int_sel)))))
        else:
            if row_mode == "relu_stratified":
                relu_path, relu_int, _nr, n = load_relu_mask(BATCH_SIZE)
                pre_path = resolve_pre_relu_path(relu_path)
            else:
                relu_int = None
                pre_path = resolve_pre_relu_standalone(BATCH_SIZE)
            pre = np.load(pre_path).astype(np.float64)
            num_rows, n = pre.shape
            print("Loaded FC1 pre-ReLU raw X: %s  shape=%s" % (pre_path, pre.shape))
            if num_rows < TARGET_ROWS:
                raise RuntimeError("pool has %s rows < %s" % (num_rows, TARGET_ROWS))
            if int_scale == -1:
                X_pool = (pre != 0).astype(np.int64)
                print("Integer pool: 0/1 binarization (non-zero -> 1), shape %s" % (X_pool.shape,))
            else:
                X_pool = np.floor(pre * int_scale).astype(np.int64)
                print(
                    "Integer pool: floor(raw * %s), shape %s"
                    % (int_scale, X_pool.shape)
                )
            if row_mode == "relu_stratified":
                if relu_int is None:
                    raise RuntimeError("MHLCP_ROW_MODE=relu_stratified requires relu_mask_fc.npy")
                sel = select_rows_relu_stratified(relu_int, X_pool, int(n), TARGET_ROWS, seed=ROW_SELECT_SEED)
            elif row_mode == "condnum":
                print(
                    "Row selection: condition-number greedy (seed ROW_SELECT_SEED=%s)"
                    % ROW_SELECT_SEED
                )
                sel = select_rows_condnum_greedy(X_pool, int(n), TARGET_ROWS, seed=ROW_SELECT_SEED)
            else:
                print(
                    "Row selection: random + column rank=%s (seed ROW_SELECT_SEED=%s)"
                    % (n, ROW_SELECT_SEED)
                )
                sel = select_rows_random_full_rank(X_pool, int(n), TARGET_ROWS, seed=ROW_SELECT_SEED)
            pre_sel = pre[sel, :]
            X_int_sel = X_pool[sel, :]
            b_x = int(max(1, int(np.ceil(np.max(np.abs(X_int_sel))))))

    m = int(X_int_sel.shape[0])
    # recover_max: search budget, depends only on n (eliminating cross-batch_size unfairness caused by B_X randomness).
    # Empirical formula: 500*n^2, lower bound 220000, upper bound 1200000.
    recover_max_env = os.environ.get("MHLCP_RECOVER_BOX_MAX", "").strip()
    if recover_max_env:
        recover_max = int(recover_max_env)
    else:
        recover_max = max(
            RECOVER_BOX_MAX_BASE,
            min(Integer(1200000), Integer(500) * int(n)^2),
        )

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    if x_source == "random":
        xdesc = "random [0,%d] non_zero=%.2f" % (b_x, float(X_NON_ZERO))
        print("\n========== Random X (%dx%d), first 10 rows ==========" % (m, n))
    elif use_relu_only:
        xdesc = "0/1 mask" + (" x100" if scale_100 else "")
        print("\n========== [mask mode] first 10 rows of sampled X (shown as floats) ==========")
    else:
        xdesc = "floor(raw*%s)" % int_scale
        print("\n========== Raw X (selected %s rows) pre-ReLU floats, first 10 rows ==========" % m)
    print(pre_sel[:10, :])
    print("========== Sampled integer X (%s), used in H == X*A (mod x0), %dx%d ==========" % (xdesc, X_int_sel.shape[0], X_int_sel.shape[1]))
    print(matrix(ZZ, X_int_sel.tolist()))
    print("================================================================\n")

    X_sage = matrix(ZZ, X_int_sel.tolist())
    rk = int(X_sage.rank())
    x_mode_desc = "random(nz=%.2f)" % float(X_NON_ZERO) if x_source == "random" else row_mode
    print(
        "Sampled X: %sx%s   rank(X)=%s   B_X=%s   mode=%s"
        % (m, n, rk, b_x, x_mode_desc)
    )
    if rk < int(n):
        print("error: rank(X)<n; cannot proceed with mHLCP.")
        return

    set_random_seed(ATTACK_SEED)
    try:
        import numpy as _np

        _np.random.seed(ATTACK_SEED)
    except ImportError:
        pass

    a_source = os.environ.get("MHLCP_A_SOURCE", "real").strip().lower()
    a_nz = float(A_NON_ZERO)
    H = multi_hssp(
        int(n),
        L,
        kappa=-1,
        B_X=b_x,
        B_A=B_A,
        nx0_bits=NX0_BITS,
        recover_box_max=recover_max,
        A_int_scale=int_scale if a_source == "real" else None,
    )
    A_given = None
    if a_source == "real":
        fc_path = resolve_fc_input_path(BATCH_SIZE)
        fc_input = np.load(fc_path).astype(np.float64)  # shape: (batch_size, D)
        if int(fc_input.shape[0]) != int(n):
            raise RuntimeError(
                "fc_input row count=%s does not match n=%s; please ensure BATCH_SIZE aligns with n."
                % (fc_input.shape[0], n)
            )
        A_pool = np.floor(fc_input * int_scale).astype(np.int64)
        cols = select_cols_full_rank(A_pool, int(n), int(L), seed=ATTACK_SEED + 97)
        A_np = A_pool[:, cols]
        A_given = matrix(ZZ, A_np.tolist())
        print("A source: real FC input features (fc_input_batch)")
        print("  path: %s  raw shape=%s  quantization: floor(fc_input * %s)" % (fc_path, fc_input.shape, int_scale))
        print("  sampled columns: %s (chosen at random from D=%s, target rank=%s)" % (L, A_pool.shape[1], min(int(n), int(L))))
    elif a_nz < 1.0:
        A_np = gen_random_sparse_matrix(
            int(n), int(L), int(B_A), a_nz, seed=ATTACK_SEED + 13,
        )
        A_given = matrix(ZZ, A_np.tolist())
        print(
            "A source: random sparse integers [0, B_A=%d]  non_zero=%.2f  seed=%d"
            % (B_A, a_nz, ATTACK_SEED + 13)
        )
    else:
        print("A source: random discrete integers [0, B_A=%d]" % B_A)
    H.gen_instance_fixed_X(X_sage, A=A_given)
    print("A (n x l) after scale quantization, |A|_max=", max(abs(Integer(x)) for x in H.A.list()))
    print("Full matrix A (%sx%s):" % (n, L))
    print(H.A)
    if not ok_instance(H):
        print("B[:l,:l] is not invertible mod x0; aborting.")
        return
    # Force default attack mode to ns (avoids branch errors when the external env var is empty / undefined).
    os.environ.setdefault("MHLCP_ATTACK", "ns")
    attack_mode = os.environ.get("MHLCP_ATTACK", "ns").strip().lower()
    print(
        "Constructed H=B: n=%s l=%s m=%s  B_X=%s B_A=%s  x0.nbits()=%s  recover_box_max=%s  MHLCP_ATTACK=%s"
        % (n, L, m, b_x, B_A, H.x0.nbits(), recover_max, attack_mode)
    )

    def run_ns():
        print("\n--- Nguyen-Stern matrix variant (hssp_attack alg=ns) ---")
        try:
            out = hssp_attack(H, "ns")
        except Exception as e:
            print("hssp_attack(ns) exception:", type(e).__name__, e)
            return False
        if out is None or out == 0:
            print("Step2 did not produce a valid solution.")
            return False
        if not hasattr(out, "nrows"):
            print("returned:", type(out), out)
            return False
        Xrows = [tuple(r) for r in H.X.T.rows()]
        Xset = set(Xrows)
        nf = sum(1 for r in out.rows() if tuple(r) in Xset)
        print(">>> NS NFound(rows vs X.T)=", nf, "/", n)
        if nf == n:
            print("OK: NS recovered all n rows of X.")
            return True
        print("NS did not fully recover X.")
        return False

    def run_stat():
        print("\n--- Statistical / FastICA (hssp_attack alg=statistical, splitting A by columns into l vector HLCP instances) ---")
        try:
            out = hssp_attack(H, "statistical")
        except Exception as e:
            print("hssp_attack(statistical) exception:", type(e).__name__, e)
            return False
        if not isinstance(out, dict):
            print("returned:", type(out), out)
            return False
        if out.get("error"):
            print("statistical did not run:", out.get("error"))
            return False
        print(">>> statistical summary:", out)
        ok = bool(out.get("all_nfound_ge_n")) and bool(out.get("all_nrafound_ge_n"))
        if ok:
            print("OK: statistical reached NFound==n and nrafound==n on every column.")
        else:
            print(
                "statistical did not achieve full scores across all columns (ICA often fails for large B_X; try lowering CNN_INT_SCALE or comparing with a mask X at B_X=1)."
            )
        return ok

    if attack_mode == "statistical":
        run_stat()
    elif attack_mode == "both":
        run_ns()
        run_stat()
    else:
        if attack_mode not in ("ns", ""):
            print("[hint] Unknown MHLCP_ATTACK=%s; falling back to ns." % attack_mode)
        run_ns()


try:
    def _parse_cli_int(name, default):
        key1 = "--%s" % name
        key2 = "--%s=" % name
        for i, a in enumerate(sys.argv):
            if a == key1 and i + 1 < len(sys.argv):
                return int(sys.argv[i + 1])
            if a.startswith(key2):
                return int(a.split("=", 1)[1])
        return int(default)

    def _parse_cli_str(name, default=""):
        key1 = "--%s" % name
        key2 = "--%s=" % name
        for i, a in enumerate(sys.argv):
            if a == key1 and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
            if a.startswith(key2):
                return a.split("=", 1)[1]
        return default

    def _parse_cli_float(name, default):
        key1 = "--%s" % name
        key2 = "--%s=" % name
        for i, a in enumerate(sys.argv):
            if a == key1 and i + 1 < len(sys.argv):
                return float(sys.argv[i + 1])
            if a.startswith(key2):
                return float(a.split("=", 1)[1])
        return float(default)

    # batch_size is also exposed as an optional argument (corresponding to n in the paper).
    batch_size_arg = _parse_cli_int("batch_size", BATCH_SIZE)
    globals()["BATCH_SIZE"] = batch_size_arg
    # l should match n as closely as possible; otherwise B[:l,:l] in ok_instance() is more likely to be non-invertible because of dimension/rank constraints.
    # Default: L = batch_size. Can be overridden via the environment variable MHLCP_L.
    if os.environ.get("MHLCP_L", "").strip():
        globals()["L"] = int(os.environ["MHLCP_L"])
    else:
        globals()["L"] = int(batch_size_arg)

    # Adaptive TARGET_ROWS: when not specified via env var, compute automatically from n.
    # Empirically (fine scan at n=10, 20, 30, 40), m ~= 3n+10 gives the best cost/benefit tradeoff:
    #   - n=20, m=60(3n): 94% success rate, ~75s
    #   - n=20, m=75: 97%, ~104s
    #   - n=30, m=100(3.3n): 71%, ~122s
    #   - Larger m has diminishing returns but scales as O(m^5) in time.
    if not os.environ.get("MHLCP_TARGET_ROWS", "").strip():
        auto_m = int(3 * batch_size_arg + 10)
        globals()["TARGET_ROWS"] = auto_m
        print("[adaptive] MHLCP_TARGET_ROWS unset; based on n=%d, auto-setting m=%d (formula: 3n+10)"
              % (batch_size_arg, auto_m), flush=True)

    subsample_times = _parse_cli_int("subsample_times", 1)
    original_data_times = _parse_cli_int("original_data_times", 1)
    file_subdir = _parse_cli_str("file", "")
    globals()["X_NON_ZERO"] = _parse_cli_float("X_non_zero", 1.0)
    globals()["A_NON_ZERO"] = _parse_cli_float("A_non_zero", 1.0)

    # Unified log directory: when --file is given, use logs/<name>/; otherwise, use logs/ directly.
    logs_dir = os.path.join(os.getcwd(), "logs")
    if file_subdir:
        logs_dir = os.path.join(logs_dir, file_subdir)
    os.makedirs(logs_dir, exist_ok=True)

    # Default attack mode (also enforced again inside main()).
    os.environ.setdefault("MHLCP_ATTACK", "ns")
    attack_mode_for_name = os.environ.get("MHLCP_ATTACK", "ns").strip().lower()

    target_rows_for_name = int(os.environ.get("MHLCP_TARGET_ROWS", str(TARGET_ROWS)))

    nfound_summary_path = os.path.join(logs_dir, "nfound_summary.log")
    nfound_lock_path = os.path.join(logs_dir, "nfound_summary.lock")

    # If the FC HSSP npy files for this batch_size are missing under expdata, auto-generate them via cnn.py.
    def _ensure_fc_hssp_npys(batch_size):
        try:
            resolve_relu_path(batch_size)
            resolve_fc_input_path(batch_size)
            resolve_pre_relu_standalone(batch_size)
            return
        except FileNotFoundError:
            pass

        dataset_name = os.environ.get("MHLCP_DATASET", "cifar10").strip()
        if dataset_name != "cifar10":
            raise RuntimeError(
                "Automatic FC HSSP npy generation is currently only wired up for the default dataset='cifar10'. "
                "You set MHLCP_DATASET=%r; please generate the files manually or reset MHLCP_DATASET back to cifar10."
                % dataset_name
            )

        print(
            "[auto] fc_hssp npy files for bs%s are missing; running cnn.py automatically to generate them..." % batch_size,
            flush=True,
        )
        py = sys.executable
        import subprocess

        cmd = [
            py,
            os.path.join(os.getcwd(), "src", "cnn.py"),
            "--mode",
            "fc_hssp",
            "--dataset",
            dataset_name,
            "--batch_size",
            str(int(batch_size)),
        ]
        r = subprocess.run(cmd, cwd=os.getcwd())
        if r.returncode != 0:
            raise RuntimeError("src/cnn.py fc_hssp failed, exit=%s" % r.returncode)

        # Re-validate.
        resolve_relu_path(batch_size)
        resolve_fc_input_path(batch_size)
        resolve_pre_relu_standalone(batch_size)

    _x_src = os.environ.get("MHLCP_X_SOURCE", "real").strip().lower()
    _a_src = os.environ.get("MHLCP_A_SOURCE", "real").strip().lower()
    if _x_src == "random" and _a_src != "real":
        print("[skip] both X and A are random; skipping the CNN npy check.", flush=True)
    else:
        _ensure_fc_hssp_npys(batch_size_arg)

    def _extract_nfound_numbers(text):
        nums = []
        nums += [
            int(x)
            for x in re.findall(r"NFound\(rows vs X\.T\)=\s*([0-9]+)", text)
        ]
        nums += [
            int(x)
            for x in re.findall(r"NFound\(row match X\.T\)=\s*([0-9]+)", text)
        ]
        # Do not sort or de-duplicate; preserving the original order is closer to a "set" semantic.
        return nums

    total_trials = int(subsample_times) * int(original_data_times)
    trial_idx = 0
    for si in range(int(subsample_times)):
        for oi in range(int(original_data_times)):
            trial_idx += 1
            # Randomize rowseed / attseed for every trial.
            globals()["ROW_SELECT_SEED"] = int.from_bytes(os.urandom(4), "big")
            globals()["ATTACK_SEED"] = int.from_bytes(os.urandom(4), "big")
            cur_row_seed = globals()["ROW_SELECT_SEED"]
            cur_att_seed = globals()["ATTACK_SEED"]

            _tfile = __import__("time").strftime("%Y%m%d_%H%M%S")
            _xsrc_tag = re.sub(
                r"[^a-z0-9]+",
                "",
                os.environ.get("MHLCP_X_SOURCE", "real").strip().lower(),
            ) or "real"
            _fname_extra = "_xsrc%s" % _xsrc_tag
            if _x_src == "random":
                _bx_name = int(os.environ.get("MHLCP_B_X", str(globals().get("B_X_RANDOM", 1))))
                _xnz_name = ("%g" % float(globals().get("X_NON_ZERO", 1.0))).replace(".", "p").replace(
                    "-", "m"
                )
                _fname_extra += "_bx%d_xnz%s" % (_bx_name, _xnz_name)
            raw_log_path = os.path.join(
                logs_dir,
                "trial_%04d_%s_n%d_m%d_attack_%s_rowseed_%d_attseed_%d%s.log"
                % (
                    trial_idx,
                    _tfile,
                    int(batch_size_arg),
                    int(target_rows_for_name),
                    attack_mode_for_name,
                    int(cur_row_seed),
                    int(cur_att_seed),
                    _fname_extra,
                ),
            )

            # Redirect the entire trial's output to raw_log_path.
            import time as _time_mod
            _trial_t0 = _time_mod.time()
            try:
                print(
                    "[batch] trial %d/%d -> %s (ROW_SELECT_SEED=%d, ATTACK_SEED=%d)"
                    % (trial_idx, total_trials, raw_log_path, cur_row_seed, cur_att_seed),
                    flush=True,
                )
                with open(raw_log_path, "w") as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        main()
            except Exception as e:
                with open(raw_log_path, "a") as f:
                    f.write("\n[trial exception] %s: %s\n" % (type(e).__name__, str(e)))
                    import traceback as _tb
                    _tb.print_exc(file=f)
            _trial_elapsed = _time_mod.time() - _trial_t0
            with open(raw_log_path, "a") as f:
                f.write("\n[total_elapsed] %.1fs\n" % _trial_elapsed)

            # Extract NFound numbers from the raw log and write them into the summary.
            try:
                with open(raw_log_path, "r") as f:
                    raw_text = f.read()
                nfound_list = _extract_nfound_numbers(raw_text)
            except Exception:
                nfound_list = []

            if nfound_list:
                for _nf in nfound_list:
                    print(
                        ">>> NS NFound(rows vs X.T)= %s / %s  [%.1fs]"
                        % (_nf, batch_size_arg, _trial_elapsed),
                        flush=True,
                    )
            else:
                print(
                    ">>> NFound: (no result extracted from the log)  [%.1fs]" % _trial_elapsed,
                    flush=True,
                )

            with open(nfound_summary_path, "a") as sf:
                # Expected format (ns-dominant): nfound_list has exactly one element, and success_rate = nfound / batch_size.
                if len(nfound_list) == 0:
                    nfound_value = 0
                elif len(nfound_list) == 1:
                    nfound_value = int(nfound_list[0])
                else:
                    # Multi-valued (likely from statistical's multi-column output): take the mean to represent the "hit ratio".
                    nfound_value = sum(int(x) for x in nfound_list) / float(len(nfound_list))

                success_rate = float(nfound_value) / float(batch_size_arg)
                _tstr = __import__("time").ctime()
                summary_line = (
                    "batch_size(n)=%d %s m=%d rowseed=%d attseed=%d nfound=%s success_rate=%.2f elapsed=%.1fs"
                    % (
                        int(batch_size_arg),
                        _tstr,
                        int(target_rows_for_name),
                        cur_row_seed,
                        cur_att_seed,
                        (str(int(nfound_value)) if isinstance(nfound_value, (int,)) else str(nfound_value)),
                        success_rate,
                        _trial_elapsed,
                    )
                )

                # Use a file lock to serialize updates under multi-process concurrency, avoiding cross-writes / reordering.
                with open(nfound_lock_path, "w") as lockf:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
                    try:
                        if os.path.isfile(nfound_summary_path):
                            with open(nfound_summary_path, "r") as rf:
                                lines = rf.readlines()
                        else:
                            lines = []

                        grouped = {}  # batch_size(int) -> [lines...]
                        other = []  # lines whose format cannot be recognized
                        pat = re.compile(r"^batch_size\(n\)=([0-9]+)\s+")
                        header_pat = re.compile(r"^===== Batch_size\(n\)=([0-9]+) =====$")
                        cur_bn = None
                        for ln in lines:
                            stripped = ln.strip()
                            if stripped == "":
                                continue

                            hm = header_pat.match(stripped)
                            if hm:
                                cur_bn = int(hm.group(1))
                                grouped.setdefault(cur_bn, [])
                                continue

                            m = pat.match(stripped)
                            if m:
                                bn = int(m.group(1))
                                remainder = stripped[m.end() :].strip()
                                if remainder:
                                    grouped.setdefault(bn, []).append(remainder)
                                cur_bn = bn
                                continue

                            # A "time line" under an existing group header is assigned directly to the current batch_size.
                            if cur_bn is not None and not stripped.startswith("====="):
                                grouped.setdefault(cur_bn, []).append(stripped)
                                continue

                            if stripped.startswith("====="):
                                continue
                            other.append(stripped)

                        # Append the current trial's new record into the appropriate batch_size group.
                        m_new = pat.match(summary_line)
                        if m_new:
                            bn_new = int(m_new.group(1))
                            rem_new = summary_line[m_new.end() :].strip()
                            grouped.setdefault(bn_new, []).append(rem_new)
                        else:
                            other.append(summary_line)

                        out_lines = []
                        for bn in sorted(grouped.keys()):
                            out_lines.append("===== Batch_size(n)=%d =====" % bn)
                            out_lines.extend(grouped[bn])
                            out_lines.append("")
                        if other:
                            out_lines.append("===== Other =====")
                            out_lines.extend(other)
                            out_lines.append("")

                        with open(nfound_summary_path, "w") as wf:
                            wf.write("\n".join(out_lines).rstrip() + "\n")
                    finally:
                        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
except Exception as _e:
    import traceback
    print("\n*** run_mhlcp_cnn_quant_x exception (batch log wrapper stage) ***", flush=True)
    print(repr(_e), flush=True)
    traceback.print_exc()
    sys.exit(1)
