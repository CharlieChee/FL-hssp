#!/usr/bin/env sage
r"""
Matrix HLCP (multi_dim_hssp): B = X * A (mod x0); X is m x n full-rank with entries in
[0, B_X] (this script uses B_X=100).

Runs hssp_attack(H, 'ns') (Step1_Mat + Step2_BK_mat) over random seeds until the leading
l x l block of B is invertible mod x0 and, where possible, NFound==n.

Usage (either from the repo root or from the lattice/ subdirectory)::

  sage lattice/run_mhlcp_random.sage
  cd lattice && sage run_mhlcp_random.sage
"""
import os, sys

# Self-locate: chdir to the lattice/ directory so internal load() calls resolve
_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
if os.path.isfile(os.path.join(_d, "multi_dim_hssp.sage")):
    os.chdir(_d)
elif os.path.isfile("lattice/multi_dim_hssp.sage"):
    os.chdir("lattice")
load("multi_dim_hssp.sage")

# Small parameters keep BKZ tractable; B_X=100 requires a sufficiently large x0
# (the default iota formula gives only ~11 bits at n=5, which degenerates the instance)
N, L, M = 5, 3, 25
B_X = 100
NX0_BITS = 96


def ok_instance(H):
    l = H.l
    try:
        r = rank(Matrix(Integers(H.x0), H.B[:l, :l]))
        return r >= l
    except Exception:
        return False


def main():
    for seed in range(200):
        set_random_seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        H = multi_hssp(N, L, kappa=-1, B_X=B_X, nx0_bits=NX0_BITS)
        H.gen_instance(m=M)
        if not ok_instance(H):
            continue
        print("seed=%s  n=%s l=%s m=%s  B_X=%s  x0.nbits()=%s" % (seed, N, L, M, B_X, H.x0.nbits()))
        print("  rank(X)=", H.X.rank(), "  B shape=", H.B.dimensions())
        try:
            out = hssp_attack(H, "ns")
        except Exception as e:
            print("  hssp_attack exception:", type(e).__name__, e)
            continue
        if out is None or out == 0:
            continue
        if not hasattr(out, "nrows"):
            print("  returned:", type(out), out)
            continue
        MB = out
        Xrows = [tuple(r) for r in H.X.T.rows()]
        Xset = set(Xrows)
        nfound = sum(1 for r in MB.rows() if tuple(r) in Xset)
        print("  >>> script check NFound(rows match X.T)=", nfound, "/", N)
        if nfound == N:
            print("OK: matrix mHLCP (multi_dim_hssp, alg=ns) succeeded; recovered n rows of X.")
            print("\n--- secret matrix A for this instance (n x l = %s x %s), entries mod x0 ---" % (H.A.nrows(), H.A.ncols()))
            print(H.A)
            print("\n--- X for this instance (m x n = %s x %s), entries in [0, B_X], B = X*A mod x0 ---" % (H.X.nrows(), H.X.ncols()))
            print(H.X)
            print("\n--- x0 (first 200 characters) ---")
            xs = str(H.x0)
            print(xs if len(xs) <= 200 else xs[:200] + "...")
            return
    print("Did not reach NFound==n within 200 seeds; consider increasing M / nx0 or relaxing parameters.")


main()
