#!/usr/bin/env sage
r"""
mHLCP (batch of random instances): X is m x n (default 100 x 10) with entries in [0,100];
A is n x l (default 10 x 10) with entries in [0,100]; B = X * A (mod x0).
Runs hssp_attack(H, 'ns') over 30 seeds.

Usage (from the repo root or the lattice/ subdirectory)::

  sage lattice/run_mhlcp_random_batch.sage
  cd lattice && sage run_mhlcp_random_batch.sage
"""
import os, sys

_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
if os.path.isfile(os.path.join(_d, "multi_dim_hssp.sage")):
    os.chdir(_d)
elif os.path.isfile("lattice/multi_dim_hssp.sage"):
    os.chdir("lattice")
load("multi_dim_hssp.sage")

N, L, M = 10, 10, 100
B_X = 100
B_A = 100
NX0_BITS = 128
# recoverBox needs a larger cap when n=10
RECOVER_BOX_MAX = 80000
MAX_SEED = 30


def ok_instance(H):
    l = H.l
    try:
        r = rank(Matrix(Integers(H.x0), H.B[:l, :l]))
        return r >= l
    except Exception:
        return False


def main():
    for seed in range(MAX_SEED):
        set_random_seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        H = multi_hssp(
            N,
            L,
            kappa=-1,
            B_X=B_X,
            B_A=B_A,
            nx0_bits=NX0_BITS,
            recover_box_max=RECOVER_BOX_MAX,
        )
        H.gen_instance(m=M)
        if not ok_instance(H):
            print("seed=%s skipped: B[:l,:l] is not invertible" % seed)
            continue
        print(
            "seed=%s  n=%s l=%s m=%s  B_X=%s B_A=%s  x0.nbits()=%s"
            % (seed, N, L, M, B_X, B_A, H.x0.nbits())
        )
        print("  rank(X)=", H.X.rank())
        try:
            out = hssp_attack(H, "ns")
        except Exception as e:
            print("  hssp_attack exception:", type(e).__name__, e)
            continue
        if out is None or out == 0:
            print("  Step2 produced no valid MB")
            continue
        if not hasattr(out, "nrows"):
            print("  returned:", type(out), out)
            continue
        MB = out
        Xrows = [tuple(r) for r in H.X.T.rows()]
        Xset = set(Xrows)
        nfound = sum(1 for r in MB.rows() if tuple(r) in Xset)
        print("  >>> NFound(rows match X.T)=", nfound, "/", N)
        if nfound == N:
            print("OK: 100x10 / 10x10 instance succeeded.")
            return
    print("Did not reach NFound==n within %s seeds; consider increasing M, NX0_BITS, RECOVER_BOX_MAX, or MAX_SEED." % MAX_SEED)


main()
