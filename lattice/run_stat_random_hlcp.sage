#!/usr/bin/env sage
r"""
Run the statistical pipeline on random HLCP instances: genParams + Step1 +
statistical_1 + statistical_2. The statistical(...) helper invoked from
hlcp_attack in hlcp.sage is undefined, so we wire statistical_1/2 together here.
Requires sklearn (FastICA); intermediate products (ICA unmixing reference) are
written under ``lattice/ICA/``.

Default B=1 (sample-matrix entries in 0/1): ICA and subsequent inversion succeed
fairly easily, typically reaching nrafound=n within a few dozen tries. With B=100
the pipeline usually completes but nrafound is often 0 (the statistical route is
fragile at large scale).

Usage (from the repo root or the lattice/ subdirectory)::

  sage lattice/run_stat_random_hlcp.sage
  cd lattice && sage run_stat_random_hlcp.sage
"""
import os
import sys
import math
from time import time as wall_time

_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
if os.path.isfile(os.path.join(_d, "hlcp.sage")):
    os.chdir(_d)
elif os.path.isfile("lattice/hlcp.sage"):
    os.chdir("lattice")

os.makedirs("ICA", exist_ok=True)

try:
    from sklearn.decomposition import FastICA
except ImportError as e:
    raise RuntimeError("requires sklearn: conda/pip install scikit-learn") from e

load("hlcp.sage")
load("statistical.sage")

globals()["FastICA"] = FastICA
globals()["math"] = math
# statistical.sage ICA uses time(); do not bind it to the time module, or Step1's time() will break
globals()["time"] = wall_time


def statistical(MO, n, m, x0, X, a, b, kappa, B):
    MOn, MO_lll = statistical_1(MO, n, m, x0, X, a, b, kappa, B)
    tica, tt2, nrafound, _nf = statistical_2(
        MOn, MO_lll, n, m, x0, X, a, b, kappa, B
    )
    return tica, tt2, nrafound


def main():
    n, m = 10, 100
    kappa = -1
    # B=1: {0,1} samples, more consistent with the scale assumption of ICA inside statistical;
    # for B>100 the pipeline typically only "completes" but nrafound is often 0
    B = 1
    iota = 0.035
    nx0 = Integer(28)

    max_trials = 80
    for t in range(max_trials):
        set_random_seed(t)
        x0, a, X, b = genParams(n, m, nx0, kappa, B)
        print("trial", t + 1, "/", max_trials, " x0 nbits~", x0.nbits())

        MO, tt1, tt10, tt1O = Step1(n, kappa, x0, a, X, b, m, BKZ=True)
        if MO == -1:
            print("  Step1 timeout, retry")
            continue

        try:
            tica, tt2, nrafound = statistical(MO, n, m, x0, X, a, b, kappa, B)
        except Exception as e:
            print("  statistical exception:", type(e).__name__, e)
            continue

        print("  statistical done: nrafound =", nrafound, "/", n)
        if nrafound == n:
            print("OK: statistical recovered all n coefficients on a random instance.")
            return

    print("Did not reach nrafound==n within", max_trials, "trials; consider increasing max_trials or tuning nx0 (28~40).")


main()
