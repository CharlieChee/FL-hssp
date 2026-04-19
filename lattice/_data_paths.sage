#!/usr/bin/env sage
"""Locate the FC HSSP NPY files exported by ``src/cnn.py --mode fc_hssp``.

These resolvers tolerate several legacy directory layouts so that older
``expdata/`` trees keep working. They expect the current working directory to
be the repository root (``run_mhlcp_cnn_quant_x.sage`` chdirs there before
``load()``-ing this file).
"""
import os

# ``BATCH_SIZE`` is set by the parent script; we reference it indirectly via
# the function arguments below, so no module-level state is needed here.

def resolve_relu_path(batch_size=10):
    relu_path = "expdata/fc_hssp/relu_mask_fc_bs%s.npy" % batch_size
    alt_relu = "expdata/fc_hssp/cifar10/bs%s/relu_mask_fc.npy" % batch_size
    alt_legacy_relu = "expdata/fc_hssp/cifar10/relu_mask_fc.npy"
    if os.path.exists(relu_path):
        return relu_path
    if os.path.exists(alt_relu):
        return alt_relu
    if batch_size == 10 and os.path.exists(alt_legacy_relu):
        return alt_legacy_relu
    alt_legacy_bs = "expdata/fc_hssp/cifar10/bs%s/relu_mask_fc.npy" % batch_size
    if os.path.exists(alt_legacy_bs):
        return alt_legacy_bs
    raise FileNotFoundError(
        "relu_mask_fc.npy not found. From the project root, run: python src/cnn.py --mode fc_hssp --batch_size %s"
        % batch_size
    )


def resolve_pre_relu_path(relu_path):
    d = os.path.dirname(os.path.abspath(relu_path))
    p = os.path.join(d, "fc1_pre_relu_fc.npy")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(
        "fc1_pre_relu_fc.npy not found (in the same directory as %s). Run: python src/cnn.py --mode fc_hssp --batch_size %s"
        % (relu_path, BATCH_SIZE)
    )


def resolve_pre_relu_standalone(batch_size=10):
    """When the relu file is not required, locate the pre-ReLU npy directly."""
    cands = [
        "expdata/fc_hssp/cifar10/bs%s/fc1_pre_relu_fc.npy" % batch_size,
    ]
    for p in cands:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "fc1_pre_relu_fc.npy not found. Run: python src/cnn.py --mode fc_hssp --batch_size %s" % batch_size
    )


def resolve_fc_input_path(batch_size=10):
    cands = [
        "expdata/fc_hssp/cifar10/bs%s/fc_input_batch.npy" % batch_size,
        "expdata/fc_hssp/fc_input_batch_bs%s.npy" % batch_size,
        "expdata/fc_hssp/cifar10/fc_input_batch.npy",
    ]
    for p in cands:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "fc_input_batch.npy not found. Run: python src/cnn.py --mode fc_hssp --batch_size %s" % batch_size
    )


