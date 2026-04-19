"""Shared helpers for experiments/*.py drivers.

Every experiment driver needs the same handful of utilities (seed pinning,
dataset un-normalization constants, image grid dumping, gradient losses).
Centralising them here keeps each experiment file focused on its own
configuration/logic and makes cross-experiment tweaks land in one place.

Any driver that imports from this module indirectly also gets ``src/`` on
``sys.path`` (via the small shim at the top of this file), so the usual
``from cnn import ...`` / ``from cnn_metrics import ...`` imports work even
when the driver is launched from the repository root as
``python experiments/<name>.py``.
"""
import os
import random
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# Put src/ on sys.path once, so drivers that `from _common import *` (or import
# anything from this module first) can then do `from cnn import ...`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def set_all_seeds(seed):
    """Pin ``random`` / ``numpy`` / ``torch`` (CPU and all CUDA devices) seeds.

    Use this at the top of every driver for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_norm_tensors(dataset_name, device):
    """Return ``(cfg, mean_t, std_t)`` for a dataset from ``cnn_data``.

    ``mean_t`` / ``std_t`` are shaped ``(1, C, 1, 1)`` so they broadcast over
    a ``(B, C, H, W)`` image tensor.
    """
    import cnn_data  # lazy: needs src/ on sys.path, which the shim above handles
    cfg = cnn_data.DATASET_CONFIG[dataset_name]
    mean_t = torch.tensor(cfg["mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(cfg["std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return cfg, mean_t, std_t


def total_variation(x):
    """Mean absolute horizontal + vertical differences (TV prior for GIA)."""
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dh.abs().mean() + dw.abs().mean()


def cosine_grad_loss(dummy_grads, true_grads):
    """Sum over layers of ``1 - cos(dummy_grad, true_grad)`` (Geiping et al., 2020)."""
    loss = 0.0
    for g_fake, g_true in zip(dummy_grads, true_grads):
        cos = F.cosine_similarity(
            g_fake.reshape(-1).unsqueeze(0),
            g_true.reshape(-1).unsqueeze(0),
        )
        loss = loss + (1.0 - cos)
    return loss


def mse_grad_loss(dummy_grads, true_grads):
    """Sum of per-layer MSE between reconstructed and true gradients (DLG)."""
    loss = 0.0
    for g_fake, g_true in zip(dummy_grads, true_grads):
        loss = loss + F.mse_loss(g_fake, g_true)
    return loss


def save_batch_img(tensor, path):
    """Dump a ``(B, C, H, W)`` image tensor as a single horizontal strip PNG."""
    imgs = tensor.detach().cpu().clamp(0, 1)
    n = imgs.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[i].axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
