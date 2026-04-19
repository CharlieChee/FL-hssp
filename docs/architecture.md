# Architecture Guide

This document explains how the codebase is layered. For a higher-level project map (with experiment recipes, symbol tables and parameter sweeps) see [project_map.md](project_map.md).

## Layered view

```
Layer A : Lattice theory          (lattice/*.sage)
Layer B : CNN feature export      (src/cnn.py --mode fc_hssp)
Layer C : CNN → mHLCP bridge      (lattice/run_mhlcp_cnn_quant_x.sage)
Layer D : GIA evaluation          (src/cnn.py --mode gia / gia_unknown / ...)
Layer E : Analysis & plotting     (analysis/ + scripts/)
```

## Layer A — Lattice theory (Sage)

Core implementations of HSSP / HLCP / mHLCP and the lattice-based attacks against them.

### Problem definitions

| problem | equation                          | unknowns                                |
|---------|-----------------------------------|-----------------------------------------|
| HSSP    | `b ≡ X · a (mod x0)`              | sample matrix `X` (m×n) and `a` (n)     |
| HLCP    | HSSP with bounded entries `\|X_{ij}\| ≤ B` | same as HSSP                       |
| mHLCP   | `H ≡ X · A (mod x0)`              | `X` (m×n) and `A` (n×l)                 |

### Attack pipeline

**Step 1 — orthogonal lattice.** Build a lattice `L` whose rows satisfy `v · H ≡ 0 (mod x0)`. After LLL reduction the first `m − n` rows lie in the right kernel of `X` (up to scaling). Implemented in `ortho_attack.sage` (vector) and inside the `Step1_Mat` of `multi_dim_hssp.sage` (matrix).

**Step 2 — recovery.** From the kernel basis recover individual columns of `X` via one of:

- **Nguyen–Stern (NS)** — `ns.sage`. Iterated BKZ + bounded enumeration in `recoverBox`/`recoverBinary`.
- **Eigenspace / multivariate** — `multi.sage`. Uses a mod-3 kernel and eigenvalue splitting.
- **Statistical (FastICA)** — `statistical.sage`. Treats the columns of `X` as independent sources.

### Sage load chain

```
building.sage   ─┐
ortho_attack    ─┤
multi.sage      ─┼─► hssp.sage  ──► hlcp.sage
ns.sage         ─┘        │
                          └─► multi_dim_hssp.sage
                                     │
                          ┌──────────┼─────────────┐
                          ▼          ▼             ▼
              run_mhlcp_cnn_quant_x  run_mhlcp_random   run_mhlcp_random_batch
```

`run_stat_random_hlcp.sage` and `run_cnn_x_ns_stat.sage` load `hlcp.sage` + `statistical.sage` directly.

`extendhssp.sage` is an earlier matrix extension by the Gini line of work, kept as a reference; the active matrix code lives in `multi_dim_hssp.sage`.

## Layer B — CNN feature export (Python)

`src/cnn.py --mode fc_hssp` loads (or trains) `BasicCNN` from `src/cnn_model.py` and saves the FC-layer intermediate tensors:

| tensor               | shape                | meaning                                           |
|----------------------|----------------------|---------------------------------------------------|
| `fc_input_batch.npy` | `(batch_size, D)`    | flattened conv feature feeding into FC1           |
| `fc1_pre_relu_fc.npy`| `(1000, batch_size)` | FC1 linear output (before ReLU), transposed      |
| `relu_mask_fc.npy`   | `(1000, batch_size)` | binary ReLU mask, transposed                     |
| `fc1_weight_grad.npy`| `(1000, D)`          | weight gradient `(1/n) · Δ^T · fc_input`         |

Output goes to `expdata/fc_hssp/<dataset>/bs<n>/`.

## Layer C — CNN → mHLCP bridge (Sage)

Two parallel bridges are provided:

### C.1 Forward attack — `lattice/run_mhlcp_cnn_quant_x.sage` (the main pipeline)

1. Load the FC tensors saved by Layer B.
2. Quantize floats to integers: `X_int = floor(fc1_pre_relu_fc · CNN_INT_SCALE)`.
3. Select `m` rows (`MHLCP_TARGET_ROWS`) keeping `rank(X) = n`.
4. Construct an mHLCP instance `H = X · A (mod x0)`.
5. Run `hssp_attack(H, "ns")` (or `"statistical"`).
6. Report **NFound** = number of recovered rows that match a row of `X^T`.

Important environment variables: `MHLCP_TARGET_ROWS`, `CNN_INT_SCALE`, `MHLCP_X_SOURCE`, `MHLCP_A_SOURCE`, `MHLCP_ATTACK`, `MHLCP_ROW_SELECT_SEED`, `MHLCP_ATTACK_SEED`. See [project_map.md](project_map.md) for the full table and recipes.

### C.2 Transpose attack — `lattice/run_mhlcp_transpose.sage` (alternative)

Same Step1 / Step2 lattice machinery, but builds the instance from
`X = backprop_masked_fc` (the gradient backprop factor) instead of `fc1_pre_relu`.

Row selection is the only knob; both modes share `A = fc_input_batch`:

- `MHLCP_ROW_MODE=random` (default, **the main method** for this pipeline) —
  random permutation, seeded by `ROWSEED`.
- `MHLCP_ROW_MODE=bias` (**exploration**) — sort rows by `|∇b_1|`
  (attacker-observable FC1 bias gradient).

See [project_map.md §7.7](project_map.md#77-transpose-attack-alternative-pipeline) for usage and aggregation.

## Layer D — Gradient Inversion (Python)

`src/cnn.py` evaluates GIA in several modes:

- `gia` — labels are known, optimise images by gradient matching. Compares **naive GIA** (gradient term only) vs **HSSP + GIA** (gradient + FC feature matching).
- `gia_unknown` — labels are unknown and optimised jointly with the images. Adds partial-knowledge variants when `0 < known_rate < 1`.
- `gia_loss_probe` — record `‖∇‖²` and `λ · L_feat` over batch sizes (initial and final values).
- `gia_norm` — same as `gia` but with normalised objective `L_grad/L_grad₀ + λ · L_feat/L_feat₀`.

Per-image metrics: PSNR, SSIM, FID (FID disabled by default because it brings in a large InceptionV3 weight).

## Layer E — Analysis & plotting

| script                                           | purpose                                                             |
|--------------------------------------------------|----------------------------------------------------------------------|
| `analysis/plot_metrics.py`                       | render PSNR / SSIM / FID figures from sweep NPZ                      |
| `analysis/add_hlcp_gia_to_npz.py`                | mix `HSSP + GIA` and `naive GIA` curves by an external success-rate weight `w(n)` |
| `analysis/plot_nfound_max_success_rate.py`       | extract per-batch max success rate from `logs/nfound.log` and smooth |
| `analysis/gen_cifar_txt_and_plot.py`             | post-processing on success-rate text                                |
| `scripts/summarize_time_and_success.py`          | aggregate `trial_*.log` by `(n, m, attack)`                          |
| `scripts/summarize_scaling.py`                   | aggregate `CNN_INT_SCALE` sweep                                      |
| `scripts/summarize_random_dense.py`              | aggregate random-X density sweep                                     |
| `scripts/plot_scaling.py`                        | plot scaling sweep (dual-Y axes)                                     |
| `scripts/plot_nfound_hist.py`                    | nfound histogram from log files                                      |

Note: the **`HLCP + GIA`** curve produced by `add_hlcp_gia_to_npz.py` is not a per-trial closed-loop attack. It is a success-rate-weighted mix between `HSSP + GIA` (assuming oracle access) and `naive GIA`. Treat it as a calibrated upper-bound estimate, not a raw measurement.

## Data flow

```
training batch (images, labels)
        │
        ▼
forward pass ─►  fc_input,  fc1_pre_relu,  relu_mask,  ∂L/∂W_FC
        │
        ├── Layer B : export NPYs ──────────────┐
        │                                       │
        │   Layer C : quantize → mHLCP          │
        │                │                      │
        │   Layer A : NS / BKZ / FastICA        │
        │                │                      │
        │   recovered rows of X                 │
        │                │                      │
        │   Layer D : feature-aware GIA  ◄──────┘
        │                │
        │   reconstructed images
        │
        ▼
Layer E : aggregate, plot (PSNR / SSIM / FID / success rate)
```
