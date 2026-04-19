# Gradient Inversion via Lattice Attacks on Hidden Structure Problems

Official implementation of:

> Qiongxiu Li, Lixia Luo, Agnese Gini, Changlong Ji, Zhanhao Hu, Xiao Li,
> Chengfang Fang, Jie Shi, Xiaolin Hu.
> **Perfect Gradient Inversion in Federated Learning: A New Paradigm from
> the Hidden Subset Sum Problem.**
> *arXiv preprint arXiv:2409.14260, 2024.* [[Paper]](https://arxiv.org/abs/2409.14260)

This repository implements lattice-based attacks on the **Hidden Subset Sum Problem (HSSP)**, the **Hidden Linear Congruential Problem (HLCP)** and their matrix extension (**mHLCP**), and uses them to mount **gradient inversion attacks** against the FC layer of a CNN trained in a federated-learning setting.

The high-level pipeline is:

```
CNN training batch
      │
      ▼
FC layer forward pass
G = (1/n) · Δ^T · fc_input            ← FC weight gradient
      │
      ▼
quantize FC tensors to integers
H = X · A  (mod x0)                   ← matrix HLCP instance
      │
      ▼
Step 1 : orthogonal lattice + LLL
Step 2 : Nguyen–Stern / BKZ / FastICA
      │
      ▼
recover X (rows of fc1 pre-ReLU output)
      │
      ▼
feature-aware GIA  →  PSNR / SSIM / FID
```

## Project structure

```
.
├── lattice/                            # Sage : lattice theory & attacks
│   ├── building.sage                   #   parameter & instance generation
│   ├── ortho_attack.sage               #   orthogonal lattice, kernelLLL
│   ├── ns.sage                         #   Nguyen-Stern Step2, BKZ recovery
│   ├── multi.sage                      #   eigenspace / multivariate attack
│   ├── statistical.sage                #   FastICA-based statistical attack
│   ├── hssp.sage                       #   vector HSSP interface
│   ├── hlcp.sage                       #   vector HLCP interface (bounded)
│   ├── multi_dim_hssp.sage             #   matrix mHLCP (core implementation)
│   ├── extendhssp.sage                 #   alternative matrix extension (reference)
│   ├── test_hssp.sage                  #   smoke test : vector HSSP
│   ├── test_hlcp.sage                  #   smoke test : vector HLCP
│   ├── run_mhlcp_random.sage           #   single random mHLCP instance
│   ├── run_mhlcp_random_batch.sage     #   batch random mHLCP (30 seeds)
│   ├── run_stat_random_hlcp.sage       #   random HLCP via FastICA (statistical)
│   ├── run_mhlcp_cnn_quant_x.sage      #   ★ main forward attack : CNN tensors → mHLCP
│   ├── run_mhlcp_transpose.sage        #   alternative transpose attack (X = backprop_masked)
│   └── run_cnn_x_ns_stat.sage          #   NS vs statistical on CNN tensors
│
├── src/                                # Python : CNN models & GIA pipeline
│   ├── cnn_model.py                    #   BasicCNN architecture
│   ├── cnn_data.py                     #   dataset loaders (MNIST / CIFAR-10 / CIFAR-100)
│   ├── cnn_train.py                    #   training & evaluation loops
│   ├── cnn_metrics.py                  #   PSNR, SSIM, FID
│   ├── cnn_gia.py                      #   FC tensor export utilities
│   ├── cnn.py                          #   ★ main entry : train / fc_hssp / gia / gia_unknown
│   └── mlp_hlcp.py                     #   MLP variant for HLCP study
│
├── analysis/                           # publication-quality figures
│   ├── plot_metrics.py                 #   PSNR / SSIM / FID curves from sweep NPZ
│   ├── add_hlcp_gia_to_npz.py          #   weight-mix HLCP+GIA into existing sweeps
│   ├── plot_nfound_max_success_rate.py #   smoothed success-rate curve
│   └── gen_cifar_txt_and_plot.py       #   generate / transform success-rate text
│
├── scripts/                            # log aggregation utilities
│   ├── summarize_time_and_success.py   #   (n, m, attack) → mean success / time
│   ├── summarize_scaling.py            #   CNN_INT_SCALE sweep summary
│   ├── summarize_random_dense.py       #   random-X density sweep summary
│   ├── aggregate_subsample.py          #   transpose-attack sweep aggregator (random / bias)
│   ├── plot_scaling.py                 #   dual-Y scaling plot
│   └── plot_nfound_hist.py             #   nfound histogram
│
├── experiments/                        # reproducible experiment drivers (seeded)
│   ├── run_exp12_part1.sh              #   CIFAR-10  GIA/gia_unknown sweep, bs 1..128
│   ├── run_exp12_part2.sh              #   CIFAR-100 GIA/gia_unknown sweep, bs 1..128
│   ├── run_gia_baseline.py             #   CIFAR-10 GIA baseline (SEED=2026)
│   ├── run_gia_cifar100_{distinct,improved,methods}.py  # CIFAR-100 GIA variants
│   ├── run_gia_clamp_vs_hlcp{,_bs40,_mixed,_unknown}.py # clamp vs HLCP ablation
│   ├── run_pool_{known_mixed,unknown_mixed,unknown_same}.py  # pool ablations
│   ├── run_all_recon_compare.py        #   full reconstruction comparison
│   ├── run_hlcp_failure_patch.py       #   simulate HLCP failure + patch
│   ├── launch_all_recon.sh             #   fan out run_all_recon_compare.py (96 exps)
│   ├── bench_real_cnn_v2.sage          #   real-data lattice timing benchmark
│   └── bench_time_m3n10.sage           #   synthetic lattice timing benchmark
│
├── docs/
│   ├── architecture.md                 #   layered architecture guide
│   └── project_map.md                  #   detailed project map & experiment recipes
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- **SageMath** ≥ 9.0 (for lattice computations)
- **Python** ≥ 3.8
- **PyTorch** ≥ 1.9.0
- **fpylll** (bundled with SageMath)
- **scikit-learn** ≥ 0.24 (for the FastICA path)
- **matplotlib** ≥ 3.3 (for plotting)
- **scipy** ≥ 1.6 (for success-rate smoothing, optional)

### Installing SageMath via Conda (recommended)

```bash
# 1. Install Miniconda (skip if already installed)
#    macOS (Apple Silicon):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
#    Linux:
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. Create a Sage environment
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n sage sage python=3.9
conda activate sage

# 3. Verify
sage --version
```

### Installing Python dependencies

```bash
# inside the sage conda environment, or in a separate Python venv:
pip install -r requirements.txt
```

## Quick start

All commands assume you are at the repository root.

### 1. Lattice smoke tests (no CNN data required)

```bash
# Vector HSSP / HLCP (Gini's original tests)
sage lattice/test_hssp.sage
sage lattice/test_hlcp.sage

# Matrix mHLCP, single random instance
sage lattice/run_mhlcp_random.sage

# Matrix mHLCP, 30 random seeds
sage lattice/run_mhlcp_random_batch.sage

# Random HLCP attacked by the FastICA / statistical path
sage lattice/run_stat_random_hlcp.sage
```

### 2. Export CNN FC-layer tensors

```bash
python src/cnn.py --mode fc_hssp --batch_size 10 --dataset cifar10
```

This writes
`expdata/fc_hssp/cifar10/bs10/{relu_mask_fc.npy, fc1_pre_relu_fc.npy, fc_input_batch.npy, fc1_weight_grad.npy}`.

### 3. CNN tensor → mHLCP attack

#### 3.a Forward attack (the main pipeline)

`X = quantized fc1_pre_relu`. This is the canonical attack used in the paper.

```bash
sage lattice/run_mhlcp_cnn_quant_x.sage
```

By default this uses `BATCH_SIZE=20`, so first run

```bash
python src/cnn.py --mode fc_hssp --batch_size 20 --dataset cifar10
```

If the required NPYs are missing, the Sage script will auto-invoke `src/cnn.py` to regenerate them.

#### 3.b Transpose attack (alternative)

`X = quantized backprop_masked_fc` (the FC1 weight gradient's upstream factor),
`A = quantized fc_input_batch`. Two row-selection strategies are exposed:

- `MHLCP_ROW_MODE=random` (**default, the main row-selection method**) — pure
  permutation seeded by `ROWSEED`.
- `MHLCP_ROW_MODE=bias` (**exploration alternative**) — sort rows by `|∇b_1|`,
  the FC1 bias gradient (attacker-observable).

```bash
# random row selection (default)
BS=20 sage lattice/run_mhlcp_transpose.sage

# bias-gradient row selection (exploration)
BS=20 MHLCP_ROW_MODE=bias sage lattice/run_mhlcp_transpose.sage
```

The script reads from `expdata/fc_hssp/cifar10/bs<BS>/{backprop_masked_fc,fc_input_batch}.npy`
(or `bs<BS>/run_<RUN>/...` if `HSSP_RUN_ID=<RUN>` was set during FC export). Each
trial writes one `RESULT|...` line to stdout; aggregate many trials with
`scripts/aggregate_subsample.py` (see `docs/project_map.md` §7.7).

### 4. Gradient Inversion Attack (GIA)

```bash
# Known labels
python src/cnn.py --mode gia --dataset cifar10 --batch_size 10

# Unknown labels (label is also optimised)
python src/cnn.py --mode gia_unknown --dataset cifar10 --batch_size 10

# Full sweep over batch sizes, in parallel
python src/cnn.py \
    --mode gia \
    --dataset cifar10 \
    --pool \
    --n_runs 10 \
    --n_jobs 20 \
    --batch_size_sweep 1,2,3,4,5,6,7,8,9,10
```

Available `--mode` values:

| mode             | meaning                                                      |
|------------------|--------------------------------------------------------------|
| `train`          | regular CNN training                                         |
| `fc_hssp`        | export FC tensors only (input to the lattice attack)         |
| `gia`            | known-label GIA (DLG-style, with optional FC feature prior)  |
| `gia_unknown`    | unknown-label GIA (label is optimised jointly)               |
| `gia_loss_probe` | record initial / final ‖∇‖² and λ·L_feat across batch sizes |
| `gia_norm`       | normalised objective `L_grad/L_grad₀ + λ·L_feat/L_feat₀`     |

## Configuration

### Environment variables for the Sage CNN bridge

| variable                  | default | description                                                   |
|---------------------------|---------|---------------------------------------------------------------|
| `MHLCP_TARGET_ROWS`       | `150`   | number of sample rows `m` taken from `fc1_pre_relu_fc`        |
| `CNN_INT_SCALE`           | `1000`  | float-to-integer quantization scale                           |
| `MHLCP_ATTACK`            | `ns`    | `ns`, `statistical`, or `both`                                |
| `MHLCP_X_SOURCE`          | `real`  | `real` (CNN) or `random` (synthetic X)                        |
| `MHLCP_A_SOURCE`          | `real`  | `real` (FC input) or `random`                                 |
| `MHLCP_B_X`               | `1`     | upper bound on entries of synthetic X                         |
| `MHLCP_B_A`               | `100`   | upper bound on entries of synthetic A                         |
| `MHLCP_A_DISCRETE`        | unset   | set to `1` to use discrete `{0..B_A}` entries for A           |
| `MHLCP_L`                 | `n`     | number of columns of A (defaults to batch_size)               |
| `MHLCP_ROW_MODE`          | `random`| `random` or `relu_stratified` (Hamming-weight selection)      |
| `MHLCP_ROW_SELECT_SEED`   | `0`     | seed for X row sampling (parallel-friendly)                   |
| `MHLCP_ATTACK_SEED`       | `1`     | seed for A column sampling and attack RNG                     |
| `MHLCP_RECOVER_BOX_MAX`   | `220000`| upper bound passed to `recoverBox`                            |
| `MHLCP_DATASET`           | `cifar10` | dataset for auto-generating FC tensors                      |
| `MHLCP_REPO_ROOT`         | (auto)  | override repo-root detection (useful when Sage runs from `/tmp`) |

### Selected `cnn.py` flags (GIA modes)

| flag               | meaning                                                                  |
|--------------------|--------------------------------------------------------------------------|
| `--batch_size`     | single batch size                                                        |
| `--batch_size_sweep` | comma list (`1,2,5`) or range (`1,...,50`) — one set of runs per batch  |
| `--n_runs`         | repetitions per batch size                                               |
| `--n_jobs`         | parallel worker count                                                    |
| `--pool`           | use a randomly pooled batch (mixed labels)                               |
| `--same_label`     | all images in the batch share one CIFAR-10 class                         |
| `--label_idx`      | fix the class index when `--same_label` is on                            |
| `--save_fig`       | write originals + reconstructions to `expdata/.../`                      |
| `--known_rate`     | proportion of FC features assumed exactly known                          |
| `--known_residual` | use the (true − partial) residual as a soft prior                        |
| `--known_peel`     | strip known rows from the gradient term (unknown-label only)             |
| `--noise_on_fc`    | σ for Gaussian noise added to FC features (unknown-label only)           |

See `docs/project_map.md` for full experiment recipes (success-rate sweeps, scaling studies, mixing rules) and `docs/architecture.md` for the layered architecture.

## Citation

```bibtex
@article{li2024perfect,
  title   = {Perfect Gradient Inversion in Federated Learning:
             A New Paradigm from the Hidden Subset Sum Problem},
  author  = {Li, Qiongxiu and Luo, Lixia and Gini, Agnese and Ji, Changlong
             and Hu, Zhanhao and Li, Xiao and Fang, Chengfang and Shi, Jie
             and Hu, Xiaolin},
  journal = {arXiv preprint arXiv:2409.14260},
  year    = {2024}
}
```

## Acknowledgement

The `lattice/` core (HSSP / HLCP / mHLCP attack stack) builds on the open-source implementation released with Agnese Gini's prior work on the Hidden Subset Sum Problem
([github.com/agnesegini/solving_hssp](https://github.com/agnesegini/solving_hssp)).
The CNN bridge, GIA pipeline and the matrix-mHLCP extension on top are contributed by this work.

## License

MIT License. See [LICENSE](LICENSE).
