# Project Map

A reference for what is in this repository, what each piece is used for, and how to reproduce the headline experiments. For the *layered* view see [architecture.md](architecture.md).

## 1. One-line summary

> Can the lattice attacks for HSSP / HLCP and their matrix extension recover the FC-layer features of a CNN from a single shared gradient, and how much does that recovery improve image reconstruction (GIA) over plain DLG?

This is a research codebase. Every script is meant to serve some part of that question, not to be a turnkey product.

## 2. Directory map

```
.
├── lattice/                  # Sage : lattice theory, attacks, run scripts
├── src/                      # Python : CNN model, GIA pipeline, FC export
├── experiments/              # Python / Sage / Bash : reproducible experiment drivers
├── analysis/                 # Python : publication-quality figures
├── scripts/                  # Python : log aggregation utilities
├── docs/                     # Markdown : this file + architecture.md
├── README.md
├── requirements.txt
└── LICENSE
```

## 3. Symbol table (theory ↔ code)

| symbol | role in theory                       | CNN mapping                                          | typical shape         |
|--------|---------------------------------------|------------------------------------------------------|-----------------------|
| `n`    | hidden dimension                      | usually `batch_size`; columns of `X`, rows of `A`   | scalar                |
| `m`    | number of samples / observations      | rows sampled from `fc1_pre_relu_fc`                  | `MHLCP_TARGET_ROWS`   |
| `l`    | columns of `A`                        | columns of `A`; defaults to `l = n`                  | `MHLCP_L`             |
| `x0`   | modulus                               | pseudo-prime generated per instance                  | `nx0_bits` long       |
| `X`    | hidden sample matrix                  | quantized `fc1_pre_relu_fc` rows                     | `m × n`               |
| `A`    | hidden coefficient matrix             | quantized `fc_input_batch` columns (real mode)       | `n × l`               |
| `H`/`B`| observed product                      | `H = X · A (mod x0)`                                 | `m × l`               |
| `B_X`  | upper bound on `\|X_{ij}\|`           | `MHLCP_B_X` (only for synthetic X)                   | scalar                |
| `B_A`  | upper bound on `\|A_{ij}\|`           | `MHLCP_B_A` (only for synthetic A)                   | scalar                |
| `real_feat` | not part of standard HSSP        | `model.last_feature` (FC input prior used by GIA)    | `batch_size × D`      |

Notes:

- `X` and `A` are *roles* assigned to CNN tensors so we can fit them into the mHLCP equation. They are not the standard FL notation — see comment in `run_mhlcp_cnn_quant_x.sage`.
- `relu_mask_fc` is `fc1_relu_mask.T`; older row-selection strategies use it.
- Success criterion in most scripts is **NFound = number of recovered rows that match a row of `X^T`**.

## 4. Attack pipeline (mHLCP)

```
quantize  →  H = X · A (mod x0)
           │
Step 1   ──┤  L = orthogonal lattice such that L · H ≡ 0 (mod x0)
           │  L.LLL()  →  short vectors lie in right kernel of X
           │
Step 2   ──┤  NS  : iterated BKZ until vectors live in {-1, 0, 1}^*
           │       then recoverBox / recoverBinary
           │  ICA : Step1 + statistical_1 (whiten) + statistical_2 (FastICA + integer round)
           │
Recover  ──┘  X  ≈  rows that hit X.T (count = NFound)
              A  ≈  (X^T X)^{-1} · X^T · H  (when needed)
```

## 5. Workflow recipes

All commands assume `cwd = repo root` and a Sage conda env where `sage --version` works.

### A. Theory smoke tests (no CNN data)

```bash
# Vector HSSP (Gini's original test). Expect "HSSP_n Test: success!".
sage lattice/test_hssp.sage

# Vector HLCP. Expect "HLCP_{n,B} Test: success!" and "HLCP_{n,B}^kappa Test: success!".
sage lattice/test_hlcp.sage

# Matrix mHLCP, single random instance (~1 minute on a laptop).
sage lattice/run_mhlcp_random.sage

# Matrix mHLCP, 30 random seeds with 100×10 X / 10×10 A.
sage lattice/run_mhlcp_random_batch.sage

# Random HLCP attacked by FastICA. Expect nrafound==n within ~80 trials.
sage lattice/run_stat_random_hlcp.sage
```

### B. Export CNN FC tensors

```bash
python src/cnn.py --mode fc_hssp --batch_size 10 --dataset cifar10
python src/cnn.py --mode fc_hssp --batch_size 20 --dataset cifar10
```

Outputs land in `expdata/fc_hssp/<dataset>/bs<n>/`.

### C. CNN → mHLCP attack

```bash
# Default: BATCH_SIZE=20, MHLCP_TARGET_ROWS=150, NS attack.
sage lattice/run_mhlcp_cnn_quant_x.sage

# A few common variations:
MHLCP_TARGET_ROWS=400 sage lattice/run_mhlcp_cnn_quant_x.sage
MHLCP_A_SOURCE=random MHLCP_TARGET_ROWS=20 sage lattice/run_mhlcp_cnn_quant_x.sage --batch_size 10
CNN_INT_SCALE=100 MHLCP_A_SOURCE=random MHLCP_TARGET_ROWS=20 \
    sage lattice/run_mhlcp_cnn_quant_x.sage --batch_size 10 --file scaling
```

### D. Time / success-rate sweeps

```bash
# Sweep TARGET_ROWS for batch_size=10 (random A).
for r in 70 75 80 85 90 95 100; do
  nohup env MHLCP_A_SOURCE=random MHLCP_TARGET_ROWS=$r \
       sage lattice/run_mhlcp_cnn_quant_x.sage --batch_size 10 --file time_and_success \
       > log_n10_r${r}.out 2>&1 &
done

# Then aggregate:
python scripts/summarize_time_and_success.py logs/time_and_success
```

### E. CNN_INT_SCALE scaling study

```bash
for n in 10 20 30 40 50 60 70; do
  for s in -1 10 100 1000 10000; do
    r=$((2*n+20))
    nohup env CNN_INT_SCALE=$s MHLCP_A_SOURCE=random MHLCP_TARGET_ROWS=$r \
         sage lattice/run_mhlcp_cnn_quant_x.sage --batch_size $n --file scaling \
         > log_n${n}_scale${s}.out 2>&1 &
  done
done
python scripts/summarize_scaling.py logs/scaling
python scripts/plot_scaling.py logs/scaling
```

### F. Random-X density sweep (synthetic X with controllable sparsity)

```bash
for n in 50 60 70 80 90 100; do
  r=$((2*n+20))
  nohup env MHLCP_X_SOURCE=random MHLCP_A_SOURCE=random \
            MHLCP_B_X=10 MHLCP_B_A=100 MHLCP_TARGET_ROWS=$r \
       sage lattice/run_mhlcp_cnn_quant_x.sage \
            --batch_size $n --X_non_zero 0.5 --file random_dense \
       > log_n${n}.out 2>&1 &
done
python scripts/summarize_random_dense.py logs/random_dense
```

### G. GIA (PSNR / SSIM)

```bash
# Single batch size, known labels.
python src/cnn.py --mode gia --dataset cifar10 --batch_size 10

# Sweep across batch sizes, mixed-class batches, in parallel.
python src/cnn.py \
    --mode gia \
    --dataset cifar10 \
    --pool \
    --n_runs 10 \
    --n_jobs 20 \
    --batch_size_sweep 1,2,3,4,5,6,7,8,9,10

# Unknown labels, sweep with image dump.
nohup python src/cnn.py \
    --mode gia_unknown \
    --dataset cifar10 \
    --same_label \
    --n_runs 10 \
    --n_jobs 20 \
    --batch_size_sweep 50,100,150,200,250,300,500,1000,2000 \
    --save_fig \
    > cnn.log 2>&1 &

# Mix HLCP success-rate weighting into an existing sweep NPZ:
python analysis/add_hlcp_gia_to_npz.py \
    expdata/gia_sweep/cifar10/<run-dir> \
    --weight_curve logs/cifar10.txt
```

### H. Statistical (FastICA) on CNN tensors

```bash
sage lattice/run_cnn_x_ns_stat.sage
```

This script also auto-invokes `python src/cnn.py --mode fc_hssp` if the NPYs are missing.

## 6. Environment variable reference

### Lattice / mHLCP scripts

| variable                  | default     | applies to                                         |
|---------------------------|-------------|----------------------------------------------------|
| `MHLCP_REPO_ROOT`         | (auto)      | repo-root override (Sage running from `/tmp`)      |
| `MHLCP_TARGET_ROWS`       | `150`       | `m`, sample-row count                              |
| `MHLCP_L`                 | `n`         | `l`, columns of `A`                                |
| `MHLCP_ATTACK`            | `ns`        | `ns`, `statistical`, or `both`                     |
| `MHLCP_X_SOURCE`          | `real`      | `real` (CNN) or `random`                           |
| `MHLCP_A_SOURCE`          | `real`      | `real` (CNN FC input) or `random`                  |
| `MHLCP_B_X`               | `1`         | upper bound on synthetic `X` entries               |
| `MHLCP_B_A`               | `100`       | upper bound on synthetic `A` entries               |
| `MHLCP_A_DISCRETE`        | unset       | set to `1` for discrete `{0..B_A}` entries         |
| `MHLCP_ROW_MODE`          | `random`    | `random` or `relu_stratified`                      |
| `MHLCP_ROW_SELECT_SEED`   | `0`         | seed for X row sampling (parallel sweeps)          |
| `MHLCP_ATTACK_SEED`       | `1`         | seed for A col sampling and attack RNG             |
| `MHLCP_RECOVER_BOX_MAX`   | `220000`    | upper bound for `recoverBox`                       |
| `MHLCP_DATASET`           | `cifar10`   | dataset for auto-generated FC tensors              |
| `CNN_INT_SCALE`           | `1000`      | quantization factor for `floor(fc · CNN_INT_SCALE)`|
| `CNN_USE_RELU_MASK_X`     | unset       | use ReLU mask as X (legacy variant)                |
| `CNN_RELU_SCALE_100`      | unset       | scale ReLU-mask X by 100                           |
| `CNN_FC_SCALES`           | `1000`      | comma list of quantization factors (`run_cnn_x_ns_stat.sage`) |
| `CNN_STAT_LIGHT`          | unset       | shorter sweeps in `run_cnn_x_ns_stat.sage`         |
| `CNN_STAT_SKIP_NS`        | unset       | skip NS in `run_cnn_x_ns_stat.sage`                |

### `cnn.py` flags (selected)

| flag                  | meaning                                                   |
|-----------------------|-----------------------------------------------------------|
| `--mode`              | `train` / `fc_hssp` / `gia` / `gia_unknown` / `gia_loss_probe` / `gia_norm` |
| `--dataset`           | `cifar10`, `cifar100`, `mnist`                            |
| `--batch_size`        | single batch size                                          |
| `--batch_size_sweep`  | comma list `1,2,5` or range `1,...,50` (`50,...,1` reverses) |
| `--n_runs`            | repetitions per batch size                                 |
| `--n_jobs`            | parallel worker count                                      |
| `--pool`              | random pooled batch (mixed labels)                         |
| `--same_label`        | all images share one CIFAR-10 class                        |
| `--label_idx`         | fix the class index when `--same_label` is on              |
| `--save_fig`          | dump originals + reconstructions                           |
| `--known_rate`        | proportion of FC features assumed exactly known            |
| `--known_residual`    | use the (true − partial) residual as a soft prior          |
| `--known_peel`        | strip known rows from the gradient term (unknown-label only)|
| `--noise_on_fc`       | σ for Gaussian noise on FC features (unknown-label only)   |
| `--gpus`              | comma list of GPU ids                                      |

## 7. Reproducibility: experiment drivers

All experiment-level drivers live in `experiments/`. They are self-contained:
each Python driver pins a RNG seed, calls `cnn.py` (or re-uses helpers from
`src/cnn.py`) with explicit flags, and writes its outputs under `expdata/` at
the repository root. Nothing else (seeds, batch sizes, restarts, label config)
is left implicit. Invoke each driver **from the repository root** so that
relative `./expdata/…` and `./logs/…` paths resolve against it.

### 7.1 Main GIA sweeps — `run_exp12_part{1,2}.sh`

The backbone of the paper's GIA numbers. Each shell script runs 6 configurations
of `src/cnn.py`, sweeping batch size 1..128 with 20 independent runs per
configuration:

| script | dataset | configurations (one row per launch) |
|---|---|---|
| `run_exp12_part1.sh` | CIFAR-10 | `gia` no-pool / `gia` pool / `gia_unknown` no-pool / `gia_unknown` pool / `gia_unknown` no-pool same-label / `gia_unknown` pool same-label |
| `run_exp12_part2.sh` | CIFAR-100 | same 6 configurations |

Shared arguments (pinned):

```
--n_runs 20
--n_jobs ${N_JOBS:-16}
--batch_size_sweep 128,...,1
```

Launch::

  N_JOBS=16 nohup bash experiments/run_exp12_part1.sh > run_exp12_part1.out 2>&1 &
  N_JOBS=16 nohup bash experiments/run_exp12_part2.sh > run_exp12_part2.out 2>&1 &

Outputs per configuration go to `expdata/gia_sweep/<dataset>/<timestamp>_…`
and `expdata/gia_unknown_sweep/<dataset>/<timestamp>_…`; per-run stdout is
captured in `logs/part{1,2}_{01..06}_*.log`.

### 7.2 Seed-pinned one-shot GIA experiments

Each of these drivers fixes `SEED = 2026` (or an explicit `SEED_BASE`) so they
produce the same batch of images, and hence the same figures, across machines.

| script | dataset / setup | seed | fixed knobs |
|---|---|---|---|
| `run_gia_baseline.py` | CIFAR-10, 10 distinct labels, pool on/off | `SEED=2026` | `BATCH_SIZE=10`, `GIA_STEPS=4000`, `N_RESTARTS=5` |
| `run_gia_cifar100_distinct.py` | CIFAR-100, 10 distinct classes | `SEED=2026` | `BATCH_SIZE=10`, `GIA_STEPS=4000`, `N_RESTARTS=5` |
| `run_gia_cifar100_improved.py` | CIFAR-100, improved restarts / schedule | `SEED=2026` | `BATCH_SIZE=10`, `GIA_STEPS=8000`, `N_RESTARTS=8` |
| `run_gia_cifar100_methods.py` | CIFAR-100, 6-way method comparison (DLG / Cosine+TV / L-BFGS / Cosine+MSE+TV / strong TV / Adam+pixel-clamp) | `SEED=2026` | `BATCH_SIZE=10`, distinct labels, no pool |
| `run_gia_clamp_vs_hlcp.py` | CIFAR-100 clamp vs HLCP, known labels | `SEED=2026` | base sweep |
| `run_gia_clamp_vs_hlcp_bs40.py` | same at bs=40 | `SEED=2026` | `BATCH_SIZE=40` |
| `run_gia_clamp_vs_hlcp_mixed.py` | same, mixed-class batches | `SEED=2026` | mixed labels |
| `run_gia_clamp_vs_hlcp_unknown.py` | same, unknown-label variant | `SEED=2026` | `--mode gia_unknown` internally |
| `run_pool_known_mixed.py` | CIFAR-100, pool + known labels + mixed classes | `SEED=2026` | bs={10,20,40} |
| `run_pool_unknown_mixed.py` | CIFAR-100, pool + unknown labels + mixed classes | `SEED=2026` | bs={10,20,40} |
| `run_pool_unknown_same.py` | CIFAR-100, pool + unknown labels + all samples from class `SAME_LABEL=3` | `SEED=2026` | bs={10,20,40} |

Invoke any of them as (after activating a torch environment)::

  python experiments/run_gia_baseline.py
  python experiments/run_gia_cifar100_methods.py
  # ...

Each writes figures and metrics under `expdata/<experiment-name>/`.

### 7.3 Full reconstruction-comparison matrix — `launch_all_recon.sh`

Fans out `run_all_recon_compare.py` across a 2×2×2×2 configuration grid
(dataset × pool × label-distribution × label-known). Each job internally sweeps
`bs ∈ {10, 20, 40}` over two reconstruction methods, for a total of
`16 jobs × 3 bs × 2 methods = 96` experiments.

```
bash experiments/launch_all_recon.sh          # launches 16 background jobs
# per-job logs: logs/recon_compare/<tag>.log
```

`run_all_recon_compare.py` uses `SEED=2026` and, for CIFAR-100 same-label runs,
fixes the class to `SAME_LABEL_CIFAR100=3`.

### 7.4 HLCP failure simulation — `run_hlcp_failure_patch.py`

For each (`dataset`, `pool`, `label_dist`, `label_known`, `bs ∈ {20, 40}`)
configuration, simulates HLCP failures at random positions with a random success
rate (`B=20`: `n_fail ~ Uniform{1,2}`; `B=40`: `success_rate ~ U[78%, 96%]`),
re-runs a fresh GIA (1 restart, unique `SEED_BASE=99999`) for the full batch,
and patches the failed indices back into the HLCP strip image under
`expdata/recon_compare_patched/`. Consumes `expdata/recon_compare/` produced by
`run_all_recon_compare.py` — run the reconstruction comparison first.

```
python experiments/run_hlcp_failure_patch.py
```

### 7.5 Lattice-attack timing benchmarks — `bench_*.sage`

| script | what it measures | key seed |
|---|---|---|
| `bench_time_m3n10.sage` | Wall-clock time of the NS attack on synthetic HSSP (`B_X=1`) and HLCP (`B_X ∈ {100, 1000}`) with `m = 3n + 10` and `n ∈ {10, 20, 30, 40, 50, 60, 70}` | `SEED_BASE = 20260411`; per-instance seed `SEED_BASE + n·1000 + B_X` |
| `bench_real_cnn_v2.sage` | Same sweep but with real CNN `fc1_pre_relu` data, quantized at `INT_SCALE=1000` | `SEED = 42` for row selection and `A` generation (`SEED + n` / `SEED + n + 10000`) |

Prerequisite for `bench_real_cnn_v2.sage`: run FC export first for every `n`
you want to benchmark::

  for n in 10 20 30 40 50 60 70; do
      python src/cnn.py --mode fc_hssp --batch_size $n --dataset cifar10
  done
  sage experiments/bench_real_cnn_v2.sage

Results are appended to `bench_real_cnn_v2_<hostname>.csv` / `bench_time_m3n10_<hostname>.csv`.

### 7.6 Post-processing

After the experiments finish, the `scripts/` utilities fold the per-trial logs
into aggregate metrics, and `analysis/*.py` turns them into publication figures.
See §5 (workflow recipes) and §6 (environment variables) above, or the brief
summaries in [`architecture.md`](architecture.md).

### 7.7 Transpose attack (alternative pipeline)

Whereas the main forward attack (`run_mhlcp_cnn_quant_x.sage`) treats
`X = quantized fc1_pre_relu` as the secret, the transpose attack
(`run_mhlcp_transpose.sage`) takes the *gradient backprop factor*
`X = quantized backprop_masked_fc` instead. The companion
`A = quantized fc_input_batch` is shared with the forward attack, so the same
FC export data is reused — no extra preparation is needed.

Two row-selection strategies are exposed, controlled by `MHLCP_ROW_MODE`:

| mode | semantics | role |
|---|---|---|
| `random` | random permutation seeded by `ROWSEED` | **default — the main row-selection method** |
| `bias`   | sort rows by `\|sum_j X_int[i, j]\| = \|∇b_1[i]\|` (attacker-observable FC1 bias gradient) | **exploration alternative**; useful as ablation, not as the canonical baseline |

Per-trial invocation::

  # random (default)
  BS=20 RUN=0 C=50 sage lattice/run_mhlcp_transpose.sage

  # bias (exploration)
  BS=20 RUN=0 C=50 MHLCP_ROW_MODE=bias sage lattice/run_mhlcp_transpose.sage

Each trial prints exactly one pipe-separated line, e.g.

```
RESULT|sel=random|bs=20|run=0|c=50|nf=20|unique=20|n=20|st=S|bx=...|beta=...|...|ttotal=...
```

Aggregating a sweep::

  for BS in 10 20 30 40 50; do
      MT=$((7 * BS + 10))
      for C in 50 100 1000; do
          for RUN in $(seq 0 9); do
              BS=$BS RUN=$RUN C=$C M_TARGET=$MT NX0=20 ROWSEED=$((42+RUN)) \
                  sage lattice/run_mhlcp_transpose.sage \
                  > logs/random_subsample/bs${BS}_c${C}_run${RUN}.log 2>&1 &
          done
      done
  done
  wait
  python scripts/aggregate_subsample.py \
      --logs-dir logs/random_subsample \
      --pattern 'bs*_c*_run*.log' \
      --mode random \
      --out logs/summary_random_subsample.json

The same recipe works for `MHLCP_ROW_MODE=bias` — only the log directory and the
`--mode bias` flag change.

Multi-run data (`HSSP_RUN_ID=<R>`)::

  for R in $(seq 0 19); do
      HSSP_RUN_ID=$R python src/cnn.py --mode fc_hssp --batch_size 20 --dataset cifar10
  done
  # then the transpose attack picks `expdata/fc_hssp/cifar10/bs20/run_<R>/...`
  # automatically when invoked with RUN=<R>.

## 8. Mainline vs legacy

### Mainline (read these first)

1. `lattice/multi_dim_hssp.sage`
2. `lattice/run_mhlcp_cnn_quant_x.sage`  (★ main forward attack)
3. `lattice/run_mhlcp_transpose.sage`    (alternative transpose attack — random main, bias exploration)
4. `src/cnn_model.py`
5. `src/cnn.py`
6. `analysis/plot_metrics.py`
7. `scripts/summarize_*.py`, `scripts/aggregate_subsample.py`

### Reference (for context)

- `lattice/extendhssp.sage` — earlier matrix extension, kept for reference. The active matrix code lives in `multi_dim_hssp.sage`.
- `lattice/test_hssp.sage`, `lattice/test_hlcp.sage` — Gini's smoke tests of the vector versions.
- `src/mlp_hlcp.py` — MLP variant for HLCP study, useful for ablations.
- `analysis/add_hlcp_gia_to_npz.py` — produces the *post-processed* `HLCP + GIA` curve (success-rate-weighted mix), not a per-trial closed-loop measurement.

## 9. Common gotchas

- **Where does `cnn.py` write?** Relative to `cwd`. Always run `python src/cnn.py` from the repo root, not from inside `src/`, otherwise `expdata/` ends up under `src/`.
- **Sage `load("foo.sage")` is relative to `cwd`.** All run scripts in `lattice/` self-locate the lattice library directory and `os.chdir` before loading; you can launch them either from the repo root (`sage lattice/run_xxx.sage`) or from inside `lattice/`.
- **`run_mhlcp_cnn_quant_x.sage` chdirs to the repo root** so that `expdata/fc_hssp/...` resolves correctly. The script then does `load("lattice/multi_dim_hssp.sage")`.
- **NS vs statistical.** NS is robust on integer mHLCP up to a fairly large `B_X`. Statistical (FastICA) is best with `B = 1` or near-binary X; with large `B_X` or large `CNN_INT_SCALE` it usually finishes but recovers little.
- **NFound is the success metric.** It counts how many recovered row-vectors hit a row of `X^T`; it does *not* automatically translate into GIA quality.
- **`HLCP + GIA` curves are post-processed.** The mix `w(n) · (HSSP + GIA) + (1 − w(n)) · naive` from `analysis/add_hlcp_gia_to_npz.py` is a calibrated upper-bound estimate, not a per-trial closed-loop attack.
