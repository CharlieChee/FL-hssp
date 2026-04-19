#!/usr/bin/env bash
# Experiments 1--6: CIFAR-10 GIA sweep (known/unknown labels, pool/no-pool,
# mixed/same class) over batch sizes 128..1, with 20 independent runs each.
# Run from the repository root; outputs land in ./logs/part1_*.log
# and ./expdata/gia_sweep / gia_unknown_sweep.
#
# Prerequisite: activate an environment with torch installed, e.g.
#   source /path/to/conda.sh && conda activate <env>
set -euo pipefail

cd "$(dirname "$0")/.."  # run from repository root

mkdir -p logs

# Tune inside-experiment parallelism to avoid CUDA OOM.
# You can override at launch time, e.g.:
#   N_JOBS=16 nohup bash experiments/run_exp12_part1.sh > run_exp12_part1.out 2>&1 &
N_JOBS="${N_JOBS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Common args
COMMON_ARGS=(
  --n_runs 20
  --n_jobs "${N_JOBS}"
  --batch_size_sweep 128,...,1
)

echo "[$(date '+%F %T')] [1/6] cifar10 gia no_pool same_label=False"
python src/cnn.py \
  --mode gia \
  --dataset cifar10 \
  "${COMMON_ARGS[@]}" \
  > logs/part1_01_cifar10_gia_nopool_mixed.log 2>&1

echo "[$(date '+%F %T')] [2/6] cifar10 gia pool same_label=False"
python src/cnn.py \
  --mode gia \
  --dataset cifar10 \
  --pool \
  "${COMMON_ARGS[@]}" \
  > logs/part1_02_cifar10_gia_pool_mixed.log 2>&1

echo "[$(date '+%F %T')] [3/6] cifar10 gia_unknown no_pool same_label=False"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar10 \
  "${COMMON_ARGS[@]}" \
  > logs/part1_03_cifar10_gia_unknown_nopool_mixed.log 2>&1

echo "[$(date '+%F %T')] [4/6] cifar10 gia_unknown pool same_label=False"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar10 \
  --pool \
  "${COMMON_ARGS[@]}" \
  > logs/part1_04_cifar10_gia_unknown_pool_mixed.log 2>&1

echo "[$(date '+%F %T')] [5/6] cifar10 gia_unknown no_pool same_label=True"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar10 \
  --same_label \
  "${COMMON_ARGS[@]}" \
  > logs/part1_05_cifar10_gia_unknown_nopool_same.log 2>&1

echo "[$(date '+%F %T')] [6/6] cifar10 gia_unknown pool same_label=True"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar10 \
  --pool \
  --same_label \
  "${COMMON_ARGS[@]}" \
  > logs/part1_06_cifar10_gia_unknown_pool_same.log 2>&1

echo "[$(date '+%F %T')] part1 done."
