#!/usr/bin/env bash
# Experiments 1--6: CIFAR-100 GIA sweep (known/unknown labels, pool/no-pool,
# mixed/same class) over batch sizes 128..1, with 20 independent runs each.
# Run from the repository root; outputs land in ./logs/part2_*.log
# and ./expdata/gia_sweep / gia_unknown_sweep.
#
# Prerequisite: activate an environment with torch installed, e.g.
#   source /path/to/conda.sh && conda activate <env>
set -euo pipefail

cd "$(dirname "$0")/.."  # run from repository root

mkdir -p logs

# Tune inside-experiment parallelism to avoid CUDA OOM.
# You can override at launch time, e.g.:
#   N_JOBS=16 nohup bash experiments/run_exp12_part2.sh > run_exp12_part2.out 2>&1 &
N_JOBS="${N_JOBS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Common args
COMMON_ARGS=(
  --n_runs 20
  --n_jobs "${N_JOBS}"
  --batch_size_sweep 128,...,1
)

echo "[$(date '+%F %T')] [1/6] cifar100 gia no_pool same_label=False"
python src/cnn.py \
  --mode gia \
  --dataset cifar100 \
  "${COMMON_ARGS[@]}" \
  > logs/part2_01_cifar100_gia_nopool_mixed.log 2>&1

echo "[$(date '+%F %T')] [2/6] cifar100 gia pool same_label=False"
python src/cnn.py \
  --mode gia \
  --dataset cifar100 \
  --pool \
  "${COMMON_ARGS[@]}" \
  > logs/part2_02_cifar100_gia_pool_mixed.log 2>&1

echo "[$(date '+%F %T')] [3/6] cifar100 gia_unknown no_pool same_label=False"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar100 \
  "${COMMON_ARGS[@]}" \
  > logs/part2_03_cifar100_gia_unknown_nopool_mixed.log 2>&1

echo "[$(date '+%F %T')] [4/6] cifar100 gia_unknown pool same_label=False"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar100 \
  --pool \
  "${COMMON_ARGS[@]}" \
  > logs/part2_04_cifar100_gia_unknown_pool_mixed.log 2>&1

echo "[$(date '+%F %T')] [5/6] cifar100 gia_unknown no_pool same_label=True"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar100 \
  --same_label \
  "${COMMON_ARGS[@]}" \
  > logs/part2_05_cifar100_gia_unknown_nopool_same.log 2>&1

echo "[$(date '+%F %T')] [6/6] cifar100 gia_unknown pool same_label=True"
python src/cnn.py \
  --mode gia_unknown \
  --dataset cifar100 \
  --pool \
  --same_label \
  "${COMMON_ARGS[@]}" \
  > logs/part2_06_cifar100_gia_unknown_pool_same.log 2>&1

echo "[$(date '+%F %T')] part2 done."
