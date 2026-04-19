#!/bin/bash
# Launch the full 16-configuration reconstruction-comparison matrix in parallel.
#   2 datasets (cifar10/cifar100) x 2 pool x 2 label_dist (same/mixed) x 2 label_known = 16 jobs.
# Each job internally sweeps batch sizes 10, 20, 40 across two reconstruction methods.
# Total: 16 * 3 * 2 = 96 individual experiments.
#
# Run from the repository root. Prerequisite: activate an environment with torch installed
# (e.g. `source /path/to/conda.sh && conda activate <env>`).
#
# Usage:
#   cd /path/to/gradient-lattice-attack
#   bash experiments/launch_all_recon.sh

set -euo pipefail

cd "$(dirname "$0")/.."  # run from repository root

mkdir -p logs/recon_compare

for dataset in cifar10 cifar100; do
  for pool in 0 1; do
    for label_dist in same mixed; do
      for label_known in 0 1; do
        tag="${dataset}_pool${pool}_${label_dist}_known${label_known}"
        echo "Launching: $tag"
        python -u experiments/run_all_recon_compare.py \
          --dataset $dataset --pool $pool \
          --label_dist $label_dist --label_known $label_known \
          > logs/recon_compare/${tag}.log 2>&1 &
      done
    done
  done
done

echo "All 16 jobs launched. PIDs:"
jobs -p
wait
echo "All done!"
