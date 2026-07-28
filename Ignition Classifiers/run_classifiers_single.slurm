#!/bin/bash
#SBATCH --job-name=ignition-splits
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_splits_output.log
#SBATCH --error=logs/%j_splits_error.log

set -euo pipefail

CLASSIFIER_DIR="${SLURM_SUBMIT_DIR}"
OVERLAY="/scratch/me3144/ignition_hpc/ignition_python.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

mkdir -p "${CLASSIFIER_DIR}/results" "${CLASSIFIER_DIR}/logs"
cd "${CLASSIFIER_DIR}"

unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind "${CLASSIFIER_DIR}:${CLASSIFIER_DIR}" \
  "${IMAGE}" \
  /bin/bash -lc '
    source /ext3/env.sh
    cd "'"${CLASSIFIER_DIR}"'"
    python fable_splits.py \
      --data "'"${CLASSIFIER_DIR}"'/Microgravity_Database_reduced.csv" \
      --out results/splits \
      --n-seeds 3 \
      --n-group-folds 5 \
      --n-row-folds 5
  '