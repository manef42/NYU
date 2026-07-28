#!/bin/bash
#SBATCH --job-name=ignition-classifier-array
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a_eval_output.log
#SBATCH --error=logs/%A_%a_eval_error.log

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CLASSIFIER_DIR="${SLURM_SUBMIT_DIR}"
OVERLAY="/scratch/me3144/ignition_hpc/ignition_python.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

mkdir -p "${CLASSIFIER_DIR}/logs" "${CLASSIFIER_DIR}/results/evaluation"
cd "${CLASSIFIER_DIR}"

unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV

singularity exec \
  --overlay "${OVERLAY}:ro" \
  --bind "${CLASSIFIER_DIR}:${CLASSIFIER_DIR}" \
  "${IMAGE}" \
  /bin/bash -lc '
    source /ext3/env.sh
    cd "'"${CLASSIFIER_DIR}"'"
    python fable_evaluate.py \
      --data "'"${CLASSIFIER_DIR}"'/Microgravity_Database_reduced.csv" \
      --splits results/splits \
      --config configs/candidates.yaml \
      --out results/evaluation \
      --search-iterations 40 \
      --inner-group-folds 3 \
      --bootstrap-iterations 2000 \
      --model-id ${SLURM_ARRAY_TASK_ID}
  '
