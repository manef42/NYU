#!/bin/bash
#SBATCH --job-name=fsr-finalize
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_finalize_output.log
#SBATCH --error=logs/%j_finalize_error.log

# Submit with a dependency on the array job, for example:
#   ARRAY_ID=$(sbatch --parsable run_regressors_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_ID} run_regressors_finalize.slurm

set -euo pipefail

REGRESSOR_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(dirname "${REGRESSOR_DIR}")"
DATA="${FSR_DATA:-${REPO_DIR}/Microgravity_Database.csv}"
OVERLAY="${FSR_OVERLAY:-/scratch/me3144/ignition_hpc/ignition_python.ext3}"
IMAGE="${FSR_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
BOOTSTRAP_ITERATIONS="${FSR_BOOTSTRAP_ITERATIONS:-2000}"

mkdir -p "${REGRESSOR_DIR}/artifacts" "${REGRESSOR_DIR}/logs"
cd "${REGRESSOR_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    "${IMAGE}" \
    /bin/bash -lc '
        source /ext3/env.sh
        cd "'"${REGRESSOR_DIR}"'"
        python fsr_aggregate.py \
            --evaluation results/evaluation \
            --out results/evaluation/merged \
            --bootstrap-iterations '"${BOOTSTRAP_ITERATIONS}"' && \
        python fsr_select.py \
            --evaluation results/evaluation/merged \
            --policy configs/selection_policy.yaml \
            --out results/selection.json && \
        python fsr_refit.py \
            --data "'"${DATA}"'" \
            --selection results/selection.json \
            --evaluation results/evaluation/merged \
            --champion interpolation \
            --out artifacts/interpolation_champion && \
        python fsr_refit.py \
            --data "'"${DATA}"'" \
            --selection results/selection.json \
            --evaluation results/evaluation/merged \
            --champion extrapolation \
            --out artifacts/extrapolation_champion && \
        python fsr_report.py \
            --data "'"${DATA}"'" \
            --evaluation results/evaluation/merged \
            --selection results/selection.json \
            --artifacts artifacts \
            --out results/report'
