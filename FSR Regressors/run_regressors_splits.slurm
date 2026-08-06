#!/bin/bash
#SBATCH --job-name=fsr-splits
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_splits_output.log
#SBATCH --error=logs/%j_splits_error.log

set -euo pipefail

REGRESSOR_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(dirname "${REGRESSOR_DIR}")"
DATA="${FSR_DATA:-${REPO_DIR}/Microgravity_Database.csv}"
OVERLAY="${FSR_OVERLAY:-/scratch/me3144/ignition_hpc/ignition_python.ext3}"
IMAGE="${FSR_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
N_REPEATS="${FSR_N_REPEATS:-10}"

mkdir -p "${REGRESSOR_DIR}/results" "${REGRESSOR_DIR}/logs"
cd "${REGRESSOR_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV

singularity exec \
    --overlay "${OVERLAY}:ro" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    "${IMAGE}" \
    /bin/bash -lc '
        source /ext3/env.sh
        cd "'"${REGRESSOR_DIR}"'"
        python fsr_splits.py \
            --data "'"${DATA}"'" \
            --out results/splits \
            --n-repeats '"${N_REPEATS}"' \
            --test-size 0.2 \
            --split-candidates 50 \
            --target-bins 6'
