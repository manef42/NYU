#!/bin/bash
#SBATCH --job-name=fsr-mlp-candidates
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%A_%a_mlp_output.log
#SBATCH --error=logs/%A_%a_mlp_error.log

# One array task per MLP candidate configuration:
# 0=mlp_baseline
# 1=mlp_weighted_sqrt
# 2=mlp_weighted_inverse
# 3=mlp_focal_sqrt
# 4=mlp_focal_inverse
#
# Each task tunes and benchmarks its candidate on every persisted split and
# writes a self-contained shard under results/evaluation/mlp/<candidate_id>.
# CUDA MPS lets concurrent inner-fold fits share the allocated GPU.
#
# Typical submission (after splits and the main array have completed):
#   sbatch run_regressors_mlp_array.slurm
# Or chained:
#   ARRAY_JID=$(sbatch --parsable run_regressors_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JID} run_regressors_mlp_array.slurm

set -euo pipefail

REGRESSOR_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(dirname "${REGRESSOR_DIR}")"
DATA="${FSR_DATA:-${REGRESSOR_DIR}/Microgravity_Database.csv}"
OVERLAY="${FSR_OVERLAY:-/scratch/me3144/ignition_hpc/ignition_python.ext3}"
IMAGE="${FSR_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
SEARCH_ITERATIONS="${FSR_SEARCH_ITERATIONS:-40}"
INNER_FOLDS="${FSR_INNER_FOLDS:-5}"
BOOTSTRAP_ITERATIONS="${FSR_BOOTSTRAP_ITERATIONS:-2000}"

CANDIDATES=(
    mlp_baseline
    mlp_weighted_sqrt
    mlp_weighted_inverse
    mlp_focal_sqrt
    mlp_focal_inverse
)

CANDIDATE_ID="${CANDIDATES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR="${REGRESSOR_DIR}/results/evaluation/mlp/${CANDIDATE_ID}"

mkdir -p "${REGRESSOR_DIR}/logs" "${OUT_DIR}"
cd "${REGRESSOR_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    "${IMAGE}" \
    /bin/bash -lc '
        source /ext3/env.sh
        cd "'"${REGRESSOR_DIR}"'"
        python fsr_evaluate.py \
            --data "'"${DATA}"'" \
            --splits results/splits \
            --config configs/candidates.yaml \
            --out "'"${OUT_DIR}"'" \
            --search-iterations '"${SEARCH_ITERATIONS}"' \
            --inner-folds '"${INNER_FOLDS}"' \
            --search-jobs '"${SLURM_CPUS_PER_TASK}"' \
            --bootstrap-iterations '"${BOOTSTRAP_ITERATIONS}"' \
            --augmentations rd,rd_bt \
            --model-id 3 \
            --candidate-id "'"${CANDIDATE_ID}"'"'

echo quit | nvidia-cuda-mps-control
