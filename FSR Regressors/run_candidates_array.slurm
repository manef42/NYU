#!/bin/bash
#SBATCH --job-name=fsr-candidates
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%A_%a_eval_output.log
#SBATCH --error=logs/%A_%a_eval_error.log

# One array task per candidate (30 total):
#  0  xgb_baseline              (xgboost)
#  1  xgb_paper_sqrt            (xgboost)
#  2  xgb_paper_inverse         (xgboost)
#  3  xgb_absoluteerror_sqrt    (xgboost)
#  4  xgb_absoluteerror_inverse (xgboost)
#  5  xgb_pseudohuber_sqrt      (xgboost)
#  6  xgb_pseudohuber_inverse   (xgboost)
#  7  xgb_log_target_sqrt       (xgboost)
#  8  xgb_log_target_inverse    (xgboost)
#  9  xgb_paper_bagging_sqrt    (xgboost)
# 10  rf_baseline               (random_forest)
# 11  rf_paper_sqrt             (random_forest)
# 12  rf_paper_inverse          (random_forest)
# 13  knn_uniform               (knn)
# 14  knn_uniform_sqrt          (knn)
# 15  knn_uniform_inverse       (knn)
# 16  knn_distance              (knn)
# 17  knn_distance_sqrt         (knn)
# 18  knn_distance_inverse      (knn)
# 19  mlp_baseline              (mlp)
# 20  mlp_paper_sqrt            (mlp)
# 21  mlp_paper_inverse         (mlp)
# 22  mlp_log_target_sqrt       (mlp)
# 23  mlp_log_target_inverse    (mlp)
# 24  svr_linear                (svr)
# 25  svr_linear_sqrt           (svr)
# 26  svr_linear_inverse        (svr)
# 27  svr_rbf                   (svr)
# 28  svr_rbf_sqrt              (svr)
# 29  svr_rbf_inverse           (svr)
#
# Each task tunes and benchmarks its single candidate on every persisted split
# and writes a self-contained shard under results/evaluation/<family>/<candidate_id>.
# CUDA MPS lets concurrent inner-fold fits share the allocated GPU.
#
# Typical submission (splits must already exist):
#   SPLITS_JID=$(sbatch --parsable run_regressors_splits.slurm)
#   sbatch --dependency=afterok:${SPLITS_JID} run_candidates_array.slurm

set -euo pipefail

REGRESSOR_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(dirname "${REGRESSOR_DIR}")"
DATA="${FSR_DATA:-${REGRESSOR_DIR}/Microgravity_Database.csv}"
OVERLAY="${FSR_OVERLAY:-/scratch/me3144/ignition_hpc/ignition_python.ext3}"
IMAGE="${FSR_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
SEARCH_ITERATIONS="${FSR_SEARCH_ITERATIONS:-40}"
INNER_FOLDS="${FSR_INNER_FOLDS:-5}"
BOOTSTRAP_ITERATIONS="${FSR_BOOTSTRAP_ITERATIONS:-2000}"

# ---- candidate index → (family, candidate_id) --------------------------------
FAMILIES=(
    xgboost xgboost xgboost xgboost xgboost
    xgboost xgboost xgboost xgboost xgboost
    random_forest random_forest random_forest
    knn knn knn knn knn knn
    mlp mlp mlp mlp mlp
    svr svr svr svr svr svr
)
CANDIDATES=(
    xgb_baseline
    xgb_paper_sqrt
    xgb_paper_inverse
    xgb_absoluteerror_sqrt
    xgb_absoluteerror_inverse
    xgb_pseudohuber_sqrt
    xgb_pseudohuber_inverse
    xgb_log_target_sqrt
    xgb_log_target_inverse
    xgb_paper_bagging_sqrt
    rf_baseline
    rf_paper_sqrt
    rf_paper_inverse
    knn_uniform
    knn_uniform_sqrt
    knn_uniform_inverse
    knn_distance
    knn_distance_sqrt
    knn_distance_inverse
    mlp_baseline
    mlp_paper_sqrt
    mlp_paper_inverse
    mlp_log_target_sqrt
    mlp_log_target_inverse
    svr_linear
    svr_linear_sqrt
    svr_linear_inverse
    svr_rbf
    svr_rbf_sqrt
    svr_rbf_inverse
)

FAMILY="${FAMILIES[${SLURM_ARRAY_TASK_ID}]}"
CANDIDATE_ID="${CANDIDATES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR="${REGRESSOR_DIR}/results/evaluation/${FAMILY}/${CANDIDATE_ID}"

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
            --candidate-id "'"${CANDIDATE_ID}"'"'

echo quit | nvidia-cuda-mps-control
