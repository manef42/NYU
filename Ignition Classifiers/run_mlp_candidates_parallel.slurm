#!/bin/bash
#SBATCH --job-name=ignition-mlp
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%A_%a_mlp_output.log
#SBATCH --error=logs/%A_%a_mlp_error.log

set -euo pipefail

# Avoid each library / subprocess attempting to use all 16 cores at once.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CLASSIFIER_DIR="${SLURM_SUBMIT_DIR}"
OVERLAY="/scratch/me3144/ignition_hpc/ignition_python.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

CANDIDATES=(
    "mlp_baseline"
    "mlp_weighted_sqrt"
    "mlp_weighted_inverse"
    "mlp_focal_sqrt"
    "mlp_focal_inverse"
)

CANDIDATE_ID="${CANDIDATES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR="${CLASSIFIER_DIR}/results/evaluation/mlp/${CANDIDATE_ID}"

mkdir -p "${CLASSIFIER_DIR}/logs" "${OUT_DIR}"
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
    --data "'"${CLASSIFIER_DIR}"'/Microgravity_Database.csv" \
    --splits results/splits \
    --config configs/candidates.yaml \
    --out "'"${OUT_DIR}"'" \
    --search-iterations 15 \
    --inner-group-folds 3 \
    --bootstrap-iterations 2000 \
    --model mlp \
    --candidate-id "'"${CANDIDATE_ID}"'"
'