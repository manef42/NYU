#!/bin/bash
#SBATCH --job-name=ignition-finalize
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_finalize_output.log
#SBATCH --error=logs/%j_finalize_error.log

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CLASSIFIER_DIR="${SLURM_SUBMIT_DIR}"
OVERLAY="/scratch/me3144/ignition_hpc/ignition_python.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

mkdir -p "${CLASSIFIER_DIR}/artifacts" "${CLASSIFIER_DIR}/logs"
cd "${CLASSIFIER_DIR}"

unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV

singularity exec \
    --overlay "${OVERLAY}:ro" \
    --bind "${CLASSIFIER_DIR}:${CLASSIFIER_DIR}" \
    "${IMAGE}" \
    /bin/bash -lc '
set -euo pipefail

source /ext3/env.sh
cd "$1"

python fable_select.py \
    --evaluation results/evaluation \
    --policy configs/selection_policy.yaml \
    --out results/selection.json

python fable_refit.py \
    --data "$1/Microgravity_Database.csv" \
    --selection results/selection.json \
    --champion interpolation \
    --out artifacts/interpolation_champion

python fable_refit.py \
    --data "$1/Microgravity_Database.csv" \
    --selection results/selection.json \
    --champion extrapolation \
    --out artifacts/extrapolation_champion

python fable_report.py \
    --data "$1/Microgravity_Database.csv" \
    --evaluation results/evaluation \
    --selection results/selection.json \
    --artifacts artifacts \
    --out results/report
' bash "${CLASSIFIER_DIR}"