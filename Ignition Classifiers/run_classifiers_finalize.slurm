#!/bin/bash
#SBATCH --job-name=ignition-finalize
#SBATCH --account=torch_pr_1163_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_finalize_output.log
#SBATCH --error=logs/%j_finalize_error.log

set -euo pipefail

CLASSIFIER_DIR="${SLURM_SUBMIT_DIR}"
OVERLAY="/scratch/me3144/ignition_hpc/ignition_python.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

mkdir -p "${CLASSIFIER_DIR}/artifacts" "${CLASSIFIER_DIR}/logs"
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
python fable_select.py \
--evaluation results/evaluation \
--policy configs/selection_policy.yaml \
--out results/selection.json && \
python fable_refit.py \
--data "'"${CLASSIFIER_DIR}"'/Microgravity_Database_reduced.csv" \
--selection results/selection.json \
--champion interpolation \
--out artifacts/interpolation_champion && \
python fable_refit.py \
--data "'"${CLASSIFIER_DIR}"'/Microgravity_Database_reduced.csv" \
--selection results/selection.json \
--champion extrapolation \
--out artifacts/extrapolation_champion && \
python fable_report.py \
--data "'"${CLASSIFIER_DIR}"'/Microgravity_Database_reduced.csv" \
--evaluation results/evaluation \
--selection results/selection.json \
--artifacts artifacts \
--out results/report'