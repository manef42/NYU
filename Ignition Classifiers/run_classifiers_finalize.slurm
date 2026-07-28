#!/bin/bash
#SBATCH --job-name=ignition-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_finalize_output.log
#SBATCH --error=logs/%j_finalize_error.log

set -euo pipefail

source /scratch/me3144/venv_ignition/bin/activate

CLASSIFIER_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${CLASSIFIER_DIR}/artifacts" "${CLASSIFIER_DIR}/logs"
cd "${CLASSIFIER_DIR}"

python fable_select.py \
  --evaluation results/evaluation \
  --policy configs/selection_policy.yaml \
  --out results/selection.json && \
python fable_refit.py \
  --data "${CLASSIFIER_DIR}/Microgravity_Database_reduced.csv" \
  --selection results/selection.json \
  --champion interpolation \
  --out artifacts/interpolation_champion && \
python fable_refit.py \
  --data "${CLASSIFIER_DIR}/Microgravity_Database_reduced.csv" \
  --selection results/selection.json \
  --champion extrapolation \
  --out artifacts/extrapolation_champion && \
python fable_report.py \
  --data "${CLASSIFIER_DIR}/Microgravity_Database_reduced.csv" \
  --evaluation results/evaluation \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report