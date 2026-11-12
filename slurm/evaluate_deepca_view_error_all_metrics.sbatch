#!/bin/bash
#SBATCH -J fair_comparison
#SBATCH -o /export/home2/reny0012/result/runlog/fair_comparison_%j.log
#SBATCH -e /export/home2/reny0012/result/runlog/fair_comparison_%j.err
#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH -n 1
#SBATCH -c 8

# DeepCA: RCA and LCA, two input views, offsets 0.5/1/2/4 degrees.
# Dice/clDice: raw-vessel tubes on the 0.5-mm grid.
# CD/radius MAE v2: shared fair-comparison evaluator.
# Requires both scripts/evaluate_deepca_view_error_all_metrics.sh and
# scripts/summarize_deepca_view_error.py in the server repository.
set -euo pipefail

module load python3 2>/dev/null || true
source "${VENV:-/export/home2/reny0012/vir_env/autocar}/bin/activate"
cd /export/home2/reny0012/code/autocar_release_training

export RAW_RCA=/dataset/reny0012/vessel_code/rca
export RAW_LCA=/dataset/reny0012/vessel_code/lca
export OUT=/export/home2/reny0012/code/autocar_release_training/logs/inaccurate_view_inspect_final
export OPTIONS=/export/home2/reny0012/code/vessel_tree_generator/methods/fair_comparison/options.json
export VESSEL_TREE_GENERATOR_ROOT=/export/home2/reny0012/code/vessel_tree_generator
export PRED_ROOT="${PRED_ROOT:-/export/home2/reny0012/result/deepca_imagecas}"
# Each root contains the four error_*deg directories; override at submission if needed.
export RCA_EVAL_ROOT="${RCA_EVAL_ROOT:-$PRED_ROOT/rca/evaluation_inacc_view}"
export LCA_EVAL_ROOT="${LCA_EVAL_ROOT:-$PRED_ROOT/lca/evaluation_inacc_view}"
export PYTHON="$(command -v python)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# Keep existing results unless explicitly submitted with OVERWRITE=1.
export OVERWRITE="${OVERWRITE:-0}"

printf '[STARTING] job=%s host=%s python=%s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "$PYTHON"
printf 'Predictions: %s\nRaw RCA: %s\nRaw LCA: %s\nOptions: %s\nOutput: %s\n' \
  "$PRED_ROOT" "$RAW_RCA" "$RAW_LCA" "$OPTIONS" "$OUT"

printf "RCA evaluation root: %s\nLCA evaluation root: %s\n" "$RCA_EVAL_ROOT" "$LCA_EVAL_ROOT"

bash scripts/evaluate_deepca_view_error_all_metrics.sh

echo "[COMPLETED] Table: $OUT/deepca_view_error_table.md"
