#!/bin/bash
#SBATCH -J fair_comparison_seg
#SBATCH -o /export/home2/reny0012/result/runlog/fair_comparison_seg_%j.log
#SBATCH -e /export/home2/reny0012/result/runlog/fair_comparison_seg_%j.err
#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH -n 1
#SBATCH -c 8

# Fair comparison of saved DeepCA variable-view-model predictions,
# evaluated with two segmentation-artifact-corrupted input views.
set -euo pipefail

module load python3 2>/dev/null || true
source "${VENV:-/export/home2/reny0012/vir_env/autocar}/bin/activate"
cd /export/home2/reny0012/code/autocar_release_training

export VESSEL_TREE_GENERATOR_ROOT=/export/home2/reny0012/code/vessel_tree_generator
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

RAW_RCA=/dataset/reny0012/vessel_code/rca
RAW_LCA=/dataset/reny0012/vessel_code/lca
OPTIONS=/export/home2/reny0012/code/vessel_tree_generator/methods/fair_comparison/options.json
PRED_RCA=/export/home2/reny0012/result/deepca_imagecas/rca_variable_views/evaluation_seg_err_artifacts_normal/test/predictions/views_2
PRED_LCA=/export/home2/reny0012/result/deepca_imagecas/lca_variable_views/evaluation_seg_err_artifacts_normal/test/predictions/views_2
OUT=/export/home2/reny0012/code/autocar_release_training/logs/seg_error_artifacts_deepca_variable_views_fair_comparison

[[ -f "$OPTIONS" ]] || { echo "Missing options: $OPTIONS" >&2; exit 1; }
[[ -f scripts/evaluate_fair_comparison.py ]] || { echo "Missing fair-comparison evaluator" >&2; exit 1; }
# Check both inputs before starting; do not overwrite previous evaluations.
for artery in rca lca; do
  if [[ "$artery" == rca ]]; then
    predictions="$PRED_RCA"; raw="$RAW_RCA"
  else
    predictions="$PRED_LCA"; raw="$RAW_LCA"
  fi
  [[ -d "$raw" ]] || { echo "Missing raw-vessel directory: $raw" >&2; exit 1; }
  [[ -d "$predictions" ]] || { echo "Missing predictions: $predictions" >&2; exit 1; }
  compgen -G "$predictions/*.npz" > /dev/null || { echo "No NPZ predictions: $predictions" >&2; exit 1; }
done

for artery in rca lca; do
  if [[ "$artery" == rca ]]; then
    predictions="$PRED_RCA"; raw="$RAW_RCA"
  else
    predictions="$PRED_LCA"; raw="$RAW_LCA"
  fi
  echo "[STARTING] DeepCA $artery segmentation artifacts, 2 views"
  python scripts/evaluate_fair_comparison.py \
    --prediction-method deepca \
    --artery "$artery" \
    --prediction-dir "$predictions" \
    --raw-vessel-dir "$raw" \
    --options-json "$OPTIONS" \
    --output-dir "$OUT/deepca_${artery}_views_2"
  echo "[COMPLETED] DeepCA $artery: $OUT/deepca_${artery}_views_2"
done
