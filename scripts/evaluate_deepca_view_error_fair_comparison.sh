#!/usr/bin/env bash
# Run from the AutoCAR repository with its evaluation environment activated.
set -euo pipefail

: "${RAW_RCA:?Set RAW_RCA to the RCA raw-vessel directory}"
: "${RAW_LCA:?Set RAW_LCA to the LCA raw-vessel directory}"
: "${OPTIONS:?Set OPTIONS to the shared fair-comparison options JSON}"
: "${OUT:?Set OUT to the fair-comparison output root}"
PRED_ROOT="${PRED_ROOT:-/export/home2/reny0012/result/deepca_imagecas}"
PYTHON="${PYTHON:-python}"
conditions=(error_0p5deg error_1deg error_2deg error_4deg)

[[ -f scripts/evaluate_fair_comparison.py ]] || { echo 'Run from the AutoCAR repository.' >&2; exit 1; }
[[ -f "$OPTIONS" ]] || { echo "Missing options: $OPTIONS" >&2; exit 1; }
# Check all eight inputs before starting the first evaluation.
for artery in rca lca; do
  if [[ "$artery" == rca ]]; then raw="$RAW_RCA"; else raw="$RAW_LCA"; fi
  [[ -d "$raw" ]] || { echo "Missing raw-vessel directory: $raw" >&2; exit 1; }
  for condition in "${conditions[@]}"; do
    prediction_dir="$PRED_ROOT/$artery/evaluation_inacc_view/$condition/test/predictions/views_2"
    [[ -d "$prediction_dir" ]] || { echo "Missing predictions: $prediction_dir" >&2; exit 1; }
    compgen -G "$prediction_dir/*.npz" > /dev/null || { echo "No NPZ predictions: $prediction_dir" >&2; exit 1; }
  done
done

for artery in rca lca; do
  if [[ "$artery" == rca ]]; then raw="$RAW_RCA"; else raw="$RAW_LCA"; fi
  for condition in "${conditions[@]}"; do
    echo "[STARTING] DeepCA $artery $condition, 2 views"
    "$PYTHON" scripts/evaluate_fair_comparison.py \
      --prediction-method deepca \
      --artery "$artery" \
      --prediction-dir "$PRED_ROOT/$artery/evaluation_inacc_view/$condition/test/predictions/views_2" \
      --raw-vessel-dir "$raw" \
      --options-json "$OPTIONS" \
      --output-dir "$OUT/deepca_${artery}_views_2_inacc_view/$condition"
    echo "[COMPLETED] DeepCA $artery $condition, 2 views"
  done
done
