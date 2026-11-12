#!/usr/bin/env bash
# Run from the AutoCAR repository with its evaluation environment activated.
set -euo pipefail

: "${RAW_RCA:?Set RAW_RCA to the RCA raw-vessel directory}"
: "${RAW_LCA:?Set RAW_LCA to the LCA raw-vessel directory}"
: "${OPTIONS:?Set OPTIONS to the shared fair-comparison options JSON}"
: "${OUT:?Set OUT to the fair-comparison output root}"
PRED_ROOT="${PRED_ROOT:-/export/home2/reny0012/result/deepca_imagecas}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
overwrite_args=()
if [[ "${OVERWRITE:-0}" == 1 ]]; then overwrite_args=(--overwrite); fi
conditions=(error_0p5deg error_1deg error_2deg error_4deg)

[[ -f scripts/evaluate_fair_comparison.py ]] || { echo 'Run from the AutoCAR repository.' >&2; exit 1; }
[[ -f scripts/evaluate_prediction_directory.py ]] || { echo "Missing overlap evaluator" >&2; exit 1; }
[[ -f "$SCRIPT_DIR/summarize_deepca_view_error.py" ]] || { echo "Missing table collector beside this script" >&2; exit 1; }
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
      --protocol-version 1 \
      --artery "$artery" \
      --prediction-dir "$PRED_ROOT/$artery/evaluation_inacc_view/$condition/test/predictions/views_2" \
      --raw-vessel-dir "$raw" \
      --options-json "$OPTIONS" \
      --output-dir "$OUT/deepca_${artery}_views_2_inacc_view/$condition" \
      "${overwrite_args[@]}"
    "$PYTHON" scripts/evaluate_prediction_directory.py \
      --allow-clipped-raw-reference \
      --artery "$artery" \
      --prediction-method deepca \
      --prediction-dir "$PRED_ROOT/$artery/evaluation_inacc_view/$condition/test/predictions/views_2" \
      --allow-missing-view-indices \
      --raw-vessel-dir "$raw" \
      --output-dir "$OUT/deepca_${artery}_views_2_inacc_view_overlap/$condition" \
      --evaluation-bbox-min-xyz-mm -58.5 -58.5 -58.5 \
      --evaluation-bbox-max-xyz-mm 58.5 58.5 58.5 \
      --center-offset-tolerance-mm 1 \
      "${overwrite_args[@]}"
    echo "[COMPLETED] DeepCA $artery $condition, 2 views"
  done
done

"$PYTHON" "$SCRIPT_DIR/summarize_deepca_view_error.py" --output-root "$OUT"
