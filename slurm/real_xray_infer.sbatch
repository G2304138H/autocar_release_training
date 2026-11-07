#!/bin/bash
#SBATCH -J autocar_xray
#SBATCH -o /export/home2/reny0012/result/runlog/autocar_xray_%j.log
#SBATCH -e /export/home2/reny0012/result/runlog/autocar_xray_%j.err
#SBATCH -p RTXA6Kq
#SBATCH -w node11
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8

# GPU job: AutoCAR reconstruction of real angiograms exported by angioseg-nnunet step 08 (one job, all cases of one
# artery). Same inputs, view choice, camera, centring and summary sheet as the PPCAR-Net real-xray branch.
# Submit from the repository root:
#   test, 5 cases:  sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2,LIMIT=5,OUTPUT_MODE=summary slurm/real_xray_infer.sbatch
#   all cases:      sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2,OUTPUT_MODE=summary slurm/real_xray_infer.sbatch
#   checkpoint:     CKPT=/path/to/checkpoint.ckpt (required; one per artery)
#   options:        OUTPUT_MODE=full|summary (default full), CAMERA=acquisition|training (default acquisition),
#                   CENTER=mask_centroid|bbox_center|none (default mask_centroid), NUM_VIEWS=2,
#                   CASES="Study00605_RCA Study00087_RCA", VENV=..., REPO_ROOT=...
# Input:   $RECON_INPUTS/<METHOD>/<ART>/*.npz          (step 08 output)
# Output:  $RECON_OUTPUTS/autocar_<METHOD>_<camera>[_<CENTER>][_summary][_first<LIMIT>|_selected]/<ART>/

set -euo pipefail

module load python3 2>/dev/null || true
source "${VENV:-/export/home2/reny0012/vir_env/autocar}/bin/activate"
REPO_ROOT="${REPO_ROOT:-/export/home2/reny0012/code/autocar_release_training}"
cd "$REPO_ROOT"

ART="${ART:?set ART=lca or ART=rca}"
METHOD="${METHOD:?set METHOD=<step-08 output folder>, e.g. xravos_A_v2}"
CKPT="${CKPT:?set CKPT=/path/to/<ART> checkpoint .ckpt}"
CAMERA="${CAMERA:-acquisition}"
CENTER="${CENTER:-mask_centroid}"
OUTPUT_MODE="${OUTPUT_MODE:-full}"
RECON_INPUTS="${RECON_INPUTS:-/dataset/reny0012/angioseg/recon_inputs}"
RECON_OUTPUTS="${RECON_OUTPUTS:-/dataset/reny0012/angioseg/recon_outputs}"
TAG="autocar_${METHOD}_${CAMERA}"
[ "$CENTER" != "none" ] && TAG="${TAG}_${CENTER}"
[ "$OUTPUT_MODE" = "summary" ] && TAG="${TAG}_summary"
EXTRA=(--checkpoint "$CKPT" --camera "$CAMERA" --input_centering "$CENTER" --num_views "${NUM_VIEWS:-2}"
       --output_mode "$OUTPUT_MODE")
[ -n "${LIMIT:-}" ] && { EXTRA+=(--limit "$LIMIT"); TAG="${TAG}_first${LIMIT}"; }
[ -n "${CASES:-}" ] && { EXTRA+=(--case_ids $CASES); TAG="${TAG}_selected"; }

echo "[autocar] host=$(hostname) job=${SLURM_JOB_ID:-none} python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[autocar] $RECON_INPUTS/$METHOD/$ART -> $RECON_OUTPUTS/$TAG/$ART  (checkpoint $CKPT, camera $CAMERA, centring $CENTER)"

python -m src.real_xray --config "configs/real_xray/$ART.json" \
  --input_dir "$RECON_INPUTS/$METHOD/$ART" --output_dir "$RECON_OUTPUTS/$TAG/$ART" "${EXTRA[@]}"
