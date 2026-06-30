#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_hcp_to_ping_direct_paths.sh <dataset_abs_path> <tag> [gpu_id]
# Example:
#   bash run_hcp_to_ping_direct_paths.sh /mnt/storage/junb/Ping_preproc_brainlife_fixed storageorig 1
#   bash run_hcp_to_ping_direct_paths.sh /mnt/scratch/junb/TractSegVis/PIng_preproc_brainlife_fixed_dwi_aligned alignedreal 1

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset_abs_path> <tag> [gpu_id]"
  exit 1
fi

DATASET_ABS_PATH="$1"
TAG="$2"
GPU_ID="${3:-1}"

if [[ ! -d "$DATASET_ABS_PATH" ]]; then
  echo "Dataset path does not exist: $DATASET_ABS_PATH"
  exit 1
fi

DATA_PATH="/mnt/scratch/junb/TractSegVis"
TRSEG_DIR="$DATA_PATH/TractSeg"
EXPERIMENT_ROOT="/mnt/storage/junb/hcp_ecp"

cd "$TRSEG_DIR"

echo "[INFO] Start HCP->PING inference"
echo "[INFO] dataset_path=$DATASET_ABS_PATH"
echo "[INFO] tag=$TAG"
echo "[INFO] gpu_id=$GPU_ID"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_inference() {
  local fold="$1"
  local config_name="$2"
  local experiment_name="$3"
  local weights_path="$4"
  local add_gradient_clip="${5:-0}"

  local score_file="$EXPERIMENT_ROOT/$experiment_name/score_test${fold}.pkl"

  if [[ -f "$score_file" ]]; then
    echo "[$(timestamp)] [SKIP] $experiment_name (found $score_file)"
    return
  fi

  if [[ ! -f "$weights_path" ]]; then
    echo "[$(timestamp)] [ERROR] Missing weight: $weights_path"
    exit 1
  fi

  echo "[$(timestamp)] [RUN] $experiment_name"

  local cmd=(
    ./ExpRunner
    --fold "$fold"
    --data_path "$DATA_PATH"
    --config "$config_name"
    --en "$experiment_name"
    --train False
    --test True
    --lw
    --weights_path "$weights_path"
    --seg
  )

  if [[ "$add_gradient_clip" == "1" ]]; then
    cmd+=(--gradient_clip 1)
  fi

  WANDB_MODE=disabled \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  TRACTSEG_PING_DATASET_FOLDER="$DATASET_ABS_PATH" \
  conda run -n masam_blackwell "${cmd[@]}"

  if [[ ! -f "$score_file" ]]; then
    echo "[$(timestamp)] [ERROR] Expected score file was not generated: $score_file"
    exit 1
  fi

  echo "[$(timestamp)] [DONE] $experiment_name"
}

# TractSeg2D
run_inference 0 Ping_experiment_TractSeg    "HCP_to_PING_${TAG}_experiment_TractSeg_inference_x1"    "$EXPERIMENT_ROOT/wholeHCP_experiment_TractSeg_x14/best_weights_ep73.npz"
run_inference 1 Ping_experiment_TractSeg    "HCP_to_PING_${TAG}_experiment_TractSeg_inference_x2"    "$EXPERIMENT_ROOT/wholeHCP_experiment_TractSeg_x1/best_weights_ep68.npz"
run_inference 2 Ping_experiment_TractSeg    "HCP_to_PING_${TAG}_experiment_TractSeg_inference_x3"    "$EXPERIMENT_ROOT/wholeHCP_experiment_TractSeg_x14/best_weights_ep73.npz"

# TractSeg3D
run_inference 0 Ping_experiment_TractSeg3D  "HCP_to_PING_${TAG}_experiment_TractSeg3D_inference_x1"  "$EXPERIMENT_ROOT/wholeHCP_experiment_TractSeg3D_x14/best_weights_ep17.npz"
run_inference 1 Ping_experiment_TractSeg3D  "HCP_to_PING_${TAG}_experiment_TractSeg3D_inference_x2"  "$EXPERIMENT_ROOT/wholeHCP_experiment_TractSeg3D_x1/best_weights_ep13.npz"
run_inference 2 Ping_experiment_TractSeg3D  "HCP_to_PING_${TAG}_experiment_TractSeg3D_inference_x3"  "$EXPERIMENT_ROOT/wholeHCP_experiment_TractSeg3D_x14/best_weights_ep17.npz"

# SwinUNETR2D
run_inference 0 Ping_experiment_swinunetr   "HCP_to_PING_${TAG}_experiment_swinunetr_inference_x1"   "$EXPERIMENT_ROOT/run_with_wholeHCP_swinunetr_x11/best_weights_ep67.npz"
run_inference 1 Ping_experiment_swinunetr   "HCP_to_PING_${TAG}_experiment_swinunetr_inference_x2"   "$EXPERIMENT_ROOT/run_with_wholeHCP_swinunetr_x12/best_weights_ep28.npz"
run_inference 2 Ping_experiment_swinunetr   "HCP_to_PING_${TAG}_experiment_swinunetr_inference_x3"   "$EXPERIMENT_ROOT/run_with_wholeHCP_swinunetr_x13/best_weights_ep22.npz"

# SwinUNETR3D
run_inference 0 Ping_experiment_swinunetr3D "HCP_to_PING_${TAG}_experiment_swinunetr3D_inference_x1" "$EXPERIMENT_ROOT/run_with_wholeHCP_swinunetr3D_x12/best_weights_ep26.npz"
run_inference 1 Ping_experiment_swinunetr3D "HCP_to_PING_${TAG}_experiment_swinunetr3D_inference_x2" "$EXPERIMENT_ROOT/run_with_wholeHCP_swinunetr3D_x80/best_weights_ep6.npz"
run_inference 2 Ping_experiment_swinunetr3D "HCP_to_PING_${TAG}_experiment_swinunetr3D_inference_x3" "$EXPERIMENT_ROOT/run_with_wholeHCP_swinunetr3D_x81/best_weights_ep6.npz"

# MASAM3D
run_inference 0 Ping_experiment_MASAM3D_x1  "HCP_to_PING_${TAG}_experiment_MASAM3D_inference_x1"    "$EXPERIMENT_ROOT/run_with_wholeHCP_MASAM3D_x22/best_weights_ep28.npz" 1
run_inference 1 Ping_experiment_MASAM3D_x1  "HCP_to_PING_${TAG}_experiment_MASAM3D_inference_x2"    "$EXPERIMENT_ROOT/run_with_wholeHCP_MASAM3D_x1/best_weights_ep27.npz"  1
run_inference 2 Ping_experiment_MASAM3D_x1  "HCP_to_PING_${TAG}_experiment_MASAM3D_inference_x3"    "$EXPERIMENT_ROOT/run_with_wholeHCP_MASAM3D_x23/best_weights_ep37.npz" 1

# MedNeXt3D
run_inference 0 Ping_experiment_mednext3D   "HCP_to_PING_${TAG}_experiment_mednext3D_inference_x1"   "$EXPERIMENT_ROOT/wholeHCP_experiment_mednext3D_x20/best_weights_ep17.npz"
run_inference 1 Ping_experiment_mednext3D   "HCP_to_PING_${TAG}_experiment_mednext3D_inference_x2"   "$EXPERIMENT_ROOT/wholeHCP_experiment_mednext3D_x21/best_weights_ep18.npz"
run_inference 2 Ping_experiment_mednext3D   "HCP_to_PING_${TAG}_experiment_mednext3D_inference_x3"   "$EXPERIMENT_ROOT/wholeHCP_experiment_mednext3D_x22/best_weights_ep19.npz"

echo "[$(timestamp)] [ALL DONE] tag=$TAG"
