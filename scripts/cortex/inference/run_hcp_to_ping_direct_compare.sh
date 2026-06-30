#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:-1}"
SCRIPT_DIR="/mnt/scratch/junb/TractSegVis/TractSeg/scripts/cortex/inference"
DATASET_ROOT="${TRACTSEG_DATASET_ROOT:-/mnt/storage/junb/TractSeg_datasets}"

bash "$SCRIPT_DIR/run_hcp_to_ping_direct_paths.sh" "$DATASET_ROOT/Ping_preproc_brainlife_fixed" storageorig_direct "$GPU_ID"
bash "$SCRIPT_DIR/run_hcp_to_ping_direct_paths.sh" "$DATASET_ROOT/PIng_preproc_brainlife_fixed_dwi_aligned" alignedreal_direct "$GPU_ID"
