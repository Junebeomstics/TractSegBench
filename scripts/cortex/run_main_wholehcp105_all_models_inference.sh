#!/bin/bash -l 

source /home/junb/miniconda3/bin/activate
conda activate masam

cd /mnt/scratch/junb/TractSegVis/TractSeg

DATA_PATH="/mnt/scratch/junb/TractSegVis/"

#  HCP to HCP
# # TractSeg2D from HCP105 (same)
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP105_experiment_TractSeg --en HCP105_to_HCP105_experiment_TractSeg_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_TractSeg_x11/best_weights_ep183.npz &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP105_experiment_TractSeg --en HCP105_to_HCP105_experiment_TractSeg_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_TractSeg_x12/best_weights_ep191.npz &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP105_experiment_TractSeg --en HCP105_to_HCP105_experiment_TractSeg_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_TractSeg_x15/best_weights_ep231.npz &
sleep 20

# TractSeg3D from HCP105 (same)
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP105_experiment_TractSeg3D --en HCP105_to_HCP105_experiment_TractSeg3D_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_TractSeg3D_x11/best_weights_ep95.npz &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP105_experiment_TractSeg3D --en HCP105_to_HCP105_experiment_TractSeg3D_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_TractSeg3D_x12/best_weights_ep138.npz &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP105_experiment_TractSeg3D --en HCP105_to_HCP105_experiment_TractSeg3D_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_TractSeg3D_x9/best_weights_ep145.npz &
sleep 20

# swinunetr from HCP105 (same)
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP105_experiment_swinunetr --en HCP105_to_HCP105_experiment_swinunetr_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_swinunetr_x12/best_weights_ep206.npz & 
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP105_experiment_swinunetr --en HCP105_to_HCP105_experiment_swinunetr_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_swinunetr_x21/best_weights_ep102.npz & 
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP105_experiment_swinunetr --en HCP105_to_HCP105_experiment_swinunetr_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_swinunetr_x22/best_weights_ep197.npz & 
wait


# # swinunetr3D from HCP105 (same)
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP105_experiment_swinunetr3D --en HCP105_to_HCP105_experiment_swinunetr3D_inference_x1  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_swinunetr3D_x15/best_weights_ep117.npz & 
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP105_experiment_swinunetr3D --en HCP105_to_HCP105_experiment_swinunetr3D_inference_x2  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_swinunetr3D_x64/best_weights_ep21.npz & 
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP105_experiment_swinunetr3D --en HCP105_to_HCP105_experiment_swinunetr3D_inference_x3  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_swinunetr3D_x65/best_weights_ep11.npz & 
sleep 20

# # # # MASAM3D from HCP105 (same)
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP105_experiment_MASAM3D_x1 --en HCP105_to_HCP105_experiment_MASAM3D_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_MASAM3D_x9/best_weights_ep285.npz & 
wait
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP105_experiment_MASAM3D_x1 --en HCP105_to_HCP105_experiment_MASAM3D_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_MASAM3D_x10/best_weights_ep292.npz  &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP105_experiment_MASAM3D_x1 --en HCP105_to_HCP105_experiment_MASAM3D_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP105_MASAM3D_x12/best_weights_ep234.npz &
sleep 20

# # MedNext3D from HCP105 (same)
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP105_experiment_mednext3D --en HCP105_to_HCP105_experiment_mednext3D_inference_x1  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_mednext3D_x21/best_weights_ep160.npz & 
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP105_experiment_mednext3D --en HCP105_to_HCP105_experiment_mednext3D_inference_x2  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_mednext3D_x22/best_weights_ep191.npz & 
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP105_experiment_mednext3D --en HCP105_to_HCP105_experiment_mednext3D_inference_x3  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP105_experiment_mednext3D_x23/best_weights_ep197.npz & 
wait
