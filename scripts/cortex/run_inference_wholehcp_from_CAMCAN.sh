#!/bin/bash -l 


source /home/junb/miniconda3/bin/activate
conda activate masam

cd /mnt/scratch/junb/TractSegVis/TractSeg

DATA_PATH="/mnt/scratch/junb/TractSegVis/"

#From here

# TractSeg2D from CAMCAN
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en wholeHCP_experiment_TractSeg --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg_x1/best_weights_ep224.npz --seg &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en wholeHCP_experiment_TractSeg --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg_x4/best_weights_ep45.npz --seg &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en wholeHCP_experiment_TractSeg --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg_x5/best_weights_ep107.npz --seg &
sleep 20

# TractSeg3D from CAMCAN
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en wholeHCP_experiment_TractSeg3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg3D_x1/best_weights_ep58.npz --seg &
wait

CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en wholeHCP_experiment_TractSeg3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg3D_x6/best_weights_ep16.npz --seg &
sleep 20

#CUDA_VISIBLE_DEVICES=1 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en wholeHCP_experiment_TractSeg3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg3D_x6/best_weights_ep16.npz &
#sleep 20
# TractSeg3D fold1 and fold2 use the same exp_path as x6

# SwinUNETR2D from CAMCAN
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en wholeHCP_experiment_swinunetr --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr_x2/best_weights_ep205.npz --seg &
sleep 20

CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en wholeHCP_experiment_swinunetr --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr_x5/best_weights_ep68.npz --seg &
sleep 20

CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en wholeHCP_experiment_swinunetr --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr_x6/best_weights_ep65.npz --seg  &
wait

# SwinUNETR3D from CAMCAN
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en run_with_wholeHCP_swinunetr3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr3D_x2/best_weights_ep20.npz --seg &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en run_with_wholeHCP_swinunetr3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr3D_x6/best_weights_ep13.npz --seg &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en run_with_wholeHCP_swinunetr3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr3D_x6/best_weights_ep13.npz --seg &
sleep 20
# SwinUNETR3D fold1 and fold2 use the same exp_path as x6

# MASAM3D from CAMCAN
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en wholeHCP_experiment_MASAM3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_MASAM3D/best_weights_ep45.npz --seg  &
sleep 20
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en wholeHCP_experiment_MASAM3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_MASAM3D_x1/best_weights_ep36.npz --seg &
wait

CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en wholeHCP_experiment_MASAM3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_MASAM3D_x2/best_weights_ep50.npz --seg &
sleep 20


# MedNext3D from CAMCAN
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en wholeHCP_experiment_mednext3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_mednext3D/best_weights_ep26.npz --seg &
sleep 20

CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en wholeHCP_experiment_mednext3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_mednext3D_x5/best_weights_ep26.npz --seg &
sleep 20

CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en wholeHCP_experiment_mednext3D --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_mednext3D_x4/best_weights_ep30.npz --seg &
wait