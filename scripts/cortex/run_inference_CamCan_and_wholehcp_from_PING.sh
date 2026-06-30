#!/bin/bash -l 


source /home/junb/miniconda3/bin/activate
conda activate masam

cd /mnt/scratch/junb/TractSegVis/TractSeg

DATA_PATH="/mnt/scratch/junb/TractSegVis/"

# from all HCP (same)

# TractSeg2D
./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en PING_to_HCP_experiment_TractSeg_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x9/best_weights_ep105.npz --seg &
sleep 20
./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_TractSeg --en PING_to_CamCan_experiment_TractSeg_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x9/best_weights_ep105.npz --seg &
sleep 20

./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en PING_to_HCP_experiment_TractSeg_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x10/best_weights_ep105.npz --seg &
sleep 20
./ExpRunner --fold 1 --data_path $DATA_PATH --config CamCan_experiment_TractSeg --en PING_to_CamCan_experiment_TractSeg_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x10/best_weights_ep105.npz --seg &
wait

./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en PING_to_HCP_experiment_TractSeg_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x12/best_weights_ep193.npz --seg &
sleep 20
./ExpRunner --fold 2 --data_path $DATA_PATH --config CamCan_experiment_TractSeg --en PING_to_CamCan_experiment_TractSeg_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x12/best_weights_ep193.npz --seg &
sleep 20

#TractSeg3D
./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en PING_to_HCP_experiment_TractSeg3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x1/best_weights_ep51.npz --seg &
sleep 20
./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_TractSeg3D --en PING_to_CamCan_experiment_TractSeg3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x1/best_weights_ep51.npz --seg &
sleep 20

./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en PING_to_HCP_experiment_TractSeg3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x2/best_weights_ep53.npz --seg &
sleep 20
./ExpRunner --fold 1 --data_path $DATA_PATH --config CamCan_experiment_TractSeg3D --en PING_to_CamCan_experiment_TractSeg3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x2/best_weights_ep53.npz --seg &
sleep 20

./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en PING_to_HCP_experiment_TractSeg3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x3/best_weights_ep35.npz --seg &
sleep 20
./ExpRunner --fold 2 --data_path $DATA_PATH --config CamCan_experiment_TractSeg3D --en PING_to_CamCan_experiment_TractSeg3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x3/best_weights_ep35.npz --seg &
wait

#swinunetr
./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en PING_to_HCP_experiment_swinunetr_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x17/best_weights_ep174.npz --seg &
sleep 20
./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_swinunetr --en PING_to_CamCan_experiment_swinunetr_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x17/best_weights_ep174.npz --seg &
sleep 20

./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en PING_to_HCP_experiment_swinunetr_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x18/best_weights_ep104.npz --seg &
sleep 20
./ExpRunner --fold 1 --data_path $DATA_PATH --config CamCan_experiment_swinunetr --en PING_to_CamCan_experiment_swinunetr_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x18/best_weights_ep104.npz --seg &
wait

./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en PING_to_HCP_experiment_swinunetr_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x19/best_weights_ep111.npz --seg &
sleep 20
./ExpRunner --fold 2 --data_path $DATA_PATH --config CamCan_experiment_swinunetr --en PING_to_CamCan_experiment_swinunetr_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x19/best_weights_ep111.npz --seg &
sleep 20

#swinunetr3D
./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en PING_to_HCP_experiment_swinunetr3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x7/best_weights_ep36.npz --seg &
sleep 20
./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_swinunetr3D --en PING_to_CamCan_experiment_swinunetr3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x7/best_weights_ep36.npz --seg &
sleep 20

./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en PING_to_HCP_experiment_swinunetr3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x8/best_weights_ep33.npz --seg &
sleep 20
./ExpRunner --fold 1 --data_path $DATA_PATH --config CamCan_experiment_swinunetr3D --en PING_to_CamCan_experiment_swinunetr3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x8/best_weights_ep33.npz --seg &
sleep 20

./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en PING_to_HCP_experiment_swinunetr3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x9/best_weights_ep29.npz --seg &
sleep 20
./ExpRunner --fold 2 --data_path $DATA_PATH --config CamCan_experiment_swinunetr3D --en PING_to_CamCan_experiment_swinunetr3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x9/best_weights_ep29.npz --seg &
wait

#masam3D
./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en PING_to_HCP_experiment_MASAM3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D/best_weights_ep38.npz --seg &
sleep 20
./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_MASAM3D_x1 --en PING_to_CamCan_experiment_MASAM3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D/best_weights_ep38.npz --seg &
sleep 20

./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en PING_to_HCP_experiment_MASAM3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D_x1/best_weights_ep113.npz --seg &
sleep 20
./ExpRunner --fold 1 --data_path $DATA_PATH --config CamCan_experiment_MASAM3D_x1 --en PING_to_CamCan_experiment_MASAM3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D_x1/best_weights_ep113.npz --seg &
wait

./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en PING_to_HCP_experiment_MASAM3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D_x2/best_weights_ep131.npz --seg &
sleep 20
./ExpRunner --fold 2 --data_path $DATA_PATH --config CamCan_experiment_MASAM3D_x1 --en PING_to_CamCan_experiment_MASAM3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D_x2/best_weights_ep131.npz --seg &
sleep 20

#MedNext3D
./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en PING_to_HCP_experiment_mednext3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D/best_weights_ep74.npz --seg &
sleep 20
./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_mednext3D --en PING_to_CamCan_experiment_mednext3D_inference_x1 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D/best_weights_ep74.npz --seg &
wait

./ExpRunner --fold 1 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en PING_to_HCP_experiment_mednext3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D_x1/best_weights_ep40.npz --seg &
sleep 20
./ExpRunner --fold 1 --data_path $DATA_PATH --config CamCan_experiment_mednext3D --en PING_to_CamCan_experiment_mednext3D_inference_x2 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D_x1/best_weights_ep40.npz --seg &
wait

./ExpRunner --fold 2 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en PING_to_HCP_experiment_mednext3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D_x2/best_weights_ep62.npz --seg &
sleep 20
./ExpRunner --fold 2 --data_path $DATA_PATH --config CamCan_experiment_mednext3D --en PING_to_CamCan_experiment_mednext3D_inference_x3 --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D_x2/best_weights_ep62.npz --seg &
wait

