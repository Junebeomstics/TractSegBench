#!/bin/bash -l 


source /home/junb/miniconda3/bin/activate
conda activate masam

cd /mnt/scratch/junb/TractSegVis/TractSeg

DATA_PATH="/mnt/scratch/junb/TractSegVis/"

# New script
# # from all HCP (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg --en wholeHCP_experiment_TractSeg_inference --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP_experiment_TractSeg_x14/best_weights_ep73.npz --seg &
# sleep 20
# CUDA_VISIBLE_DEVICES=1 ./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_TractSeg --en CamCan_experiment_TractSeg_inference --train False --test True --lw  --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg_x1/best_weights_ep224.npz &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Ping_experiment_TractSeg --en Ping_experiment_TractSeg_inference --train False --test False --lw  --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg_x8/best_weights_ep183.npz --seg &
# sleep 20

# # TractSeg 3D from HCP (same)
# CUDA_VISIBLE_DEVICES=3 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_TractSeg3D --en wholeHCP_experiment_TractSeg3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP_experiment_TractSeg3D_x14/best_weights_ep17.npz --seg & # --seg --rt --with_id TRAC-469

# wait
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_TractSeg3D --en CamCan_experiment_TractSeg3D_inference  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_TractSeg3D_x1/best_weights_ep58.npz  --seg & # --seg --rt --with_id TRAC-469
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Ping_experiment_TractSeg3D --en Ping_experiment_TractSeg3D_inference  --train False --test False --lw --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_TractSeg3D_x1/best_weights_ep51.npz --seg & # --seg --rt --with_id TRAC-469
sleep 20

# # # swinunetr from HCP (same)
#CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr --en run_with_wholeHCP_swinunetr_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP_swinunetr_x11/best_weights_ep67.npz --seg & # --seg --rt --with_id TRAC-469
# sleep 20
# CUDA_VISIBLE_DEVICES=3 ./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_swinunetr --en run_with_CamCan_swinunetr_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr_x2/best_weights_ep205.npz --seg & # --seg --rt --with_id TRAC-469

# wait
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Ping_experiment_swinunetr --en run_with_Ping_swinunetr_inference --train False --test False --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr_x17/best_weights_ep174.npz --seg & # --seg --rt --with_id TRAC-469
# sleep 20

# # # swinunetr3D from HCP (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_swinunetr3D --en run_with_wholeHCP_swinunetr3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP_swinunetr3D_x12/best_weights_ep26.npz --seg  # --seg --rt --with_id TRAC-469
# wait
# CUDA_VISIBLE_DEVICES=2 ./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_swinunetr3D --en run_with_CamCan_swinunetr3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_CamCan_swinunetr3D_x2/best_weights_ep20.npz --seg & # --seg --rt --with_id TRAC-469
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Ping_experiment_swinunetr3D --en run_with_Ping_swinunetr3D_inference --train False --test False --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_swinunetr3D_x7/best_weights_ep36.npz --seg & # --seg --rt --with_id TRAC-469
# sleep 20

# # # MASAM3D from HCP (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_MASAM3D_x1 --en wholeHCP_experiment_MASAM3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_wholeHCP_MASAM3D_x22/best_weights_ep28.npz --seg & # --seg --rt --with_id TRAC-469
# sleep 20
# CUDA_VISIBLE_DEVICES=1 ./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_MASAM3D_x1 --en CamCan_experiment_MASAM3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_MASAM3D/best_weights_ep45.npz --seg &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Ping_experiment_MASAM3D_x1 --en Ping_experiment_MASAM3D_inference --train False --test False --lw --weights_path /mnt/storage/junb/hcp_ecp/Ping_experiment_MASAM3D/best_weights_ep38.npz --seg &
# sleep 20

# # # MedNext3D from HCP (same)
# CUDA_VISIBLE_DEVICES=3 ./ExpRunner --fold 0 --data_path $DATA_PATH --config wholeHCP_experiment_mednext3D --en wholeHCP_experiment_mednext3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/wholeHCP_experiment_mednext3D_x20/best_weights_ep17.npz --seg & # --seg --rt --with_id TRAC-469
# wait
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config CamCan_experiment_mednext3D --en CamCan_experiment_mednext3D_inference --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/CamCan_experiment_mednext3D/best_weights_ep26.npz --seg & # --seg --rt --with_id TRAC-469
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Ping_experiment_mednext3D --en Ping_experiment_mednext3D_inference --train False --test False --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_Ping_mednext3D/best_weights_ep74.npz --seg & # --seg --rt --with_id TRAC-469
# wait

