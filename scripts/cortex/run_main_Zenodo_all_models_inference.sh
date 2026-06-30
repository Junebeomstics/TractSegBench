#!/bin/bash -l 

source /home/junb/miniconda3/bin/activate
conda activate masam

cd /mnt/scratch/junb/TractSegVis/TractSeg

DATA_PATH="/mnt/scratch/junb/TractSegVis/"

#  HCP to HCP
# # TractSeg2D from Zenedo (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_sorted_aligned_x4 --en Zenedo_to_Zenedo_experiment_TractSeg_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_sorted_aligned_x4/best_weights_ep311.npz &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config Zenedo_experiment_sorted_aligned_x4 --en Zenedo_to_Zenedo_experiment_TractSeg_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_sorted_aligned_x13/best_weights_ep181.npz &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config Zenedo_experiment_sorted_aligned_x4 --en Zenedo_to_Zenedo_experiment_TractSeg_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_sorted_aligned_x14/best_weights_ep201.npz &
# sleep 20

# # TractSeg3D from Zenedo (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_3D --en Zenedo_to_Zenedo_experiment_TractSeg3D_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_3D_x34/best_weights_ep67.npz &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config Zenedo_experiment_3D --en Zenedo_to_Zenedo_experiment_TractSeg3D_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_3D_x31/best_weights_ep73.npz &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config Zenedo_experiment_3D --en Zenedo_to_Zenedo_experiment_TractSeg3D_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_3D_x33/best_weights_ep175.npz &
# wait

# swinunetr from Zenedo (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr --en Zenedo_to_Zenedo_experiment_swinunetr_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/Zenedo_to_Zenedo_experiment_swinunetr_inference_x6/best_weights_ep143.npz & 
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr --en Zenedo_to_Zenedo_experiment_swinunetr_inference_x2 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_swinunetr_x8/best_weights_ep244.npz & 
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr --en Zenedo_to_Zenedo_experiment_swinunetr_inference_x3 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_swinunetr_x9/best_weights_ep249.npz & 
# sleep 20

# swinunetr3D from Zenedo (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr_3D --en Zenedo_to_Zenedo_experiment_swinunetr3D_inference_x1  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_swinunetr_3D_x7/best_weights_ep181.npz & 
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr_3D --en Zenedo_to_Zenedo_experiment_swinunetr3D_inference_x2  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_swinunetr_3D_x8/best_weights_ep139.npz & 
# sleep 20
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr_3D --en Zenedo_to_Zenedo_experiment_swinunetr3D_inference_x3  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_swinunetr_3D_x9/best_weights_ep128.npz & 
# sleep 20

#training
#CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr_3D --en Zenedo_experiment_swinunetr3D_x1  --train True --test True # --lw --weights_path /mnt/storage/junb/hcp_ecp/run_with_default_settings_swinunetr_3D_x7/best_weights_ep181.npz & 

#inference
CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_swinunetr_3D --en Zenedo_experiment_swinunetr3D_x1  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/Zenedo_experiment_swinunetr3D_x1/best_weights_ep49.npz --seg


# # MedNext3D from Zenedo (same)
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenodo_experiment_mednext3D --en Zenedo_to_Zenedo_experiment_mednext3D_inference_x1  --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/Zenodo_experiment_mednext3D_x8/best_weights_ep63.npz & 
# sleep 20


# # MaSAM3D from Zenedo (same)
#training 
#CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_MASAM3D_x1 --en Zenedo_to_Zenedo_experiment_MASAM3D_inference_x1 --train True --test True --lw --weights_path $DATA_PATH/MA-SAM/checkpoints/sam_vit_l_0b3195.pth 

#inference
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 0 --data_path $DATA_PATH --config Zenedo_experiment_MASAM3D_x1 --en Zenedo_to_Zenedo_experiment_MASAM3D_inference_x1 --train False --test True --lw --weights_path /mnt/storage/junb/hcp_ecp/Zenedo_to_Zenedo_experiment_MASAM3D_inference_x1/best_weights_ep298.npz & 
# wait

# sleep 30
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config Zenedo_experiment_MASAM3D_x1 --en Zenedo_to_Zenedo_experiment_MASAM3D_inference_x2 --train True --test True --lw --weights_path $DATA_PATH/MA-SAM/checkpoints/sam_vit_l_0b3195.pth 
# sleep 30
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config Zenedo_experiment_MASAM3D_x1 --en Zenedo_to_Zenedo_experiment_MASAM3D_inference_x3 --train True --test True --lw --weights_path $DATA_PATH/MA-SAM/checkpoints/sam_vit_l_0b3195.pth 
# wait

# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 1 --data_path $DATA_PATH --config Zenedo_experiment_mednext3D --en Zenedo_to_Zenedo_experiment_mednext3D_inference_x2  --train True --test True  
# wait
# CUDA_VISIBLE_DEVICES=0 ./ExpRunner --fold 2 --data_path $DATA_PATH --config Zenedo_experiment_mednext3D --en Zenedo_to_Zenedo_experiment_mednext3D_inference_x3  --train True --test True 
# wait
