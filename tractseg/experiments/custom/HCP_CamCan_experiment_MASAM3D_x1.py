#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.base import Config as BaseConfig
# originally load config from base.py -> tract_seg.py -> my_custom_experiment.py
# skipped tract_seg.py

class Config(BaseConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0] # filename becomes experiment name

    DATASET = ["HCP_all","CamCan"]  # HCP (105) | HCP_all (1061) | HCP_vis
    DATA_PATH= "/pscratch/sd/j/junbeom/"
    DATASET_FOLDER = ["HCP_preproc_brainlife_fixed","CamCan_preproc_brainlife_fixed"]# "HCP_for_training"
    FEATURES_FILENAME = ["aligned_peaks","aligned_peaks"]  # "12g90g270g_CSD_BX" # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
    LABELS_FILENAME = ["aligned_corrected_bundle_masks", "corrected_bundle_masks"] # if not set, predefined LABELS_FILENAME is used.
    NR_OF_CLASSES = 61 # number of output channel
    NR_SLICES = 1
    BATCH_SIZE = 1 # 128 | 47 : number of slices per batch
    COMPILE = True
    CLASSES = "Brainlife"

    MODEL = "MASAM"
    rank = 32
    scale = 1
    module = "sam_fact_tt_image_encoder"
    vit_name = 'vit_l'
    COMPILE = True
    DIM = "3D"  # 2D | 3D
    INPUT_DIM = (144,144,144) # Replace img_size
    SLICE_DIRECTION = "x"  # x | y | z  ("combined" needs z)
    TRAINING_SLICE_DIRECTION = "xyz"  # y | xyz
    TYPE = "single_direction"  
    WEIGHTS_PATH = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/checkpoints/sam_vit_l_0b3195.pth" # "/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/model_swinvit.pt"
    # "/grand/NeuroX/junbeom/MA-SAM/MA-SAM/checkpoints/sam_vit_b_01ec64.pth"
    # /grand/NeuroX/junbeom/MA-SAM/MA-SAM/checkpoints/sam_vit_h_4b8939.pth
    # /grand/NeuroX/junbeom/MA-SAM/MA-SAM/checkpoints/sam_vit_l_0b3195.pth
    
    # single_direction | combined 
    # single_direction use nifti files as input (sampling randomly), while combined uses predetermined slices from numpy files
    RESOLUTION = "1.25mm"  # 1.25mm|2.5mm

    LOAD_WEIGHTS = False # if True and WEIGHTS_PATH is empty: autoloading the best_weights in get_best_weights_path()

   
    # slightly less overfitting (but max f1_validate maybe slightly worse (makes sense if less overfitting))
    USE_DROPOUT = False
    FP16 = True # True 시 validation 시에 index error 발생
    USE_CONSECUTIVE_SLICES = False
    NUM_EPOCHS = 400
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"  # min | max
    LOSS_FUNCTION = "default" # default | soft_batch_dice

    DATA_AUGMENTATION = True
    LOG_PER_BUNDLE = True
    LR_SCHEDULE_TYPE = "CosineAnnealingLR"

    
