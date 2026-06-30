#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.base import Config as BaseConfig
# originally load config from base.py -> tract_seg.py -> my_custom_experiment.py
# skipped tract_seg.py

class Config(BaseConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0] # filename becomes experiment name

    DATASET = "HCP"  # HCP (105) | HCP_all (1061) | HCP_vis
    DATA_PATH= "/pscratch/sd/j/junbeom/"
    DATASET_FOLDER = "HCP_preproc_brainlife_fixed" # "HCP_for_training"
    FEATURES_FILENAME = "aligned_peaks"  # "12g90g270g_CSD_BX" # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
    LABELS_FILENAME = "aligned_corrected_bundle_masks" # if not set, predefined LABELS_FILENAME is used.
    NR_OF_CLASSES = 61 # number of output channel
    NR_SLICES = 1
    BATCH_SIZE = 1 # 128 | 47 : number of slices per batch
    VAL_BATCH_SIZE = 2
    COMPILE = True
    CLASSES = "Brainlife"

    MODEL = "SwinUNETR"
    INPUT_DIM = (160,160,160)
    DIM = "3D"  # 2D | 3D
    FEATURE_SIZE = 24
    SLICE_DIRECTION = "y"  # x | y | z  ("combined" needs z)
    TRAINING_SLICE_DIRECTION = "xyz"  # y | xyz
    TYPE = "single_direction"  
    # single_direction | combined 
    # single_direction use nifti files as input (sampling randomly), while combined uses predetermined slices from numpy files
    RESOLUTION = "1.25mm"  # 1.25mm|2.5mm

    WEIGHTS_PATH = ""  # if empty string: autoloading the best_weights in get_best_weights_path()
    LOAD_WEIGHTS = False # if True and WEIGHTS_PATH is empty: autoloading the best_weights in get_best_weights_path()

   
    # slightly less overfitting (but max f1_validate maybe slightly worse (makes sense if less overfitting))
    USE_DROPOUT = False
    FP16 = True # True 시 validation 시에 index error 발생
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.0001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"  # min | max
    LOSS_FUNCTION = "default" # default | soft_batch_dice
    LR_WARMUP_RATIO = 0.1

    LOG_PER_BUNDLE = True
    DATA_AUGMENTATION = True
    
    NUM_PROCESSES = 8
    LR_SCHEDULE_TYPE = "CosineAnnealingLR"
    #GRADIENT_CLIP = 1
    
