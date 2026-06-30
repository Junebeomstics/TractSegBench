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
    DATASET_FOLDER = "HCP_preproc_Zenedo" # "HCP_for_training"
    FEATURES_FILENAME = "aligned_peaks"  # "12g90g270g_CSD_BX" # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
    LABELS_FILENAME = "sorted_bundle_masks" # if not set, predefined LABELS_FILENAME is used.
    NR_OF_CLASSES = 72 # number of output channel
    COMPILE = True

    MODEL = "SwinUNETR"
    DIM = "2D"  # 2D | 3D
    INPUT_DIM = (160,160)
    SLICE_DIRECTION = "y"  # x | y | z  ("combined" needs z)
    TRAINING_SLICE_DIRECTION = "xyz"  # y | xyz
    TYPE = "single_direction"  
    WEIGHTS_PATH = "" # "/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/model_swinvit.pt"
    # single_direction | combined 
    # single_direction use nifti files as input (sampling randomly), while combined uses predetermined slices from numpy files
    RESOLUTION = "1.25mm"  # 1.25mm|2.5mm

    LOAD_WEIGHTS = False # if True and WEIGHTS_PATH is empty: autoloading the best_weights in get_best_weights_path()

   
    # slightly less overfitting (but max f1_validate maybe slightly worse (makes sense if less overfitting))
    USE_DROPOUT = False
    FP16 = True # Causes an index error during validation when True
    BATCH_SIZE = 1 # 128 | 47 : number of slices per batch
    NR_SLICES = 47
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"  # min | max
    LOSS_FUNCTION = "default" # default | soft_batch_dice
    LOG_PER_BUNDLE = True

    os.environ["TRITON_MAX_BLOCK_X"] = "4096"
