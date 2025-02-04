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
    NR_SLICES = 1 

    WEIGHTS_PATH = "/global/cfs/projectdirs/m4673/junbeom/TractSegVis/TractSeg/pretrained_weights_tract_segmentation_v3.npz"  # if empty string: autoloading the best_weights in get_best_weights_path()
 