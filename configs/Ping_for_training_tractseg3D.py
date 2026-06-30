#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATASET = "Ping"
    DATA_PATH = os.environ.get("TRACTSEG_DATASET_ROOT", "/mnt/storage/junb/TractSeg_datasets")
    DATASET_FOLDER = "PIng_preproc_brainlife_fixed_dwi_aligned"
    FEATURES_FILENAME = "aligned_peaks"
    LABELS_FILENAME = "corrected_bundle_masks"
    NR_OF_CLASSES = 61
    NR_SLICES = 1
    BATCH_SIZE = 2
    COMPILE = False
    CLASSES = "Brainlife"

    MODEL = "UNet3D_Pytorch_DeepSup_sm"
    DIM = "3D"
    SLICE_DIRECTION = "y"
    INPUT_DIM = (144, 144, 144)
    TRAINING_SLICE_DIRECTION = "xyz"
    TYPE = "single_direction"
    RESOLUTION = "1.25mm"

    WEIGHTS_PATH = ""
    LOAD_WEIGHTS = False

    USE_DROPOUT = False
    FP16 = True
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"
    LOSS_FUNCTION = "default"

    LOG_PER_BUNDLE = True
    DATA_AUGMENTATION = True
    GRADIENT_CLIP = 1

    UPSAMPLE_TYPE = "trilinear"
    NUM_PROCESSES = 4
    LR_SCHEDULE_TYPE = "CosineAnnealingLR"
