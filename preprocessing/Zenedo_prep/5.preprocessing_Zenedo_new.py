import os
from os.path import join
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import data_utils
from tractseg.data.subjects import get_all_subjects
from tractseg.libs import exp_utils

dataset = "HCP_final"
DATASET_FOLDER = "HCP_for_training_Zenedo"
DATASET_FOLDER_PREPROC = "HCP_preproc_Zenedo_fixed_mask"
DATA_PATH = '/global/cfs/cdirs/m4673/junbeom/TractSegVis'

filenames_data = ['aligned_peaks']
filenames_seg = ['sorted_bundle_masks']

def create_global_brain_mask(subjects):
    """
    Generate a single brain mask from all subjects, keeping only voxels that contain at least one nonzero value.
    """
    mask = None
    all_files = filenames_data + filenames_seg
    
    for subject in subjects:
        for filename in all_files:
            path_src = join(DATA_PATH, DATASET_FOLDER, subject, filename + ".nii.gz")
            if os.path.exists(path_src):
                data = nib.load(path_src).get_fdata()
                data = np.nan_to_num(data)
                
                # Create binary mask
                current_mask = (data != 0).any(axis=-1) if data.ndim == 4 else (data != 0)
                
                # Merge with existing mask
                mask = current_mask if mask is None else mask | current_mask
            else:
                print(f"Missing file: {subject}-{filename}")
                #raise IOError("File missing")
    
    # Save global mask
    mask_path = join(os.getcwd(), "global_brain_mask.nii.gz")
    exp_utils.make_dir(DATASET_FOLDER_PREPROC)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), mask_path)
    print(f"Saved global mask: {mask_path}")
    
    return mask

def create_preprocessed_files(subject, mask, bbox):
    """
    Crop images based on generated global brain mask and save preprocessed files.
    """
    for filename in filenames_data + filenames_seg:
        path_src = join(DATA_PATH, DATASET_FOLDER, subject, filename + ".nii.gz")
        path_target = join(DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")
        
        if os.path.exists(path_src):
            img = nib.load(path_src)
            data = img.get_fdata()
            affine = img.affine
            data = np.nan_to_num(data)
            data, _, _, _ = data_utils.crop_to_nonzero(data, bbox=bbox)
            nib.save(nib.Nifti1Image(data, affine), path_target)
            print(f"Saved cropped file: {path_target}")
        else:
            print(f"Missing file: {subject}-{filename}")
            raise IOError("File missing")

if __name__ == "__main__":
    print("Output folder:", DATASET_FOLDER_PREPROC)
    subjects = get_all_subjects(dataset=dataset)
    if not os.path.exists(join(os.getcwd(), "global_brain_mask.nii.gz"):
        global_mask = create_global_brain_mask(subjects)
    else:
        global_mask = nib.load(join(os.getcwd(), "global_brain_mask.nii.gz")).get_fdata()
    _, _, global_bbox, _ = data_utils.crop_to_nonzero(global_mask)
    Parallel(n_jobs=12)(delayed(create_preprocessed_files)(subject, global_mask, global_bbox) for subject in subjects)
