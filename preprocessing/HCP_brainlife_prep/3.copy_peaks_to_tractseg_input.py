import os
import shutil

# Set base directory
base_dir = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/bl_derivative"
output_root = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP_for_training_brainlife"

# Iterate over each subject folder
for subject_id in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_id)
    
    # Set peaks file path
    peaks_path = os.path.join(subject_path, "neuro", "peaks", "6_28_2023.hcp.lmax8", "peaks.nii.gz")
    
    # Check whether the peaks file exists
    if os.path.isfile(peaks_path):
        # Set new output path: HCP/subject_id/
        output_dir = os.path.join(output_root, subject_id)
        output_tractmasks = os.path.join(output_dir,'bundle_masks.nii.gz')
        # Copy peaks only for subjects that have a bundlemask (tractmask)
        if os.path.isfile(output_tractmasks):
            # Copy the peaks file to HCP/subject_id/
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(peaks_path, os.path.join(output_dir, "peaks.nii.gz"))
        else:
            continue
print("All peaks files copied.")
