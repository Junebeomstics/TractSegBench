import os
from nilearn.image import concat_imgs

# Set path to the HCP105_Zenodo_NewTrkFormat folder
base_dir = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP_tractmasks"
output_root = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP_for_training_brainlife"
tract_num = 61

# Iterate over each subject folder
for subject_id in sorted(os.listdir(base_dir)):
    subject_path = os.path.join(base_dir, subject_id)
    
    # Check whether a tractmasks folder exists inside each subject folder
    tractmasks_path = os.path.join(subject_path, "masks")
    if not os.path.isdir(tractmasks_path):
        continue  # Skip to the next subject if there's no tractmasks folder
    
    # Get paths of all NIfTI files in the masks folder
    nifti_files = [os.path.join(tractmasks_path, f) for f in sorted(os.listdir(tractmasks_path)) if f.endswith(".nii") or f.endswith(".nii.gz")]
    # Needs to be sorted here.

    # Concatenate the NIfTI files
    if len(nifti_files) == tract_num:  # Only process if files exist
        combined_img = concat_imgs(nifti_files)

        # Set new output path: HCP/subject_id/
        output_dir = os.path.join(output_root, subject_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save the combined image
        combined_img.to_filename(os.path.join(output_dir, "bundle_masks.nii.gz"))
    else:
        print(f'filter {subject_id}, because it only has {len(nifti_files)} files.')
print("All subjects processed and saved.")

