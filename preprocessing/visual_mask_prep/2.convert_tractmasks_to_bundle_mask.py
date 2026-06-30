import os
from nilearn.image import concat_imgs


base_dir = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP_wholevisualtract"
outdir = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/HCP_for_training_wholevisualtract"
target_tracts = ['track1-2.nii.gz','track1-3.nii.gz','track1-4.nii.gz','track1-5.nii.gz','track1-6.nii.gz','track2-3.nii.gz','track2-4.nii.gz','track2-5.nii.gz','track2-6.nii.gz','track3-4.nii.gz','track3-5.nii.gz','track3-6.nii.gz','track4-5.nii.gz','track4-6.nii.gz','track5-6.nii.gz']
# subject name: sub-100206

# Iterate over each subject folder
for subject_id in sorted(os.listdir(base_dir)):
    subject_path = os.path.join(base_dir, subject_id)
    
    # Check whether a tractmasks folder exists inside each subject folder
    tractmasks_path = os.path.join(subject_path, "tractmasks")
    if not os.path.isdir(tractmasks_path):
        continue  # Skip to the next subject if there's no tractmasks folder
    
    # Get paths of all NIfTI files in the tractmasks folder
    nifti_files = [os.path.join(tractmasks_path, f) for f in target_tracts if (f.endswith(".nii") or f.endswith(".nii.gz")) ]

    # Concatenate the NIfTI files
    if nifti_files:  # Only process if files exist
        print(f'subject_id:{len(nifti_files)}')
        combined_img = concat_imgs(nifti_files)

        # Set new output path: HCP/subject_id/
        output_dir = os.path.join(outdir, subject_id[4:])
        #os.path.join(subject_path, 'input_to_model')
        os.makedirs(output_dir, exist_ok=True)

        # Save the combined image
        combined_img.to_filename(os.path.join(output_dir, "bundle_masks.nii.gz"))

print("All subjects processed and saved.")

