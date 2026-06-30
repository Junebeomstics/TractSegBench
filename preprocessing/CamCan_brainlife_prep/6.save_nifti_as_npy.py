import os
import nibabel as nib
import numpy as np

# Set top-level directory path
# root_dir = "/grand/NeuroX/junbeom/TractSegVis/HCP_preproc_brainlife_fixed"
# root_dir = "/grand/NeuroX/junbeom/TractSegVis/Ping_preproc_brainlife_fixed"
root_dir = "/grand/NeuroX/junbeom/TractSegVis/CamCan_preproc_brainlife_fixed"

# Iterate over all subject folders
for subject in sorted(os.listdir(root_dir)):
    subject_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for file in os.listdir(subject_path):
        if file.endswith(".nii.gz"):
            nii_path = os.path.join(subject_path, file)
            npy_path = os.path.join(subject_path, file.replace(".nii.gz", ".npy"))

            # Load NIfTI file and extract data
            img = nib.load(nii_path)
            data = img.get_fdata(dtype=np.float32)  # Using float32 saves memory

            # Save as npy file
            if os.path.exists(npy_path):
                print(f"Skipping {npy_path}, already exists.")
                continue
            np.save(npy_path, data)
            print(f"Saved {npy_path}")
