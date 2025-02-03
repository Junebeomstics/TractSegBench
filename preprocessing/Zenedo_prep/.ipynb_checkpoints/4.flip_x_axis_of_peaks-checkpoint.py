from nilearn.image import new_img_like
import numpy as np
from nilearn.image import load_img
import os


def align_affines(img):
    """
    Align an image by flipping the x-axis and updating the affine matrix.
    """
    data = img.get_fdata()
    affine = img.affine

    # Flip the x-axis
    flipped_data = np.flip(data, axis=0)
    new_affine = affine.copy()
    new_affine[0, 0] *= -1  # Flip the x-axis orientation
    new_affine[0, 3] *= -1  # Mirror the x-origin

    return new_img_like(img, flipped_data, new_affine)


main_folder = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/HCP_for_training_Zenedo"
for subj in sorted(os.listdir(main_folder)):
    img = load_img((os.path.join(main_folder,subj,'peaks.nii.gz')))

    # Align img1 to match the orientation of img2
    aligned_img = align_affines(img)

    # Save or visualize the aligned image
    aligned_img.to_filename(os.path.join(main_folder,subj,"aligned_peaks.nii.gz"))