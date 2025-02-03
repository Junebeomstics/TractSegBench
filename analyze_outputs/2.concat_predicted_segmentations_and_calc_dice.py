import os
import nibabel as nib
from nilearn.image import concat_imgs
from pathlib import Path
import numpy as np
import csv
from nilearn.image import resample_img


def combine_nifti_files(bundle_folder, output_file):
    """
    Combine all nifti files within the 'bundle_segmentations' folder in ascending alphabet order.
    Save the combined file as 'predicted_bundle_masks.nii.gz' in 'tractseg_output'.
    """
    # Get list of nifti files in ascending alphabet order
    nifti_files = sorted([os.path.join(bundle_folder, f) for f in os.listdir(bundle_folder) if f.endswith(".nii.gz")])

    # Use nilearn's concat_img to combine nifti files
    combined_img = concat_imgs(nifti_files)

    # Save the combined nifti file
    nib.save(combined_img, output_file)
    print(f"Saved combined nifti file to {output_file}")

def calculate_dice_scores_per_channel(reference_file, predicted_file):
    """
    Calculate the Dice scores for each channel in the fourth dimension of the reference and predicted files.
    """
    ref_nii = nib.load(reference_file)
    pred_nii = nib.load(predicted_file)
    ref_data = ref_nii.get_fdata()
    #pred_data = np.flip(pred_nii.get_fdata(),axis=0) # apply x-flip
    pred_data = pred_nii.get_fdata()

    # print('ref_data.shape',ref_data.shape) # (145, 174, 145, 72)
    # print('pred_data.shape',pred_data.shape) # (145, 174, 145, 72)

    if ref_data.shape != pred_data.shape:
        raise ValueError("Reference and predicted files must have the same shape")
    
    
    num_channels = ref_data.shape[-1]
    dice_scores = []

    for i in range(num_channels):
        ref_channel = ref_data[..., i] > 0  # Binary mask
        pred_channel = pred_data[..., i] > 0  # Binary mask

        intersection = np.sum(ref_channel & pred_channel)
        overlap = np.any(ref_channel & pred_channel)

        union = np.sum(ref_channel) + np.sum(pred_channel)

        if union == 0:  # Handle case where both are empty
            dice_score = 1.0
        else:
            dice_score = (2 * intersection) / union

        # print(i,intersection,union,dice_score)
        dice_scores.append(dice_score)

    return dice_scores

def save_dice_scores(subject_scores, output_csv):
    """
    Save the Dice scores to a CSV file.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Subject", "Channel", "Dice Score"])

        for subject, scores in subject_scores.items():
            for channel, score in enumerate(scores):
                writer.writerow([subject, channel, score])

def main():
    main_folder = "HCP_for_training_Zenedo"
    subject_scores = {}
    overall_scores = []

    for subject_folder in sorted(Path(main_folder).iterdir(), key=lambda x: x.name):
        if subject_folder.is_dir():
            subject_name = subject_folder.name
            bundle_folder = subject_folder / "tractseg_output" / "bundle_segmentations"
            # output_file = subject_folder / "tractseg_output" / "predicted_bundle_masks.nii.gz"
            # affinesof peaks were corrected for this version.
            output_file = subject_folder / "tractseg_output" / "predicted_bundle_masks_aligned.nii.gz"

            # Combine nifti files
            if bundle_folder.exists():
                combine_nifti_files(bundle_folder, output_file)

                # Calculate Dice scores with ground truth
                reference_file = subject_folder / "sorted_bundle_masks.nii.gz"
                if reference_file.exists():
                    dice_scores = calculate_dice_scores_per_channel(reference_file, output_file)
                    subject_scores[subject_name] = dice_scores

                    avg_score = np.mean(dice_scores)
                    overall_scores.append(avg_score)

                    print(f"Average Dice score for subject {subject_name}: {avg_score:.4f}")
                else:
                    print(f"Reference file not found for subject {subject_name}")
            else:
                print(f"Bundle segmentations folder not found for subject {subject_name}")

    # Save subject-level Dice scores to a CSV file
    output_csv = "dice_scores_per_subject.csv"
    save_dice_scores(subject_scores, output_csv)
    print(f"Saved Dice scores to {output_csv}")

    # Calculate and print overall average Dice score
    overall_avg_score = np.mean(overall_scores) if overall_scores else 0
    print(f"Overall average Dice score: {overall_avg_score:.4f}")

if __name__ == "__main__":
    main()