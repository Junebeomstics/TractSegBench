#!/bin/bash

module load pytorch/1.13.1

# Set the main folder path
MAIN_FOLDER="HCP_for_training_Zenedo"

# Loop through all 6-digit subject folders
for SUBJECT_FOLDER in "$MAIN_FOLDER"/*; do
    # Check if it is a directory
    if [ -d "$SUBJECT_FOLDER" ]; then
        # Extract the subject name (folder name)
        SUBJECT_NAME=$(basename "$SUBJECT_FOLDER")

        # Construct the input file path
        #INPUT_FILE="$SUBJECT_FOLDER/peaks.nii.gz"
        INPUT_FILE="$SUBJECT_FOLDER/aligned_peaks.nii.gz"

        # Check if the input file exists
        if [ -f "$INPUT_FILE" ]; then
            echo "Running TractSeg for subject: $SUBJECT_NAME"

            # Run the TractSeg command
            TractSeg -i "$INPUT_FILE"
        else
            echo "Warning: peaks.nii.gz not found for subject $SUBJECT_NAME"
        fi
    fi
done
