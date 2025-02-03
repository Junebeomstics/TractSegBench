#!/bin/bash
#SBATCH -A m4673
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --exclusive
#SBATCH --output=R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
module load pytorch/1.13.1

# Set the path to the reference file (e.g., brain mask) required by your Python script.
REFERENCE_FILE="/global/cfs/cdirs/m4673/junbeom/TractSegVis/dwi_images/mask.nii"

# Navigate to the root folder of HCP data
base="/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP_wholevisualtract"

# Maximum number of parallel jobs
MAX_JOBS=128

# Function to process a .tck file
process_tck_file() {
    tck_file="$1"
    tractmasks_folder="$2"
    reference_file="$3"

    # Extract the base name of the .tck file (without path and extension)
    base_name=$(basename "$tck_file" .tck)

    # Define the output file path
    output_file="${tractmasks_folder}/${base_name}.nii.gz"

    # Run the Python script to convert the .tck file to a .nii.gz file
    echo "Converting $tck_file to $output_file"
    python trk_2_binary.py "$tck_file" "$output_file" "$reference_file"
}

export -f process_tck_file

# Process each subject folder
for subject_folder in "$base"/sub-89*; do

    # Rename dt-neuro-* folders based on their target names
    for dt_folder in "$subject_folder"/dt-*; do
        if [ -d "$dt_folder" ]; then
            target_name=$(echo "$dt_folder" | sed -E 's/.*neuro-([^.]+)\\.id.*/\\1/')
            new_name="$subject_folder/$target_name"
            if [ "$dt_folder" != "$new_name" ]; then
                echo "Renaming $dt_folder to $new_name"
                mv "$dt_folder" "$new_name"
            fi
        fi
    done

    # Remove incorrectly created folder 'tcks\n'
    incorrect_tcks_folder="$subject_folder/'tcks'$'\n'"
    if [ -d "$incorrect_tcks_folder" ]; then
        echo "Removing incorrectly created folder: $incorrect_tcks_folder"
        rm -rf "$incorrect_tcks_folder"
    fi

    # Extract the dt-neuro-tcks folder path
    dt_tcks_folder="$subject_folder/tcks"
    if [ -z "$dt_tcks_folder" ]; then
        echo "No tcks folder found in $subject_folder. Skipping..."
        continue
    fi

    # Remove incorrectly placed tractmasks folder inside tcks folder
    misplaced_tractmasks_folder="$dt_tcks_folder/tractmasks"
    if [ -d "$misplaced_tractmasks_folder" ]; then
        echo "Removing misplaced tractmasks folder: $misplaced_tractmasks_folder"
        rm -rf "$misplaced_tractmasks_folder"
    fi

    # Define the output directory (tractmasks folder under subject folder)
    tractmasks_folder="$subject_folder/tractmasks"

    # Check if the tractmasks folder exists and has fewer files than the tcks folder
    if [ ! -d "$tractmasks_folder" ] || [ $(find "$tractmasks_folder" -type f | wc -l) -lt $(find "$dt_tcks_folder" -type f | wc -l) ]; then
        echo "Processing subject folder: $subject_folder"

        # Create the tractmasks folder if it doesn't exist
        mkdir -p "$tractmasks_folder"

        # Navigate to the tcks folder
        tcks_folder="$dt_tcks_folder/tcks"

        if [ ! -d "$tcks_folder" ]; then
            echo "No tcks folder found in $dt_tcks_folder. Skipping..."
            continue
        fi

        # Process each .tck file in parallel
        job_count=0
        for tck_file in "$tcks_folder"/track*-*.tck; do
            # Check if the file exists
            if [ ! -f "$tck_file" ]; then
                echo "No .tck files found in $tcks_folder. Skipping..."
                continue
            fi

            process_tck_file "$tck_file" "$tractmasks_folder" "$REFERENCE_FILE" &
            ((job_count++))

            # Wait if maximum parallel jobs are reached
            if ((job_count >= MAX_JOBS)); then
                wait
                job_count=0
            fi
        done

        # Wait for any remaining jobs to finish
        wait
    else
        echo "$tractmasks_folder already exists and contains sufficient files. Skipping..."
    fi

done


