#!/bin/bash
#SBATCH -A m4673
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --exclusive
#SBATCH --output=R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
module load pytorch/1.13.1

# Set the path to the reference file (e.g., brain mask) required by your Python script.
REFERENCE_FILE="/global/cfs/cdirs/m4673/junbeom/TractSegVis/5967bffa9b45c212bbec8956/mask.nii"

# Navigate to the root folder of HCP data
base="/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP105_Zenodo_NewTrkFormat"


for subject_folder in */ ; do
    
    # Define the 'tractmasks' folder path
    tractmasks_folder="$base/${subject_folder}/tractmasks"
    
    # Only run if the 'tractmasks' folder doesn't exist or has fewer than 72 files
    if [ ! -d "${tractmasks_folder}" ] || [ $(find "${tractmasks_folder}" -type f | wc -l) -lt 72 ]; then
        # Create the 'tractmasks' folder
        mkdir -p "${tractmasks_folder}"
        
        # Define the 'tracts' folder path
        tracts_folder="$base/${subject_folder}/tracts"
        
        # Process all .trk files in the 'tracts' folder
        for trk_file in "${tracts_folder}"/*.trk; do
            # Extract the base name of the .trk file (without path and extension)
            base_name=$(basename "${trk_file}" .trk)
            
            # Set the output file name within the 'tractmasks' folder
            output_file="$base/${tractmasks_folder}/${base_name}.nii.gz"
            
            # Convert the .trk file to .nii.gz using the Python script
            python trk_2_binary.py "${trk_file}" "${output_file}" "${REFERENCE_FILE}"
        done
    fi
done