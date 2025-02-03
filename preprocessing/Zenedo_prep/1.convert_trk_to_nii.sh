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
    
    # 'tractmasks' 폴더 경로 정의
    tractmasks_folder="$base/${subject_folder}/tractmasks"
    
    # 'tractmasks' 폴더가 없거나 파일 개수가 72개 미만인 경우에만 실행
    if [ ! -d "${tractmasks_folder}" ] || [ $(find "${tractmasks_folder}" -type f | wc -l) -lt 72 ]; then
        # 'tractmasks' 폴더 생성
        mkdir -p "${tractmasks_folder}"
        
        # 'tracts' 폴더 경로 정의
        tracts_folder="$base/${subject_folder}/tracts"
        
        # 'tracts' 폴더 내의 모든 .trk 파일을 처리
        for trk_file in "${tracts_folder}"/*.trk; do
            # .trk 파일의 기본 이름 추출 (경로 및 확장자 제외)
            base_name=$(basename "${trk_file}" .trk)
            
            # 'tractmasks' 폴더 내의 출력 파일 이름 설정
            output_file="$base/${tractmasks_folder}/${base_name}.nii.gz"
            
            # Python 스크립트를 사용하여 .trk 파일을 .nii.gz 파일로 변환
            python trk_2_binary.py "${trk_file}" "${output_file}" "${REFERENCE_FILE}"
        done
    fi
done