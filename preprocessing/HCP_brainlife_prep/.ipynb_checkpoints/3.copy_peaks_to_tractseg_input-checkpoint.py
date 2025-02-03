import os
import shutil

# base directory 설정
base_dir = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/bl_derivative"
output_root = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP_for_training_brainlife"

# 각 subject 폴더를 순회
for subject_id in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_id)
    
    # peaks 파일 경로 설정
    peaks_path = os.path.join(subject_path, "neuro", "peaks", "6_28_2023.hcp.lmax8", "peaks.nii.gz")
    
    # peaks 파일이 존재하는지 확인
    if os.path.isfile(peaks_path):
        # 새로운 저장 경로 설정: HCP/subject_id/
        output_dir = os.path.join(output_root, subject_id)
        output_tractmasks = os.path.join(output_dir,'bundle_masks.nii.gz')
        # bundlemask (tractmask)가 있는 subject에 대해서만 peaks 복사
        if os.path.isfile(output_tractmasks):
            # peaks 파일을 HCP/subject_id/로 복사
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(peaks_path, os.path.join(output_dir, "peaks.nii.gz"))
        else:
            continue
print("All peaks files copied.")
