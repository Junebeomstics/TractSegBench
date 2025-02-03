import os
from nilearn.image import concat_imgs

# HCP105_Zenodo_NewTrkFormat 폴더 경로 설정
base_dir = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/HCP105_Zenodo_NewTrkFormat"
output_root = "/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/HCP_for_training_Zenedo"

# 각 subject 폴더를 순회
for subject_id in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_id)
    
    # 각 subject 폴더 안에 tractmasks 폴더가 있는지 확인
    tractmasks_path = os.path.join(subject_path, "tractmasks")
    if not os.path.isdir(tractmasks_path):
        continue  # tractmasks 폴더가 없으면 다음 subject로
    
    # tractmasks 폴더 내의 모든 NIfTI 파일 경로 가져오기
    nifti_files = [os.path.join(tractmasks_path, f) for f in sorted(os.listdir(tractmasks_path)) if f.endswith(".nii") or f.endswith(".nii.gz")]
    #여기서 정렬해야.

    # NIfTI 파일들을 결합
    if nifti_files:  # 파일이 있는 경우에만 처리
        combined_img = concat_imgs(nifti_files)

        # 새로운 저장 경로 설정: HCP/subject_id/
        output_dir = os.path.join(output_root, subject_id)
        os.makedirs(output_dir, exist_ok=True)

        # 결합된 이미지 저장
        combined_img.to_filename(os.path.join(output_dir, "sorted_bundle_masks.nii.gz"))

print("All subjects processed and saved.")

