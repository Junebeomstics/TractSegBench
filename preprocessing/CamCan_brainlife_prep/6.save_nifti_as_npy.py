import os
import nibabel as nib
import numpy as np

# 최상위 디렉토리 경로 설정
# root_dir = "/grand/NeuroX/junbeom/TractSegVis/HCP_preproc_brainlife_fixed"
# root_dir = "/grand/NeuroX/junbeom/TractSegVis/Ping_preproc_brainlife_fixed"
root_dir = "/grand/NeuroX/junbeom/TractSegVis/CamCan_preproc_brainlife_fixed"

# 모든 subject 폴더 순회
for subject in sorted(os.listdir(root_dir)):
    subject_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for file in os.listdir(subject_path):
        if file.endswith(".nii.gz"):
            nii_path = os.path.join(subject_path, file)
            npy_path = os.path.join(subject_path, file.replace(".nii.gz", ".npy"))

            # NIfTI 파일 로드 및 데이터 추출
            img = nib.load(nii_path)
            data = img.get_fdata(dtype=np.float32)  # float32로 하면 메모리 절약됨

            # npy 파일로 저장
            if os.path.exists(npy_path):
                print(f"Skipping {npy_path}, already exists.")
                continue
            np.save(npy_path, data)
            print(f"Saved {npy_path}")
