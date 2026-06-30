import os
import time
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# NPY Dataset
# -------------------------------
class NPYDataset(Dataset):
    def __init__(self, file_paths, use_mmap=False):
        self.file_paths = file_paths
        self.use_mmap = use_mmap

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        if self.use_mmap:
            arr = np.load(path, mmap_mode='r')
            #arr = arr[:]
        else:
            arr = np.load(path)
        return arr

# -------------------------------
# NIFTI Dataset
# -------------------------------
class NIFTIDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        return data

# -------------------------------
# Measurement function
# -------------------------------
def measure_loader_speed(dataloader, num_batches=10):
    batch_time = 0.0
    for i, batch in enumerate(dataloader):
        start = time.time()
        _ = batch.sum()
        if i + 1 >= num_batches:
            break
        end = time.time()
        batch_time += end - start
    average_time = batch_time / num_batches

    return average_time
# -------------------------------
# Path settings
# -------------------------------
# Example: use 20 files
npy_files = [os.path.join('/grand/NeuroX/junbeom/TractSegVis/HCP_preproc_brainlife_fixed',subj,'aligned_corrected_bundle_masks.npy') for subj in os.listdir('/grand/NeuroX/junbeom/TractSegVis/HCP_preproc_brainlife_fixed')]  
nii_files = [os.path.join('/grand/NeuroX/junbeom/TractSegVis/HCP_preproc_brainlife_fixed',subj,'aligned_corrected_bundle_masks.nii.gz') for subj in os.listdir('/grand/NeuroX/junbeom/TractSegVis/HCP_preproc_brainlife_fixed')]  

# -------------------------------
# DataLoader settings
# -------------------------------
batch_size = 2

dataloaders = {
    "NPY (normal)": DataLoader(NPYDataset(npy_files,use_mmap=False), batch_size=batch_size),
    "NPY (mmap)": DataLoader(NPYDataset(npy_files, use_mmap = True), batch_size=batch_size),
    "NIfTI": DataLoader(NIFTIDataset(nii_files), batch_size=batch_size),
}

# -------------------------------
# Speed comparison
# -------------------------------
print("\n📊 DataLoader loading time comparison (based on 10 batches):")
for name, loader in dataloaders.items():
    elapsed = measure_loader_speed(loader)
    print(f" - {name:12}: {elapsed:.4f}s")
