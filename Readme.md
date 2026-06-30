# Tract Segmentation Benchmark

This repository contains the code used to benchmark tract segmentation models on
brainlife-derived white matter tract datasets. It started from the original
TractSeg codebase, but the current purpose is different: this project focuses on
turning brainlife.io outputs into aligned training targets, training multiple
TractSegmentation model families, and evaluating cross-dataset generalization.

The benchmark currently uses 61 brainlife tract definitions and three main data
domains:

- HCP
- CamCan
- PING

The main workflow is:

1. Download subject-level DWI, peaks, and tract masks from brainlife.io.
2. Convert per-tract brainlife outputs into model-ready 4D NIfTI tensors.
3. Align/crop all domains to the HCP-like 1.25 mm grid used by the training code.
4. Train segmentation models with `ExpRunner`.
5. Run inference on held-out or cross-domain test subjects.
6. Evaluate Dice/F1 scores from predicted masks and reference bundle masks.

## Repository Layout

- `ExpRunner`: legacy combined training, validation, inference, segmentation, and probability-map entrypoint.
- `bin/TrainModel` and `bin/RunInference`: replication-oriented wrappers that separate training and inference defaults.
- `tractseg/`: core package with data loaders, model definitions, experiment configs, metrics, and utilities.
- `tractseg/models/segment_anything/`: vendored SAM code used by MASAM.
- `tractseg/models/mednext/`: vendored MedNeXt v1 architecture subset used by the benchmark models.
- `configs/`: benchmark experiment configs for HCP, CamCan, PING, and combined-source training, kept at the repository root so they are easy to find. `ExpRunner --config=<name>` imports `configs/<name>.py`.
- `preprocessing/`: dataset-specific scripts for converting brainlife tract masks and peaks into training-ready files.
- `scripts/polaris/`: legacy Polaris PBS examples for special cluster runs, not the default replication path.
- `scripts/cortex/`: inference/evaluation orchestration and summary CSVs used in the current benchmark workflow.
- `analyze_outputs/`: small utilities for TractSeg CLI baseline prediction and Dice calculation.
- `resources/`: original TractSeg resources kept for compatibility, including the MNI FA template.

The broader `/mnt/scratch/junb/TractSegVis` workspace also contains large data
folders, visualization scripts, and newer preprocessing helpers. Those data and
output folders should not be committed into this code repository.

## Runtime Environment

The recommended environment is `tsbench`, a faithful clone of the maintained
local `masam_blackwell` environment (Python 3.10, CUDA 12.8 GPU build, pinned
conda + pip packages). For a new checkout, create it and
install this repository in editable mode:

```bash
# 1. Create the environment (conda toolchain + pinned pip packages, incl. the
#    PyTorch cu128 wheels).
conda env create -f envs/tsbench.yml

# 2. Install staple separately: it ships a stale `SimpleITK==1.2.0` pin that the
#    pip resolver cannot satisfy alongside the rest of the freeze, so --no-deps
#    matches how masam_blackwell installed it.
conda run -n tsbench pip install --no-deps staple==0.3.2

# 3. Install this repository in editable mode and run the import smoke test.
conda run -n tsbench python -m pip install --no-build-isolation --no-deps -e .
conda run -n tsbench python tests/test_model_imports.py
```

`envs/tsbench.yml` was generated from `masam_blackwell` with
`conda env export --no-builds` and then patched so it actually replicates from a
clean machine: the pip section carries `--extra-index-url`
(`download.pytorch.org/whl/cu128`) for the `torch`/`torchvision`/`torchaudio`
`+cu128` wheels and `--find-links` (`data.pyg.org`) for the prebuilt
`torch-spline-conv` wheel. The conda toolchain (CUDA 12.8 stack) and pip
packages are otherwise pinned to known-good versions. These pins target a CUDA
12.8 host (verified on an NVIDIA RTX 6000 Ada, compute capability 8.9; cu128 also
covers Hopper/Blackwell-class GPUs with a compatible driver). On other GPUs or
CUDA versions, adjust the `cuda-*`, `libcu*`, and `torch*` pins (and the two
index URLs) to match your driver before creating the env.

`masam_blackwell` remains the maintained source environment and can still be used
directly for one-off commands:

```bash
conda run -n masam_blackwell python -m pip install --no-build-isolation --no-deps -e .
conda run -n masam_blackwell python tests/test_model_imports.py
```

The older hand-written `envs/environment.yml` (`tractseg-benchmark`, CUDA 11.8)
is kept as a lighter-weight fallback for hosts on an older CUDA toolchain.

For workflows that need FSL or MRtrix, prefer Docker unless a cluster module is
explicitly required:

```text
FSL:    brainlife/fsl:6.0.4-patched2
MRtrix: brainlife/mrtrix:latest
```

The training scripts assume CUDA is available. Unless a script overrides it,
GPU `1` is the default local preference for interactive work in this workspace.

## Data Contract

Model configs expect each preprocessed dataset to have one folder per subject:

```text
<DATA_PATH>/<DATASET_FOLDER>/<subject_id>/
  aligned_peaks.nii.gz
  corrected_bundle_masks.nii.gz
```

For HCP configs, the reference labels may be named
`aligned_corrected_bundle_masks.nii.gz` depending on the specific config.

The input and label tensors are:

- `aligned_peaks.nii.gz`: CSD peak image, usually shape `(145, 174, 145, 9)` before training-time crop/pad.
- `corrected_bundle_masks.nii.gz`: 61-channel binary brainlife bundle mask image.

Important reference roots in the current workspace:

```text
/mnt/storage/junb/HCP_preproc_brainlife_fixed
/mnt/storage/junb/CamCan_preproc_brainlife_fixed
/mnt/storage/junb/Ping_preproc_brainlife_fixed
```

For `*_to_CamCan` evaluation settings, use
`/mnt/storage/junb/CamCan_preproc_brainlife_fixed`.

## Preprocessing Brainlife Outputs

Brainlife downloads usually contain separate derivative folders for DWI/peaks
and tract masks. The benchmark preprocessing makes the model input consistent
across datasets.

### 1. Convert tract masks to a 4D bundle mask

For each subject, collect the 61 brainlife tract mask files in deterministic
bundle order and concatenate them into one 4D label image:

```bash
conda run -n masam_blackwell python preprocessing/HCP_brainlife_prep/2.convert_tractmasks_to_bundle_mask.py
```

Equivalent dataset-specific scripts exist for CamCan, PING, and Zenodo-derived
data. New refactoring should consolidate these into one argument-driven script.

### 2. Copy or generate peak inputs

If brainlife already produced MRtrix CSD peaks, copy them into the subject
folder:

```bash
conda run -n masam_blackwell python preprocessing/HCP_brainlife_prep/3.copy_peaks_to_tractseg_input.py
```

If starting from DWI, use the workspace alignment helper under
`../preprocessing/Ping_to_HCP_align/1.align_dwi_to_hcp_mni.py`. It can create a
brain mask, compute FA, register FA to the HCP-like MNI template, resample DWI,
rotate b-vectors, and optionally generate aligned peaks with MRtrix.

### 3. Align orientation and masks

The older dataset scripts flip peak images along the x-axis:

```bash
conda run -n masam_blackwell python preprocessing/HCP_brainlife_prep/4.flip_x_axis_of_peaks.py
```

For PING DWI-based processing, use the newer transform-based workflow in the
workspace:

```bash
conda run -n masam_blackwell python ../preprocessing/Ping_to_HCP_align/1.align_dwi_to_hcp_mni.py \
  --dataset-root /mnt/storage/junb/proj-60708cf9c7f80a684995e0b1 \
  --output-root /mnt/storage/junb/TractSeg_datasets/Ping_for_training_brainlife_from_dwi_hcp_align \
  --fsl-mode docker \
  --mrtrix-mode docker

conda run -n masam_blackwell python ../preprocessing/Ping_to_HCP_align/2.apply_fa_transform_to_tractmasks.py \
  --dataset-root /mnt/storage/junb/proj-60708cf9c7f80a684995e0b1 \
  --alignment-root /mnt/storage/junb/TractSeg_datasets/Ping_for_training_brainlife_from_dwi_hcp_align \
  --fsl-mode docker
```

### 4. Crop to training dimensions

After peaks and bundle masks are aligned, crop them into the fixed input layout:

```bash
conda run -n masam_blackwell python preprocessing/HCP_brainlife_prep/5.preprocessing_brainlife.py
```

The newer PING fixed-crop script lives in the workspace at
`../preprocessing/Ping_brainlife_prep_corrected/5.preprocessing_brainlife_fixed.py`
and exposes CLI arguments for source/output roots, crop bounds, workers, and
overwrite behavior.

After preprocessing, verify that every subject has both input and label files
and that their spatial dimensions match.

## Training Models

Training is controlled by `TrainModel` plus a custom config under `configs/` at
the repository root. `TrainModel` delegates to `ExpRunner` but always sets
training defaults explicitly. The `--config <name>` argument loads
`configs/<name>.py`.

Example: train TractSeg 2D on HCP fold 0.

```bash
conda run -n masam_blackwell TrainModel \
  --fold 0 \
  --data_path /mnt/storage/junb \
  --config wholeHCP_experiment_TractSeg \
  --en wholeHCP_experiment_TractSeg_x1
```

Example: train a combined-source model.

```bash
conda run -n masam_blackwell TrainModel \
  --fold 0 \
  --data_path /mnt/storage/junb \
  --config HCP_CamCan_Ping_experiment_TractSeg \
  --en HCP_CamCan_Ping_experiment_TractSeg_x1
```

Relevant config conventions:

- `DATASET`: dataset name or list of dataset names.
- `DATASET_FOLDER`: preprocessed dataset folder(s).
- `FEATURES_FILENAME`: usually `aligned_peaks`.
- `LABELS_FILENAME`: usually `corrected_bundle_masks` or `aligned_corrected_bundle_masks`.
- `CLASSES`: `Brainlife` for the 61-channel benchmark labels.
- `DIM`: `2D` or `3D`.
- `MODEL`: one of the maintained benchmark model families: `UNet_Pytorch_DeepSup`, `UNet3D_Pytorch_DeepSup_sm`, `SwinUNETR`, `MASAM`, or `mednext_v1`.

### SAM pretrained weights for MASAM (training from scratch)

The `MASAM` model fine-tunes a Segment Anything (SAM) backbone. Those original
SAM weights are **not vendored in this repository** and are not downloaded
automatically, so a first-time / from-scratch MASAM run must fetch them
manually. Download the checkpoint matching the config's `vit_name` from the
official SAM release (the same links given in the "Training" section of the
upstream MA-SAM repo, https://github.com/cchen-cc/MA-SAM):

| `vit_name` | checkpoint file | download URL |
|------------|-----------------|--------------|
| `vit_b`    | `sam_vit_b_01ec64.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |
| `vit_l`    | `sam_vit_l_0b3195.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| `vit_h`    | `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

For example:

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Then, in the MASAM config, point `WEIGHTS_PATH` at the downloaded `.pth` file,
set `LOAD_WEIGHTS = True`, and make sure `vit_name` matches the checkpoint
(e.g. `vit_b` with `sam_vit_b_01ec64.pth`). Only when all three hold does the
backbone load; otherwise it falls back to random initialization. When resuming a
run or running inference from a saved benchmark checkpoint, set `WEIGHTS_PATH` to
that benchmark checkpoint instead of the raw SAM weights.

Polaris PBS files are retained only as legacy cluster examples. Prefer the
conda-based commands above for replication and adapt fold/config/model matrices
outside the repository-specific scheduler scripts.

## Inference and Evaluation

Use `RunInference` with an explicit `--weights_path` to generate binary
segmentations. The wrapper sets `--train False`, `--test False`, `--lw`, and
`--seg` unless `--probs` is provided.

Example: evaluate an HCP-trained model on CamCan.

```bash
conda run -n masam_blackwell RunInference \
  --fold 0 \
  --data_path /mnt/storage/junb \
  --config CamCan_experiment_TractSeg \
  --en HCP_to_CamCan_experiment_TractSeg_inference_x1 \
  --weights_path /mnt/storage/junb/hcp_exp/wholeHCP_experiment_TractSeg_x14/best_weights_ep73.npz
```

The output is written under:

```text
<DEFAULT_EXP_PATH>/<experiment_name>/segmentations/<subject_id>_segmentation.nii.gz
```

Use `--probs` instead of `--seg` when probability maps are needed.

The current inference matrices and cross-domain runs are represented by scripts
under `scripts/cortex/inference/` and summary CSVs under
`scripts/cortex/summary_csv/`.

For the original TractSeg CLI baseline, use:

```bash
bash analyze_outputs/1.predict_bundle_masks_with_TractSeg_CLI.sh
conda run -n masam_blackwell python analyze_outputs/2.concat_predicted_segmentations_and_calc_dice.py
```

## Maintenance Plan

The repo still contains upstream TractSeg compatibility code, benchmark code,
cluster logs, notebook checkpoints, and generated Python caches. Cleanup should
be staged so benchmark reproducibility is not broken.

### Phase 1: Safe repository hygiene

- Remove generated caches from version control: `__pycache__/`, `*.pyc`, and `.ipynb_checkpoints/`.
- Remove tracked cluster output logs such as `R-*.out`.
- Extend `.gitignore` for generated benchmark outputs, local Neptune metadata,
  model checkpoints, and large local datasets.
- Keep source scripts and configs unchanged in this phase.

### Phase 2: Preprocessing consolidation

- Replace dataset-specific copies of `2.convert_*`, `3.copy_*`, `4.flip_*`, and
  `5.preprocessing_*` with a single CLI-driven preprocessing package.
- Move newer workspace-only helpers from `../preprocessing/Ping_to_HCP_align/`
  into this repo after path defaults are made portable.
- Add shape/order validation for the 61 brainlife bundles before writing labels.
- Save a per-subject preprocessing manifest with input paths, output paths,
  affine, shape, and skipped/failed reason.

### Phase 3: Experiment config cleanup

- Group configs by model family and dataset source/target instead of keeping
  many near-duplicate files in one flat directory.
- Replace hard-coded absolute paths with CLI arguments or environment variables.
- Normalize naming: `PING`, `Ping`, and `PIng` should use one convention.
- Move old or unused Zenodo/visual-tract configs into an archive folder if they
  are not part of the paper benchmark.

### Phase 4: Model-code boundaries

- Keep the maintained model families under `tractseg/models/`: TractSeg U-Net,
  SwinUNETR, MedNeXt v1, and MASAM/SAM.
- Add new model families only through `tractseg/models/` and a named experiment
  config so model ownership stays inside one codebase.
- Preserve original TractSeg CLI compatibility until the benchmark no longer
  needs the CLI baseline.

### Phase 5: Evaluation standardization

- Replace ad hoc inference PBS files with a small matrix runner that accepts
  source dataset, target dataset, model family, fold, and weights table.
- Treat `scripts/cortex/summary_csv/source_model_weight_matrix.csv` as the
  canonical weight registry or replace it with a versioned YAML/CSV manifest.
- Add one evaluation command that writes per-subject, per-bundle, and aggregate
  Dice/F1 tables.

### Phase 6: Tests and reproducibility

- Add smoke tests for preprocessing on one tiny subject fixture.
- Add config import tests for all benchmark configs.
- Add an inference dry-run test that verifies output paths without requiring a
  full GPU job.
- Document exact paper experiment IDs, folds, dataset roots, and output table
  locations.

## Original TractSeg Provenance

This codebase is derived from TractSeg:

- Wasserthal et al., "TractSeg - Fast and accurate white matter tract segmentation", NeuroImage 2018.
- Wasserthal et al., "Tract orientation mapping for bundle-specific tractography", MICCAI 2018.
- Wasserthal et al., "Combined tract segmentation and orientation mapping for bundle-specific tractography", Medical Image Analysis 2019.

The benchmark-specific code in this repository should be cited according to the
TractSegmentation Benchmark paper once the citation is finalized.
