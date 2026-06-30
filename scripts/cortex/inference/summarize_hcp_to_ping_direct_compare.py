#!/usr/bin/env python3
import csv
import glob
import os
import pickle
import statistics
from typing import Dict, List

BASE = '/mnt/storage/junb/hcp_ecp'
OUT_CSV = '/mnt/scratch/junb/TractSegVis/TractSeg/scripts/cortex/summary_csv/HCP_to_PING_direct_path_comparison.csv'

MODEL_SPECS = [
    ('TractSeg2D', 'TractSeg_inference'),
    ('TractSeg3D', 'TractSeg3D_inference'),
    ('SwinUNETR2D', 'swinunetr_inference'),
    ('SwinUNETR3D', 'swinunetr3D_inference'),
    ('MASAM3D', 'MASAM3D_inference'),
    ('MedNeXt3D', 'mednext3D_inference'),
]

TAGS = ['storageorig_direct', 'alignedreal_direct']


def load_subject_scores(exp_name: str) -> List[float]:
    exp_dir = os.path.join(BASE, exp_name)
    score_files = sorted(glob.glob(os.path.join(exp_dir, 'subj_f1_scores_test*.pkl')))
    if not score_files:
        raise FileNotFoundError(f'Missing subj score file for {exp_name}')

    with open(score_files[-1], 'rb') as file_obj:
        data = pickle.load(file_obj)

    if isinstance(data, dict):
        return [float(v) for v in data.values()]
    return [float(v) for v in data]


def summarize_tag(tag: str) -> Dict[str, float]:
    model_to_mean = {}
    for model_name, suffix in MODEL_SPECS:
        pooled = []
        for fold in [1, 2, 3]:
            exp_name = f'HCP_to_PING_{tag}_experiment_{suffix}_x{fold}'
            scores = load_subject_scores(exp_name)
            pooled.extend(scores)
        model_to_mean[model_name] = statistics.fmean(pooled)
    return model_to_mean


def main() -> None:
    summary = {}
    missing = []

    for tag in TAGS:
        try:
            summary[tag] = summarize_tag(tag)
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        print('INCOMPLETE')
        for msg in missing:
            print(msg)
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            'model',
            'storageorig_direct_subject_mean_f1',
            'alignedreal_direct_subject_mean_f1',
            'difference_aligned_minus_storageorig',
        ])

        for model_name, _ in MODEL_SPECS:
            orig = summary['storageorig_direct'][model_name]
            aligned = summary['alignedreal_direct'][model_name]
            writer.writerow([model_name, f'{orig:.6f}', f'{aligned:.6f}', f'{aligned - orig:.6f}'])

    print('DONE')
    print(OUT_CSV)


if __name__ == '__main__':
    main()
