#!/usr/bin/env python3
"""
Compare tract segmentation performance between TractSeg paper and Brainlife.io paper.
Reads dice coefficients from two score files and creates a comparison plot.
"""

import re
import ast
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_score_file(filepath):
    """Parse the score file and extract the tract dictionary."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the defaultdict section
    match = re.search(r'defaultdict\([^,]+,\s*({[^}]+})\)', content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse defaultdict from {filepath}")

    # Extract the dictionary content
    dict_str = match.group(1)

    # Parse the dictionary
    tract_dict = {}
    for line in dict_str.strip().split('\n'):
        line = line.strip().rstrip(',')
        if line and ':' in line:
            # Extract tract name and value
            tract_match = re.match(r"'([^']+)':\s*\[([0-9.]+)\]", line)
            if tract_match:
                tract_name = tract_match.group(1)
                dice_value = float(tract_match.group(2))
                tract_dict[tract_name] = dice_value

    return tract_dict

def create_tract_mapping():
    """
    Create mapping between TractSeg paper tract names and Brainlife.io tract names.
    Returns a dictionary: {tractseg_name: brainlife_name}
    """
    mapping = {
        # Arcuate fasciculus
        'AF_left': 'leftArc',
        'AF_right': 'rightArc',

        # Corticospinal tract
        'CST_left': 'leftCST',
        'CST_right': 'rightCST',

        # Inferior longitudinal fasciculus
        'ILF_left': 'leftILF',
        'ILF_right': 'rightILF',

        # Inferior fronto-occipital fasciculus
        'IFO_left': 'leftIFOF',
        'IFO_right': 'rightIFOF',

        # Uncinate fasciculus
        'UF_left': 'leftUncinate',
        'UF_right': 'rightUncinate',

        # Superior longitudinal fasciculus III
        'SLF_III_left': 'leftSLF3',
        'SLF_III_right': 'rightSLF3',

        # Superior longitudinal fasciculus I and II (combined in Brainlife)
        'SLF_I_left': 'leftSLF1And2',
        'SLF_II_left': 'leftSLF1And2',
        'SLF_I_right': 'rightSLF1And2',
        'SLF_II_right': 'rightSLF1And2',

        # Cingulum
        'CG_left': 'leftcingulum',
        'CG_right': 'rightcingulum',

        # Corpus callosum parts
        'CC_1': 'anterioFrontalCC',
        'CC_2': 'middleFrontalCC',
        'CC_3': 'middleFrontalCC',
        'CC_6': 'parietalCC',
        'CC_7': 'forcepsMajor',
        'CC': 'forcepsMinor',
    }

    return mapping

def identify_unique_tracts(tractseg_tracts, brainlife_tracts, mapping):
    """Identify tracts that are unique to Brainlife.io paper."""
    mapped_brainlife = set(mapping.values())
    all_brainlife = set(brainlife_tracts.keys())
    unique_tracts = all_brainlife - mapped_brainlife
    return sorted(unique_tracts)

def prepare_comparison_data(tractseg_dict, brainlife_dict, mapping):
    """
    Prepare data for comparison plot.
    Returns: tract_names, tractseg_scores, brainlife_scores
    """
    comparison_data = []

    # For mapped tracts
    for ts_name, bl_name in mapping.items():
        if ts_name in tractseg_dict and bl_name in brainlife_dict:
            # Handle cases where multiple TractSeg tracts map to one Brainlife tract
            comparison_data.append({
                'display_name': bl_name,
                'tractseg_score': tractseg_dict[ts_name],
                'brainlife_score': brainlife_dict[bl_name]
            })

    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_data = []
    for item in comparison_data:
        if item['display_name'] not in seen:
            seen.add(item['display_name'])
            unique_data.append(item)

    # Sort by average dice coefficient
    unique_data.sort(key=lambda x: (x['tractseg_score'] + x['brainlife_score']) / 2)

    tract_names = [d['display_name'] for d in unique_data]
    tractseg_scores = [d['tractseg_score'] for d in unique_data]
    brainlife_scores = [d['brainlife_score'] for d in unique_data]

    return tract_names, tractseg_scores, brainlife_scores

def create_comparison_plot(tract_names, tractseg_scores, brainlife_scores,
                          unique_brainlife, brainlife_dict, output_file='tract_comparison.png'):
    """Create a horizontal bar plot comparing dice coefficients."""

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(10, len(tract_names) * 0.4)))

    # Subplot 1: Comparison of common tracts
    y_pos = np.arange(len(tract_names))
    width = 0.35

    ax1.barh(y_pos - width/2, tractseg_scores, width, label='TractSeg Paper', alpha=0.8, color='steelblue')
    ax1.barh(y_pos + width/2, brainlife_scores, width, label='Brainlife.io Paper', alpha=0.8, color='coral')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tract_names, fontsize=10)
    ax1.set_xlabel('Dice Coefficient', fontsize=12)
    ax1.set_title('Comparison of Common Tracts\n(TractSeg Paper vs Brainlife.io Paper)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim([0, 1])

    # Add difference annotations
    for i, (ts, bl) in enumerate(zip(tractseg_scores, brainlife_scores)):
        diff = ts - bl
        color = 'green' if diff > 0 else 'red'
        ax1.text(0.95, i, f'{diff:+.3f}',
                va='center', ha='right', fontsize=8, color=color, fontweight='bold')

    # Subplot 2: Unique tracts from Brainlife.io
    if unique_brainlife:
        unique_names = sorted(unique_brainlife)
        unique_scores = [brainlife_dict[name] for name in unique_names]

        # Sort by score
        sorted_pairs = sorted(zip(unique_names, unique_scores), key=lambda x: x[1])
        unique_names, unique_scores = zip(*sorted_pairs)

        y_pos2 = np.arange(len(unique_names))
        ax2.barh(y_pos2, unique_scores, alpha=0.8, color='mediumseagreen')
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(unique_names, fontsize=10)
        ax2.set_xlabel('Dice Coefficient', fontsize=12)
        ax2.set_title('Unique Tracts in Brainlife.io Paper', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim([0, 1])

        # Add value labels
        for i, score in enumerate(unique_scores):
            ax2.text(score + 0.01, i, f'{score:.3f}',
                    va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No unique tracts found',
                ha='center', va='center', fontsize=12)
        ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def print_statistics(tract_names, tractseg_scores, brainlife_scores, unique_brainlife, brainlife_dict):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("TRACT COMPARISON STATISTICS")
    print("="*80)

    print(f"\nNumber of common tracts: {len(tract_names)}")
    print(f"Number of unique Brainlife.io tracts: {len(unique_brainlife)}")

    # Overall statistics
    ts_mean = np.mean(tractseg_scores)
    bl_mean = np.mean(brainlife_scores)
    ts_std = np.std(tractseg_scores)
    bl_std = np.std(brainlife_scores)

    print(f"\nTractSeg Paper - Mean Dice: {ts_mean:.4f} ± {ts_std:.4f}")
    print(f"Brainlife.io Paper - Mean Dice: {bl_mean:.4f} ± {bl_std:.4f}")
    print(f"Mean Difference: {ts_mean - bl_mean:+.4f}")

    # Tract-wise differences
    differences = np.array(tractseg_scores) - np.array(brainlife_scores)
    print(f"\nMean tract-wise difference: {np.mean(differences):+.4f} ± {np.std(differences):.4f}")
    print(f"Max improvement (TractSeg): {np.max(differences):+.4f}")
    print(f"Max improvement (Brainlife): {np.min(differences):+.4f}")

    # Best and worst performing tracts
    print("\n" + "-"*80)
    print("Top 5 Best Performing Tracts (Average)")
    print("-"*80)
    avg_scores = [(name, (ts + bl) / 2, ts, bl)
                  for name, ts, bl in zip(tract_names, tractseg_scores, brainlife_scores)]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    for name, avg, ts, bl in avg_scores[:5]:
        print(f"{name:40s} Avg: {avg:.4f}  (TS: {ts:.4f}, BL: {bl:.4f})")

    print("\n" + "-"*80)
    print("Top 5 Worst Performing Tracts (Average)")
    print("-"*80)
    for name, avg, ts, bl in avg_scores[-5:]:
        print(f"{name:40s} Avg: {avg:.4f}  (TS: {ts:.4f}, BL: {bl:.4f})")

    # Unique tracts statistics
    if unique_brainlife:
        print("\n" + "-"*80)
        print("Unique Brainlife.io Tracts Statistics")
        print("-"*80)
        unique_scores = [brainlife_dict[name] for name in unique_brainlife]
        print(f"Mean Dice: {np.mean(unique_scores):.4f} ± {np.std(unique_scores):.4f}")
        print(f"Min: {np.min(unique_scores):.4f}, Max: {np.max(unique_scores):.4f}")

    print("\n" + "="*80)

def main():
    # File paths
    tractseg_file = '/mnt/storage/junb/hcp_ecp/Zenedo_to_Zenedo_experiment_TractSeg3D_inference_x1/score_test-set.txt'
    brainlife_file = '/mnt/storage/junb/hcp_ecp/HCP105_to_HCP105_experiment_TractSeg3D_inference_x1/score_test-set.txt'

    # Parse files
    print("Parsing score files...")
    tractseg_dict = parse_score_file(tractseg_file)
    brainlife_dict = parse_score_file(brainlife_file)

    print(f"TractSeg paper: {len(tractseg_dict)} tracts")
    print(f"Brainlife.io paper: {len(brainlife_dict)} tracts")

    # Create mapping and identify unique tracts
    mapping = create_tract_mapping()
    unique_brainlife = identify_unique_tracts(tractseg_dict, brainlife_dict, mapping)

    print(f"\nUnique tracts in Brainlife.io paper ({len(unique_brainlife)}):")
    for tract in unique_brainlife:
        print(f"  - {tract}: {brainlife_dict[tract]:.4f}")

    # Prepare comparison data
    tract_names, tractseg_scores, brainlife_scores = prepare_comparison_data(
        tractseg_dict, brainlife_dict, mapping
    )

    # Create plot
    print("\nCreating comparison plot...")
    create_comparison_plot(
        tract_names, tractseg_scores, brainlife_scores,
        unique_brainlife, brainlife_dict
    )

    # Print statistics
    print_statistics(tract_names, tractseg_scores, brainlife_scores,
                    unique_brainlife, brainlife_dict)

if __name__ == '__main__':
    main()
