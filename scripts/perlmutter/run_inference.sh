#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --exclusive
#SBATCH --output=R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
module load pytorch/2.1.0

./ExpRunner --config load_best_weight_experiment --en inference_with_loaded_weight_sorted_official_weight --train False --test True --only_val --lw # --probs
