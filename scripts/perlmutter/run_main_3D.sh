#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --exclusive
#SBATCH --output=R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
module load pytorch/2.1.0

./ExpRunner --config Zenedo_experiment_3D --en run_with_3D --train True --test True --seg
#./ExpRunner --config wholevisualtract_experiment --en run_with_vis_settings --train True --test True
