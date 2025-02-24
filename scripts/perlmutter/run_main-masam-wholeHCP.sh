#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --exclusive
#SBATCH --output=R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
#SBATCH --chdir=/global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/
module load pytorch/2.1.0

cd /global/cfs/cdirs/m4673/junbeom/TractSegVis/TractSeg/
./ExpRunner --config wholeHCP_experiment_MASAM --en run_with_wholeHCP_MASAM --train True --test True --seg --lw
#./ExpRunner --config wholevisualtract_experiment --en run_with_vis_settings --train True --test True
