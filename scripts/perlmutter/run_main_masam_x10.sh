#!/bin/bash -l 
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

# bigger model (vit_l) -> using pretrained patch embedding layer with new conv1d layer
./ExpRunner --config Zenedo_experiment_MASAM_x10 --en run_with_default_settings_MASAM_x10 --train True --test True  --lw  # --use_dp #--only_val #--seg
#./ExpRunner --config wholevisualtract_experiment --en run_with_vis_settings --train True --test True
