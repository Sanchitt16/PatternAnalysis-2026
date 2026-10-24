#!/bin/bash
#SBATCH --job-name=hipmri2d
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=hipmri2d_%j.out
#SBATCH --error=hipmri2d_%j.err
#
# Usage (from this folder on Rangpur):  sbatch --export=MODEL=unet train_rangpur.slurm
#                                        sbatch --export=MODEL=improved train_rangpur.slurm
# The dataset is read in place from the shared course directory; the .npz cache goes to
# your home directory because /home/groups is read-only.

MODEL=${MODEL:-unet}
echo "Job $SLURM_JOB_ID on $(hostname) at $(date), model=$MODEL"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch
export MPLBACKEND=Agg

mkdir -p $HOME/hipmri_cache_$MODEL
python train.py --model $MODEL \
    --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --cache-dir $HOME/hipmri_cache_$MODEL \
    --workers 4 \
    --out runs/$MODEL

echo "Finished at $(date)"
