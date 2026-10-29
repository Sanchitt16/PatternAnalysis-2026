#!/bin/bash
#SBATCH --job-name=hipmri
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=hipmri_%x_%j.out
#SBATCH --error=hipmri_%x_%j.err
#
# Usage (from this folder on Rangpur):
#   2D:  sbatch --export=ALL,MODEL=unet train_rangpur.slurm
#        sbatch --export=ALL,MODEL=improved train_rangpur.slurm
#   3D:  sbatch --job-name=hipmri3d --export=ALL,MODEL=improved3d,EPOCHS=150,BATCH=2 train_rangpur.slurm
#   2D ablation (one shared, pre-built cache):
#        sbatch --job-name=abl --export=ALL,MODEL=improved_nodrop,SEED=42,CACHE=$HOME/hipmri_cache_2d train_rangpur.slurm
# Optional: SEED=<n>, OUT=<run folder>, CACHE=<cache folder>. Data is read in place from the shared course folder;
# the .npz caches go to your home directory because /home/groups is read-only.
# Everything the run prints is written to hipmri_<job-name>_<job-id>.out in this folder.

MODEL=${MODEL:-unet}
SEED=${SEED:-42}
STUDY=/home/groups/comp3710/HipMRI_Study_open
if [[ $MODEL == *3d ]]; then
    DATA=$STUDY; EPOCHS=${EPOCHS:-150}; BATCH=${BATCH:-2}
else
    DATA=$STUDY/keras_slices_data; EPOCHS=${EPOCHS:-40}; BATCH=${BATCH:-16}
fi
OUT=${OUT:-runs/${MODEL}_rangpur_s$SEED}

echo "Job $SLURM_JOB_ID on $(hostname) at $(date): model=$MODEL epochs=$EPOCHS batch=$BATCH seed=$SEED out=$OUT"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1   # print each epoch line to the .out file immediately

CACHE=${CACHE:-$HOME/hipmri_cache_$MODEL}   # pass CACHE=... to share one pre-built cache between jobs
mkdir -p $CACHE
python train.py --model $MODEL --data-root $DATA --cache-dir $CACHE \
    --epochs $EPOCHS --batch-size $BATCH --seed $SEED --workers 4 --out $OUT

echo "Finished at $(date)"
