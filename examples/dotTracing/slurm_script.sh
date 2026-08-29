#!/bin/bash
#SBATCH --job-name=SNN_simulation
#SBATCH --output=logs/run_%A_%a.log
#SBATCH --time=47:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

source /cluster/home/eli08/miniconda3/etc/profile.d/conda.sh
conda activate Para-HADES

python PR4.py 