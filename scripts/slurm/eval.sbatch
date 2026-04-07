#!/bin/bash
#SBATCH -t 1-08:00:00
#SBATCH --mem=32G
#SBATCH -J eval
#SBATCH --output=/work/MLShare/rek21muv/mistral/logs/eval_%j_out.txt
#SBATCH --error=/work/MLShare/rek21muv/mistral/logs/eval_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=dgx
#SBATCH --account=RPTU-ML-VAD

module load miniforge3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate vad

python /work/MLShare/rek21muv/mistral/scripts/evaluate.py

conda deactivate