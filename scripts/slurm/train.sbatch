#!/bin/bash
#SBATCH -t 2-08:00:00
#SBATCH --mem=64G
#SBATCH -J vadclipll
#SBATCH --output=/work/MLShare/rek21muv/mistral/logs/train_%j_out.txt
#SBATCH --error=/work/MLShare/rek21muv/mistral/logs/train_%j_err.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=dgx
#SBATCH --account=RPTU-ML-VAD

module load miniforge3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate vad

export PYTHONPATH=/work/MLShare/rek21muv/mistral

python /work/MLShare/rek21muv/mistral/scripts/train.py

conda deactivate