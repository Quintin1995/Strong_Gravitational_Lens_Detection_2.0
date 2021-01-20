#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=ensembles
#SBATCH --array=1-10

module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4

python3 create_input_dep_ensemble.py --run=runs/ensemble_runs/run${SLURM_ARRAY_TASK_ID}.yaml

