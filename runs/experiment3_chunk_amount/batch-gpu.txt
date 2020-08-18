#!/bin/bash
#SBATCH --time=07:30:00
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=6

module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4

python3 main.py --run=runs/experiment3_chunk_amount/run${SLURM_ARRAY_TASK_ID}.yaml

