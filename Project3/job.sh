#!/bin/bash

#SBATCH -A cs525
#SBATCH -p academic         
#SBATCH -N 3               
#SBATCH -c 8              
#SBATCH --gres=gpu:2        
#SBATCH -t 40:00:00        
#SBATCH --mem=32G           
#SBATCH --job-name="multi-node-5-with-4-gpus"

eval "$(conda shell.bash hook)"

conda activate myenv

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "No Conda environment is active."
else
  echo "The active Conda environment is: $CONDA_DEFAULT_ENV"
fi

python main.py --train_dqn