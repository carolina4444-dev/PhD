#!/bin/bash
#SBATCH --job-name=tsp_rl_a3c                     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --mem=122536
#SBATCH --cpus-per-task=32                                                                                                                    


source .tsp_env/bin/activate


python3 train_eval.py
