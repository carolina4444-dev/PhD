#!/bin/bash
#SBATCH --job-name=nas_dqnet                       
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --mem=122536
#SBATCH --cpus-per-task=32                                                                                                                  


source .venv/bin/activate


python3 dqnet_keras_policy_learning_nas.py
