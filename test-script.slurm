#!/bin/bash

#SBATCH --job-name=TrainScript

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=16gb
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --error=testing-ERR.out
#SBATCH --output=testing.out
#SBATCH --mail-user=stephen.moore@ucalgary.ca
#SBATCH --mail-type=all

source /home/stephen.moore/anaconda3/bin/activate mdsc508_dev

python /home/stephen.moore/GarbageClassification/testing.py 