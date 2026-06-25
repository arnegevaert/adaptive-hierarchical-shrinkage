#!/usr/bin/env bash

#SBATCH --job-name=adhs_general_performance
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/adhs_general_performance_classification_rf_%j.out
#SBATCH --error=logs/adhs_general_performance_classification_rf_%j.err

# Make sure joblib works
unset OMP_PROC_BIND
export OMP_NUM_THREADS=1

uv run experiments/general_performance/run.py --n-jobs 32 --experiments classification_rf
