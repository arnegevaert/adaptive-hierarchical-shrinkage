#!/usr/bin/env bash

#SBATCH --job-name=adhs_general_performance
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/adhs_general_performance_%j.out
#SBATCH --error=logs/adhs_general_performance_%j.err

# Make sure joblib works
unset OMP_PROC_BIND
export OMP_NUM_THREADS=1

#uv run general_performance/run.py --n-replications 100 --n-jobs 32 --out-dir output/general_performance
uv run general_performance/run.py --n-replications 1 --n-jobs 1 --out-dir output/general_performance
