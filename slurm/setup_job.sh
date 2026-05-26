#!/bin/bash
#SBATCH --job-name=tp_setup
#SBATCH --partition=n2c48m24
#SBATCH --account=rob_gy_73237-2026sp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/setup_%j.log
#SBATCH --error=/scratch/%u/setup_%j.log
# NOTE: No --gres=gpu line → zero GPU hours consumed.
# Package installation does not need a GPU.
# The installed packages (PyTorch+CUDA, Isaac Sim) still work on GPU at runtime.

set -e

echo "setup job: node=$(hostname) job=${SLURM_JOB_ID} started=$(date)"

cd /scratch/${USER}/triple_pendulum

bash slurm/hpc_setup.sh

echo "setup done: $(date) - log: /scratch/${USER}/setup_${SLURM_JOB_ID}.log"
