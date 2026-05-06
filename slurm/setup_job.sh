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

echo "================================================"
echo "  Triple Pendulum – HPC Environment Setup Job"
echo "  Node    : $(hostname)"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Started : $(date)"
echo "================================================"

cd /scratch/${USER}/triple_pendulum

bash slurm/hpc_setup.sh

echo ""
echo "================================================"
echo "  Setup job finished: $(date)"
echo "  Check log: /scratch/${USER}/setup_${SLURM_JOB_ID}.log"
echo "================================================"
