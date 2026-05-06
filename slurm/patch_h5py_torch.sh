#!/bin/bash
# =============================================================================
#  ONE-TIME PATCH: Install h5py + re-pin torch/torchvision to CUDA 12.1 builds
# =============================================================================
# isaacsim-rl upgraded torch to 2.11.0 (CPU). This re-pins to 2.3.0+cu121 and
# installs the missing h5py package required by isaaclab.utils.datasets.
#
# Submit with:
#   sbatch slurm/patch_h5py_torch.sh
# =============================================================================

#SBATCH --job-name=patch_h5py_torch
#SBATCH --account=rob_gy_73237-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/patch_h5py_torch_%j.log
#SBATCH --error=logs/patch_h5py_torch_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ac9374@nyu.edu

set -e

PROJECT_DIR="/scratch/${USER}/triple_pendulum"
OVERLAY="${PROJECT_DIR}/isaac_env/isaac_sim.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
SINGULARITY_TMPDIR="/scratch/${USER}/tmp"
export SINGULARITY_TMPDIR
mkdir -p "${SINGULARITY_TMPDIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "=== Patching h5py + re-pinning torch/torchvision ==="

singularity exec \
    --overlay "${OVERLAY}" \
    "${SIF}" \
    /bin/bash -c "
        source /ext3/env.sh
        PIP=/ext3/miniforge3/bin/pip

        echo '--- Installing h5py ---'
        \$PIP install h5py

        echo '--- Re-pinning torch 2.3.0+cu121 and torchvision 0.18.0+cu121 ---'
        \$PIP install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
            --index-url https://download.pytorch.org/whl/cu121 \
            --force-reinstall

        echo ''
        echo '--- Verifying ---'
        python -c \"import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())\"
        python -c \"import h5py; print('h5py:', h5py.__version__)\"
        echo ''
        echo '=== Patch complete ==='
    "
