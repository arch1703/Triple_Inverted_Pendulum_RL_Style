#!/bin/bash
# Install the isaacsim.asset (URDF importer) extension package
# usage: sbatch slurm/patch_isaacsim_asset.sh

#SBATCH --job-name=patch_isaacsim_asset
#SBATCH --account=rob_gy_73237-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/patch_isaacsim_asset_%j.log
#SBATCH --error=logs/patch_isaacsim_asset_%j.err
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

echo "installing isaacsim.asset (URDF importer)"

singularity exec \
    --overlay "${OVERLAY}" \
    "${SIF}" \
    /bin/bash -c "
        source /ext3/env.sh
        PIP=/ext3/miniforge3/bin/pip

        echo '--- Installing isaacsim asset/importer packages ---'
        \$PIP install \
            isaacsim-asset==4.5.0.0 \
            isaacsim-asset-importer==4.5.0.0 \
            isaacsim-asset-importer-urdf==4.5.0.0 \
            --extra-index-url https://pypi.nvidia.com \
            --no-build-isolation

        echo ''
        echo '--- Verifying ---'
        python -c 'from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface; print(\"URDF importer OK\")'

        echo '=== Patch complete ==='
    "
