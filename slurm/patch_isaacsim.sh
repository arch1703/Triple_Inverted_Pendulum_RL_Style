#!/bin/bash
# =============================================================================
#  ONE-TIME PATCH: Install missing Isaac Sim extension packages into overlay
# =============================================================================
# The base `isaacsim==4.5.0.0` package is only a stub; the actual submodules
# (simulation_app, extscache, etc.) are separate pip packages.
#
# Submit with:
#   sbatch slurm/patch_isaacsim.sh
# =============================================================================

#SBATCH --job-name=patch_isaacsim
#SBATCH --account=rob_gy_73237-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/patch_isaacsim_%j.log
#SBATCH --error=logs/patch_isaacsim_%j.err
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

echo "=== Patching Isaac Sim extension packages into overlay ==="
echo "Overlay: ${OVERLAY}"

singularity exec \
    --overlay "${OVERLAY}" \
    "${SIF}" \
    /bin/bash -c "
        source /ext3/env.sh
        PIP=/ext3/miniforge3/bin/pip

        echo '--- Installing Isaac Sim extension packages ---'
        \$PIP install \
            isaacsim-rl==4.5.0.0 \
            isaacsim-replicator==4.5.0.0 \
            isaacsim-extscache-physics==4.5.0.0 \
            isaacsim-extscache-kit==4.5.0.0 \
            isaacsim-extscache-kit-sdk==4.5.0.0 \
            --extra-index-url https://pypi.nvidia.com \
            --no-build-isolation

        echo ''
        echo '--- Verifying isaacsim.simulation_app is importable ---'
        python -c 'import importlib.metadata; print(\"isaacsim-rl:\", importlib.metadata.version(\"isaacsim-rl\"))'
        python -c 'from isaacsim.simulation_app import SimulationApp; print(\"SimulationApp OK\")' 2>&1 || true

        echo ''
        echo '=== Patch complete ==='
    "
