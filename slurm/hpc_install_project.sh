#!/bin/bash
# =============================================================================
#  NYU HPC Cloud Bursting – Install the triple_pendulum project package
# =============================================================================
# Run this once after hpc_setup.sh completes.
# Must be run from the project directory: /scratch/$USER/triple_pendulum
# =============================================================================

NETID="${USER}"
PROJECT_DIR="/scratch/${NETID}/triple_pendulum"
OVERLAY="${PROJECT_DIR}/isaac_env/isaac_sim.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

if [ ! -f "${OVERLAY}" ]; then
    echo "ERROR: Run hpc_setup.sh first."
    exit 1
fi

echo "Installing triple_pendulum project package into overlay..."

# Singularity needs a writable tmp dir; the default /tmp path may not exist
export SINGULARITY_TMPDIR=/scratch/${USER}/tmp
export SINGULARITY_CACHEDIR=/scratch/${USER}/singularity_cache
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

singularity exec \
    --overlay "${OVERLAY}:rw" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${SIF}" \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_DIR}/source:\${PYTHONPATH}
        cd ${PROJECT_DIR}
        pip install -e . --quiet
        echo 'Project package installed.'
        python -c 'import triple_pendulum; print(\"triple_pendulum package OK\")'
    "

echo ""
echo "Done! You can now submit jobs with:"
echo "  sbatch --array=0-2 slurm/train_job.sh"
