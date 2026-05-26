#!/bin/bash
# Install the triple_pendulum package into the Singularity overlay
# run once after hpc_setup.sh, from the project directory

NETID="${USER}"
PROJECT_DIR="/scratch/${NETID}/triple_pendulum"
OVERLAY="${PROJECT_DIR}/isaac_env/isaac_sim.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

if [ ! -f "${OVERLAY}" ]; then
    echo "ERROR: run hpc_setup.sh first"
    exit 1
fi

echo "installing triple_pendulum into overlay..."

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
        # Full import requires Isaac Sim runtime (pxr/OpenUSD) — verify via
        # package metadata instead, which doesn't trigger the import chain.
        python -c "import importlib.metadata; v=importlib.metadata.version('triple_pendulum'); print('triple_pendulum package OK, version:', v)"
    "

echo "done - submit jobs with: sbatch --array=0-2 slurm/train_job.sh"
