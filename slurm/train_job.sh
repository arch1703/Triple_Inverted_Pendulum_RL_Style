#!/bin/bash
# SLURM job for triple pendulum PPO training on NYU HPC
# single seed:  sbatch slurm/train_job.sh
# 3-seed array: sbatch --array=0-2 slurm/train_job.sh
# live log:     tail -f logs/train_<JOBID>_<ARRAYID>.log

#SBATCH --job-name=triple_ppo
#SBATCH --account=rob_gy_73237-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j_%a.log
#SBATCH --error=logs/train_%j_%a.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=ac9374@nyu.edu

NETID="${USER}"
PROJECT_DIR="/scratch/${NETID}/triple_pendulum"
OVERLAY="${PROJECT_DIR}/isaac_env/isaac_sim.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
ENV_SCRIPT="/ext3/env.sh"
SEED=${SLURM_ARRAY_TASK_ID:-42}

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-N/A} seed=${SEED} node=$(hostname)"
if [ ! -f "${OVERLAY}" ]; then
    echo "ERROR: overlay not found at ${OVERLAY} - run hpc_setup.sh first"
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Singularity image not found at ${SIF}"
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/results/runs"
mkdir -p "${PROJECT_DIR}/results/checkpoints"

module purge

OMNI_DATA="/scratch/${NETID}/omni_data_seed_${SEED}"
mkdir -p "${OMNI_DATA}"

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${OMNI_DATA}:${OMNI_DATA}" \
    "${SIF}" \
    /bin/bash -c "
        source ${ENV_SCRIPT}
        export PYTHONPATH=${PROJECT_DIR}/source:\${PYTHONPATH}
        export ISAACSIM_ACCEPT_EULA=YES
        export OMNI_KIT_ACCEPT_EULA=YES
        export OMNI_DATA_PATH=${OMNI_DATA}
        export OMNI_USER_DATA_PATH=${OMNI_DATA}
        export OMNI_CACHE_PATH=${OMNI_DATA}/cache
        cd ${PROJECT_DIR}
        python scripts/train.py \
            --num_envs 512 \
            --headless \
            --seed ${SEED} \
            --timesteps 3000000 \
            --experiment seed_${SEED}
    "

TRAIN_EXIT=$?
echo "training finished with exit code ${TRAIN_EXIT}"

# auto-evaluate the final checkpoint if training succeeded
FINAL_CKPT=$(ls -t "${PROJECT_DIR}/results/runs/"*"seed_${SEED}"*/checkpoints/*.pt 2>/dev/null | head -1)

if [ -n "${FINAL_CKPT}" ] && [ "${TRAIN_EXIT}" -eq 0 ]; then
    echo "running eval on: ${FINAL_CKPT}"
    singularity exec --nv \
        --overlay "${OVERLAY}:ro" \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        "${SIF}" \
        /bin/bash -c "
            source ${ENV_SCRIPT}
            export PYTHONPATH=${PROJECT_DIR}/source:\${PYTHONPATH}
            cd ${PROJECT_DIR}
            python scripts/eval.py \
                --checkpoint ${FINAL_CKPT} \
                --headless \
                --num_episodes 20
        "
else
    echo "skipping eval (no checkpoint or non-zero exit)"
fi
