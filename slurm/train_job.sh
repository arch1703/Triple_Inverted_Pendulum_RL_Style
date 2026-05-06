#!/bin/bash
# =============================================================================
#  NYU HPC Cloud Bursting – SLURM job script for triple pendulum PPO training
# =============================================================================
#
# Submitting a single seed (default seed 42):
#   sbatch slurm/train_job.sh
#
# Submitting all 3 seeds as a job array:
#   sbatch --array=0-2 slurm/train_job.sh
#
# Monitor logs live:
#   tail -f logs/train_<JOBID>_<ARRAYID>.log
#
# GPU partition options (from course allocation):
#   c12m85-a100-1  →  1 A100 40GB  (recommended for full 3M-step training)
#   g2-standard-12 →  1 L4 GPU     (cheaper; good for short tests)
# =============================================================================

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

# ---------------------------------------------------------------------------
# 0.  Paths  (edit NETID once; everything else is relative)
# ---------------------------------------------------------------------------
NETID="${USER}"                                   # auto-filled by SLURM
PROJECT_DIR="/scratch/${NETID}/triple_pendulum"
OVERLAY="${PROJECT_DIR}/isaac_env/isaac_sim.ext3"

# CUDA 12.x Ubuntu 22.04 image – check available with:
#   ls /share/apps/images/
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# Conda env activation script written during setup (see README_HPC_SETUP.sh)
ENV_SCRIPT="/ext3/env.sh"

# Default seed = SLURM array task ID; fallback to 42 for non-array jobs
SEED=${SLURM_ARRAY_TASK_ID:-42}

echo "=========================================="
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Array ID  : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  Seed      : ${SEED}"
echo "  Node      : $(hostname)"
echo "  Partition : ${SLURM_JOB_PARTITION}"
echo "  Overlay   : ${OVERLAY}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1.  Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "${OVERLAY}" ]; then
    echo "ERROR: Singularity overlay not found at ${OVERLAY}"
    echo "Run the setup script first:  bash ${PROJECT_DIR}/slurm/hpc_setup.sh"
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Singularity image not found at ${SIF}"
    echo "Check available images with: ls /share/apps/images/"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2.  Directories
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/results/runs"
mkdir -p "${PROJECT_DIR}/results/checkpoints"

# ---------------------------------------------------------------------------
# 3.  Training  (overlay opened read-only so multiple seeds can run in parallel)
# ---------------------------------------------------------------------------
module purge

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${SIF}" \
    /bin/bash -c "
        source ${ENV_SCRIPT}
        export PYTHONPATH=${PROJECT_DIR}/source:\${PYTHONPATH}
        cd ${PROJECT_DIR}
        python scripts/train.py \
            --num_envs 512 \
            --headless \
            --seed ${SEED} \
            --timesteps 3000000 \
            --experiment seed_${SEED}
    "

TRAIN_EXIT=$?
echo "Training finished with exit code ${TRAIN_EXIT}"

# ---------------------------------------------------------------------------
# 4.  Auto-evaluate the best checkpoint after training completes
# ---------------------------------------------------------------------------
FINAL_CKPT=$(ls -t "${PROJECT_DIR}/results/runs/"*"seed_${SEED}"*/checkpoints/*.pt 2>/dev/null | head -1)

if [ -n "${FINAL_CKPT}" ] && [ "${TRAIN_EXIT}" -eq 0 ]; then
    echo "Running evaluation on: ${FINAL_CKPT}"
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
    echo "Skipping eval (no checkpoint found or training did not exit cleanly)"
fi

# ---------------------------------------------------------------------------
# 1.  Verify image exists
# ---------------------------------------------------------------------------
if [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: Singularity image not found at ${SIF_IMAGE}"
    echo "Build it with:"
    echo "  cd \$SCRATCH"
    echo "  singularity build isaac_lab.sif docker://nvcr.io/nvidia/isaac-lab:latest"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2.  Create log directory
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/results/runs"
mkdir -p "${PROJECT_DIR}/results/checkpoints"

# ---------------------------------------------------------------------------
# 3.  Training
# ---------------------------------------------------------------------------
singularity exec --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${SCRATCH}:${SCRATCH}" \
    --env "PYTHONPATH=${PROJECT_DIR}/source:${PYTHONPATH}" \
    "${SIF_IMAGE}" \
    python "${PROJECT_DIR}/scripts/train.py" \
        --num_envs 512 \
        --headless \
        --seed "${SEED}" \
        --timesteps 3000000 \
        --experiment "seed_${SEED}"

echo "Training finished with exit code $?"

# ---------------------------------------------------------------------------
# 4.  (Optional) Auto-evaluate the final checkpoint
# ---------------------------------------------------------------------------
FINAL_CKPT=$(ls -t "${PROJECT_DIR}/results/runs/"*"seed_${SEED}"*/checkpoints/*.pt 2>/dev/null | head -1)

if [ -n "${FINAL_CKPT}" ]; then
    echo "Running evaluation on: ${FINAL_CKPT}"
    singularity exec --nv \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --env "PYTHONPATH=${PROJECT_DIR}/source:${PYTHONPATH}" \
        "${SIF_IMAGE}" \
        python "${PROJECT_DIR}/scripts/eval.py" \
            --checkpoint "${FINAL_CKPT}" \
            --headless
fi
