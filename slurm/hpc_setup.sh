#!/bin/bash
# =============================================================================
#  NYU HPC Cloud Bursting – ONE-TIME environment setup script
# =============================================================================
#
# Run this ONCE from inside an interactive job on a compute node (NOT login).
# It creates the Singularity overlay with Miniforge + all Python packages.
#
# Usage:
#   1. Start an interactive session (from OOD terminal or login node):
#        srun --partition=g2-standard-12 --account=rob_gy_73237-2026sp \
#             --cpus-per-task=4 --mem=40GB --time=04:00:00 --pty /bin/bash
#
#   2. cd to your project directory:
#        cd /scratch/${USER}/triple_pendulum
#
#   3. Run this script:
#        bash slurm/hpc_setup.sh
#
#   This takes ~30-60 minutes the first time (Isaac Sim is large).
# =============================================================================

set -e  # exit on any error

NETID="${USER}"
PROJECT_DIR="/scratch/${NETID}/triple_pendulum"
ISAAC_LAB_DIR="/scratch/${NETID}/IsaacLab"
ENV_DIR="${PROJECT_DIR}/isaac_env"
OVERLAY="${ENV_DIR}/isaac_sim.ext3"

# Singularity image – check available with: ls /share/apps/images/
# We need Ubuntu 22.04 + CUDA 12.x
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

echo "=============================================="
echo "  NYU HPC Isaac Lab Environment Setup"
echo "  Project : ${PROJECT_DIR}"
echo "  Overlay : ${OVERLAY}"
echo "  Image   : ${SIF}"
echo "=============================================="

# ---------------------------------------------------------------------------
# 1.  Verify we are NOT on the login node (memory would be too limited)
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID}" ]; then
    echo "WARNING: No SLURM_JOB_ID detected."
    echo "Are you sure you're on a compute node and not a login node?"
    echo "Login nodes have a 2GB memory limit which will crash Isaac Sim install."
    read -r -p "Continue anyway? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

# ---------------------------------------------------------------------------
# 2.  Check the Singularity image exists
# ---------------------------------------------------------------------------
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Singularity image not found: ${SIF}"
    echo "Check available CUDA images:"
    echo "  ls /share/apps/images/ | grep cuda"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3.  Check available overlay sizes and create the overlay
# ---------------------------------------------------------------------------
mkdir -p "${ENV_DIR}"

if [ -f "${OVERLAY}" ]; then
    echo "Overlay already exists at ${OVERLAY}. Skipping creation."
    echo "To rebuild from scratch: rm ${OVERLAY} and re-run this script."
else
    echo ""
    echo "Available overlay templates:"
    ls /share/apps/overlay-fs-ext3/ 2>/dev/null || echo "(none found at /share/apps/overlay-fs-ext3/)"
    echo ""
    echo "Looking for the largest available overlay (Isaac Sim needs ~20GB)..."

    # Try to find a 25GB+ overlay; fall back to largest available
    OVERLAY_SRC=""
    for size in overlay-30GB-1M overlay-25GB-500K overlay-20GB-500K overlay-15GB-500K; do
        if ls /share/apps/overlay-fs-ext3/${size}.ext3.gz 2>/dev/null; then
            OVERLAY_SRC="/share/apps/overlay-fs-ext3/${size}.ext3.gz"
            break
        fi
    done

    if [ -z "${OVERLAY_SRC}" ]; then
        echo "ERROR: Could not find a suitable overlay template."
        echo "Check what's available: ls /share/apps/overlay-fs-ext3/"
        echo "You may need to manually copy the largest one."
        exit 1
    fi

    echo "Using overlay template: ${OVERLAY_SRC}"
    echo "Copying and decompressing (this may take a few minutes)..."
    cp -rp "${OVERLAY_SRC}" "${ENV_DIR}/"
    OVERLAY_GZ="${ENV_DIR}/$(basename ${OVERLAY_SRC})"
    gunzip "${OVERLAY_GZ}"
    # Rename to our canonical name
    mv "${ENV_DIR}/$(basename ${OVERLAY_SRC%.gz})" "${OVERLAY}"
    echo "Overlay created: ${OVERLAY}"
fi

# ---------------------------------------------------------------------------
# 4.  Clone Isaac Lab (needed for source install + patch)
# ---------------------------------------------------------------------------
if [ ! -d "${ISAAC_LAB_DIR}" ]; then
    echo ""
    echo "Cloning Isaac Lab v0.54.3..."
    git clone --branch v0.54.3 --depth 1 \
        https://github.com/isaac-sim/IsaacLab.git "${ISAAC_LAB_DIR}"
else
    echo "Isaac Lab already at ${ISAAC_LAB_DIR}."
fi

# ---------------------------------------------------------------------------
# 5.  Apply the Isaac Sim 4.x compatibility patch to Isaac Lab
# ---------------------------------------------------------------------------
# Isaac Sim 4.5 does not have set_merge_fixed_ignore_inertia; guard it.
URDF_CONV="${ISAAC_LAB_DIR}/source/isaaclab/isaaclab/sim/converters/urdf_converter.py"
if [ -f "${URDF_CONV}" ]; then
    if grep -q "set_merge_fixed_ignore_inertia" "${URDF_CONV}" && \
       ! grep -q "hasattr(import_config, .set_merge_fixed_ignore_inertia.)" "${URDF_CONV}"; then
        echo "Patching urdf_converter.py for Isaac Sim 4.x compatibility..."
        sed -i 's/import_config\.set_merge_fixed_ignore_inertia(self\.cfg\.merge_fixed_joints)/if hasattr(import_config, "set_merge_fixed_ignore_inertia"):\n            import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)/' \
            "${URDF_CONV}"
        echo "Patch applied."
    else
        echo "urdf_converter.py already patched or patch not needed."
    fi
fi

# ---------------------------------------------------------------------------
# 6.  Install everything inside the Singularity overlay
# ---------------------------------------------------------------------------
echo ""
echo "Starting Singularity container to install packages..."
echo "(This will take 30-60 minutes for Isaac Sim)"

singularity exec \
    --overlay "${OVERLAY}:rw" \
    "${SIF}" \
    /bin/bash -s << 'SINGULARITY_EOF'

set -e

# ---- Inside Singularity container ----

echo "=== Step 6a: Install Miniforge ==="
if [ ! -f /ext3/miniforge3/bin/conda ]; then
    wget -q --no-check-certificate \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /ext3/miniforge3
    rm /tmp/miniforge.sh
    echo "Miniforge installed."
else
    echo "Miniforge already installed, skipping."
fi

echo "=== Step 6b: Create env.sh wrapper ==="
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:/home/${USER}/.local/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
EOF
chmod +x /ext3/env.sh

echo "=== Step 6c: Activate conda and update ==="
source /ext3/env.sh
conda config --remove channels defaults 2>/dev/null || true
conda update -n base conda -y --quiet
conda clean --all --yes --quiet

echo "=== Step 6d: Install Python 3.10 ==="
conda install -n base python=3.10 pip -y --quiet
# Re-source so PATH picks up the newly installed pip/python binaries
source /ext3/env.sh
hash -r
# Sanity-check: must be 3.10
PY=/ext3/miniforge3/bin/python
PIP=/ext3/miniforge3/bin/pip
$PY --version
$PIP --version

echo "=== Step 6e: Install PyTorch (CUDA 12.1 compatible) ==="
# Use explicit pip path to guarantee we install into the conda env, not --user
$PIP install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 6f: Install Isaac Sim 4.5.0.0 ==="
$PIP install isaacsim==4.5.0.0 \
    --extra-index-url https://pypi.nvidia.com

echo "=== Step 6g: Install Isaac Lab source packages ==="
ISAAC_LAB_DIR="/scratch/${USER}/IsaacLab"
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab"
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab_assets"
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab_tasks"
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab_mirage"

echo "=== Step 6h: Install project dependencies ==="
$PIP install skrl==2.0.0 seaborn wandb imageio tensorboard

echo "=== Step 6i: Verify installation ==="
$PY -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
$PY -c "import skrl; print('skrl:', skrl.__version__)"

echo "=== Installation complete! ==="
SINGULARITY_EOF

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  To install the project package, run:"
echo "    bash slurm/hpc_install_project.sh"
echo ""
echo "  Then submit training with:"
echo "    sbatch --array=0-2 slurm/train_job.sh"
echo "=============================================="
