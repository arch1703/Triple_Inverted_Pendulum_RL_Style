#!/bin/bash
# ONE-TIME environment setup for NYU HPC (run inside an interactive job)
# creates a Singularity overlay with Miniforge + all Python packages (~30-60 min)
#
# usage:
#   srun --partition=g2-standard-12 --account=rob_gy_73237-2026sp \
#        --cpus-per-task=4 --mem=40GB --time=04:00:00 --pty /bin/bash
#   cd /scratch/${USER}/triple_pendulum
#   bash slurm/hpc_setup.sh

set -e  # exit on any error

NETID="${USER}"
PROJECT_DIR="/scratch/${NETID}/triple_pendulum"
ISAAC_LAB_DIR="/scratch/${NETID}/IsaacLab"
ENV_DIR="${PROJECT_DIR}/isaac_env"
OVERLAY="${ENV_DIR}/isaac_sim.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

echo "setup: project=${PROJECT_DIR} overlay=${OVERLAY}"

# verify we are NOT on the login node
if [ -z "${SLURM_JOB_ID}" ]; then
    echo "WARNING: no SLURM_JOB_ID - are you on a login node? (2GB RAM limit)"
    read -r -p "Continue anyway? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

# check the Singularity image
if [ ! -f "${SIF}" ]; then
    echo "ERROR: ${SIF} not found - check: ls /share/apps/images/ | grep cuda"
    exit 1
fi

# create overlay
mkdir -p "${ENV_DIR}"

if [ -f "${OVERLAY}" ]; then
    echo "overlay already exists at ${OVERLAY}"
else
    echo "available overlay templates:"
    ls /share/apps/overlay-fs-ext3/ 2>/dev/null || echo "(none found)"
    echo "looking for 25GB+ overlay (Isaac Sim needs ~20GB)..."

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

echo "=== Step 6d.1: Bootstrap pip + setuptools ==="
# Pin setuptools to 69.x — last version that reliably ships pkg_resources.
# Newer setuptools (70+) can break old-style setup.py packages like flatdict
# which is pulled in as an Isaac Sim transitive dependency.
$PIP install --upgrade pip "setuptools==69.5.1" wheel

echo "=== Step 6e: Install PyTorch (CUDA 12.1 compatible) ==="
# Use explicit pip path to guarantee we install into the conda env, not --user
$PIP install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 6f: Install Isaac Sim 4.5.0.0 ==="
# Pre-install the exact flatdict versions both isaacsim and isaaclab need.
# flatdict uses a legacy setup.py that breaks in pip's isolated build venv
# (ModuleNotFoundError: No module named 'pkg_resources').
# Installing with --no-build-isolation uses our pinned setuptools==69.5.1
# which still ships pkg_resources.
$PIP install "flatdict==4.0.1" --no-build-isolation
$PIP install \
    isaacsim==4.5.0.0 \
    isaacsim-rl==4.5.0.0 \
    isaacsim-replicator==4.5.0.0 \
    isaacsim-extscache-physics==4.5.0.0 \
    isaacsim-extscache-kit==4.5.0.0 \
    isaacsim-extscache-kit-sdk==4.5.0.0 \
    --extra-index-url https://pypi.nvidia.com \
    --no-build-isolation

echo "=== Step 6g: Install Isaac Lab source packages ==="
ISAAC_LAB_DIR="/scratch/${USER}/IsaacLab"
# Pre-install all isaaclab transitive deps that use legacy setup.py and lack
# pre-built cp310 wheels. Each of these would otherwise be built inside pip's
# fresh isolated build venv which carries a new setuptools without pkg_resources.
# Pre-installing them here (with our pinned setuptools==69.5.1 and
# --no-build-isolation) means pip finds them already satisfied and skips building.
#   flatdict==4.0.1  — legacy setup.py, imports pkg_resources
#   toml             — imported directly in isaaclab's setup.py at build time
#   hidapi==0.14.0.post2 — C extension with setup.py
#   pyglet==1.5.31   — old enough to use setup.py (no pyproject.toml)
#   prettytable==3.3.0  — pinned old version, setup.py based
#   hatchling+hatch-vcs — isaaclab's own build backend (pyproject.toml)
$PIP install hatchling hatch-vcs --no-build-isolation
$PIP install "flatdict==4.0.1" toml "hidapi==0.14.0.post2" \
             "pyglet==1.5.31" "prettytable==3.3.0" --no-build-isolation
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab" --no-build-isolation
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab_assets" --no-build-isolation
$PIP install -e "${ISAAC_LAB_DIR}/source/isaaclab_tasks" --no-build-isolation
# isaaclab_mirage was removed in Isaac Lab v2.x — skip it
# isaaclab_tasks pulls in a CPU torchvision and overwrites our CUDA build.
# Re-pin the CUDA versions immediately after.
$PIP install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 6h: Install project dependencies ==="
$PIP install h5py skrl==2.0.0 seaborn wandb imageio tensorboard

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
