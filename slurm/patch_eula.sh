#!/bin/bash
# =============================================================================
#  ONE-TIME PATCH: Bypass Isaac Sim EULA prompt for non-interactive Slurm runs
# =============================================================================
# omni/kit_app.py calls input() unconditionally which raises EOFError in batch.
# This patch replaces check_eula() with a no-op function.
#
# Submit with:
#   sbatch slurm/patch_eula.sh
# =============================================================================

#SBATCH --job-name=patch_eula
#SBATCH --account=rob_gy_73237-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/patch_eula_%j.log
#SBATCH --error=logs/patch_eula_%j.err
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

echo "=== Patching omni/kit_app.py to bypass EULA prompt ==="

singularity exec \
    --overlay "${OVERLAY}" \
    "${SIF}" \
    /bin/bash << 'INNEREOF'

KIT_APP="/ext3/miniforge3/lib/python3.10/site-packages/omni/kit_app.py"

echo "Target file: ${KIT_APP}"
if [ ! -f "${KIT_APP}" ]; then
    echo "ERROR: ${KIT_APP} not found"
    exit 1
fi

# Show the current check_eula block so we know what we're replacing
echo "--- Current check_eula function ---"
grep -n "check_eula\|eula\|EULA\|input(" "${KIT_APP}" | head -20

# Replace check_eula with a no-op using Python in-place edit
python3 - << 'PYEOF'
import re, sys

path = "/ext3/miniforge3/lib/python3.10/site-packages/omni/kit_app.py"
with open(path, "r") as f:
    src = f.read()

# Check if already patched
if "# EULA bypassed for headless HPC" in src:
    print("Already patched — nothing to do.")
    sys.exit(0)

# Replace the entire check_eula function body with a no-op
# Pattern: def check_eula(): ... up to next top-level def or class
patched = re.sub(
    r'(def check_eula\(\):).*?(?=\ndef |\nclass |\Z)',
    r'\1\n    # EULA bypassed for headless HPC (non-interactive batch)\n    return\n\n',
    src,
    flags=re.DOTALL
)

if patched == src:
    print("WARNING: Pattern not found — check_eula may have been renamed or restructured.")
    print("Dumping relevant lines for manual inspection:")
    for i, line in enumerate(src.splitlines(), 1):
        if "eula" in line.lower() or "input(" in line:
            print(f"  {i}: {line}")
    sys.exit(1)

with open(path, "w") as f:
    f.write(patched)
print("Patched successfully.")
PYEOF

echo ""
echo "--- Verifying patch ---"
grep -n "check_eula\|EULA bypassed\|return" "${KIT_APP}" | head -10

echo ""
echo "=== EULA patch complete ==="
INNEREOF
