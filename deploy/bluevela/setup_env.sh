#!/usr/bin/env bash
# One-time environment setup for OpenMythos training on BlueVela.
#
# Usage:
#   ssh pzerfos@login4.bluevela.rmf.ibm.com
#   cd /path/to/OpenMythos
#   bash deploy/bluevela/setup_env.sh
#
# Required environment variables (set in ~/.bashrc or export before running):
#   CLEARML_API_HOST       -- ClearML server URL
#   CLEARML_API_ACCESS_KEY -- ClearML API access key
#   CLEARML_API_SECRET_KEY -- ClearML API secret key
#   HF_TOKEN               -- HuggingFace token for FineWeb-Edu access

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------
REQUIRED_VARS=(CLEARML_API_HOST CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY HF_TOKEN)
MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Set them in ~/.bashrc or export before running this script."
    echo "See docs/superpowers/specs/2026-04-22-1b-poc-training-design.md for details."
    exit 1
fi

echo "All required environment variables are set."

# ---------------------------------------------------------------------------
# Create conda env and install dependencies
# ---------------------------------------------------------------------------
CONDA_ENV_NAME="openmythos"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

echo "Setting up conda env '$CONDA_ENV_NAME' ..."

if conda info --envs | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Existing conda env found, reusing."
else
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
    echo "Created new conda env with Python 3.10."
fi

conda activate "$CONDA_ENV_NAME"

pip install --upgrade pip
pip install poetry
poetry install
pip install -r training/requirements.txt

echo "Dependencies installed."

# ---------------------------------------------------------------------------
# Verify ClearML connectivity
# ---------------------------------------------------------------------------
echo "Verifying ClearML connectivity..."
python3 -c "
from clearml import Task
print('ClearML SDK loaded successfully.')
print('ClearML OK.')
"

# ---------------------------------------------------------------------------
# Verify HuggingFace token
# ---------------------------------------------------------------------------
echo "Verifying HuggingFace token..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'  Logged in as: {user[\"name\"]}')
print('HuggingFace OK.')
"

# ---------------------------------------------------------------------------
# Verify model imports
# ---------------------------------------------------------------------------
echo "Verifying OpenMythos imports..."
python3 -c "
from open_mythos import OpenMythos
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer
cfg = mythos_1b()
print(f'  1B config: dim={cfg.dim}, experts={cfg.n_experts}, loops={cfg.max_loop_iters}')
print('OpenMythos OK.')
"

echo ""
echo "========================================="
echo "  Setup complete! Ready to submit jobs."
echo "  Run: bash deploy/bluevela/bsub_1b_poc.sh"
echo "========================================="
