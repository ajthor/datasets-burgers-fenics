#!/bin/bash
set -e

# Download the Dedalus install script to a temporary location
DEDALUS_INSTALL_SCRIPT="$(mktemp)"
curl -fsSL https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/conda_install_dedalus3.sh -o "$DEDALUS_INSTALL_SCRIPT"

# Ensure conda is initialized (adjust path if needed for your container)
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Install Dedalus only if the environment does not exist
if ! conda info --envs | grep -q dedalus3; then
    conda activate base
    bash "$DEDALUS_INSTALL_SCRIPT"
fi


conda activate dedalus3

conda env config vars set OMP_NUM_THREADS=1
conda env config vars set NUMEXPR_MAX_THREADS=1

# Install additional Python packages from requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Clean up the temporary install script
rm -f "$DEDALUS_INSTALL_SCRIPT"
