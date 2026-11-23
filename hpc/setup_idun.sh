#!/bin/bash
# One-time setup for IDUN.
# Run this on the login node after cloning the repo.
#
# Usage:
#   cd <repo-root>
#   bash hpc/setup_idun.sh

set -e

echo "=== Setting up IDUN environment ==="

# Load Python
module purge
module load Python/3.11.3-GCCcore-12.3.0
echo "Python version: $(python --version)"

# Create virtual environment
if [ -d "venv_idun" ]; then
    echo "venv_idun already exists, skipping creation"
else
    echo "Creating virtual environment..."
    python -m venv venv_idun
fi

# Activate and install dependencies
source venv_idun/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory for Slurm output
mkdir -p hpc/logs

# Verify import works
echo "Verifying installation..."
python -c "
from src.experiments.configs import get_experiment_configs
configs = get_experiment_configs('E5.2')
print(f'E5.2: {len(configs)} configs loaded successfully')
print(f'Total tasks with 10 seeds: {len(configs) * 10}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Before submitting, find your account name:"
echo "  sacctmgr show assoc format=Account%15,User,QOS | grep \$USER"
echo ""
echo "Then set --account in the hpc/submit_*.slurm script and submit, e.g.:"
echo "  sbatch hpc/submit_e1_1.slurm"
