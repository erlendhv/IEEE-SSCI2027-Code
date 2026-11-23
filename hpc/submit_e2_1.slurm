#!/bin/bash
#SBATCH --job-name=E2.1
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=16G
#SBATCH --array=0-39               # 2 configs x 20 seeds = 40 tasks
#SBATCH --output=hpc/logs/E2.1_%A_%a.out
#SBATCH --error=hpc/logs/E2.1_%A_%a.err

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Working directory: $WORKDIR"
echo "Start time: $(date)"
echo "================"

# Load Python and activate virtual environment
module purge
module load Python/3.11.3-GCCcore-12.3.0

source venv_idun/bin/activate
export PYTHONPATH=${WORKDIR}

# Run single task
python hpc/run_single_task.py E2.1 \
    --task-id $SLURM_ARRAY_TASK_ID \
    --n-seeds 20 \
    --workers 20

echo "End time: $(date)"
