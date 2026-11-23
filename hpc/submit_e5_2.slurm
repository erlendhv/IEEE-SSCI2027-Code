#!/bin/bash
#SBATCH --job-name=E5.2
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00           # 1 hour per task (runs take ~21 min, buffer for safety)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=22          # 20 workers + main process + overhead
#SBATCH --mem=16G
#SBATCH --array=0-399               # 40 configs x 10 seeds = 400 tasks
#SBATCH --output=hpc/logs/E5.2_%A_%a.out
#SBATCH --error=hpc/logs/E5.2_%A_%a.err

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
python hpc/run_single_task.py E5.2 \
    --task-id $SLURM_ARRAY_TASK_ID \
    --n-seeds 10 \
    --workers 20

echo "End time: $(date)"
