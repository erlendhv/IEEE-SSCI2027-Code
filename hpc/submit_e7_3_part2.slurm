#!/bin/bash
#SBATCH --job-name=E7.3b
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-04:00:00           # 4 hours: the 18-field settings (tasks 1600-1679) cost ~2x per evaluation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=16G
#SBATCH --array=0-919              # tasks 800-1719 of 86 configs x 20 seeds = 1720 tasks
#SBATCH --output=hpc/logs/E7.3_%A_%a.out
#SBATCH --error=hpc/logs/E7.3_%A_%a.err

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

# Slurm caps the highest array index, so part 2 indexes from 0 and offsets here
TASK_ID=$((SLURM_ARRAY_TASK_ID + 800))

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID (experiment task $TASK_ID)"
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
python hpc/run_single_task.py E7.3 \
    --task-id $TASK_ID \
    --n-seeds 20 \
    --workers 20

echo "End time: $(date)"
