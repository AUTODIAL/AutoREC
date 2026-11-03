#!/bin/bash
# Submit the lightweight continuous-FIM tuning orchestrator with: sbatch tuning_runner_job.sh

#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name='progressive_continuous_fim_tuning_runner'
#SBATCH --account=rrg-j3goals
#SBATCH --constraint='turin'
#SBATCH --output=%x_slurm-%A.out

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${SLURM_SUBMIT_DIR:-"$SCRIPT_DIR"}

TARGET_DIR='./hyperparameter_tuning'
NUM_INITIAL=5
NUM_ITERATIONS=10
BATCH_SIZE=5
MAX_TRIALS=10000
SESSION_TRIALS=1000
GRACE_TRIALS=2000
REDUCTION_FACTOR=2

MODULE_SAVELIST_NAME="basic"
PYTHON_ENV_PATH="$HOME/eis_env"

RUNNER_ARGUMENTS=(
    --target-dir "$TARGET_DIR"
    --num-initial "$NUM_INITIAL"
    --num-iterations "$NUM_ITERATIONS"
    --batch-size "$BATCH_SIZE"
    --max-trials "$MAX_TRIALS"
    --session-trials "$SESSION_TRIALS"
    --grace-trials "$GRACE_TRIALS"
    --reduction-factor "$REDUCTION_FACTOR"
)

echo "Running ${SLURM_JOB_NAME} on $(hostname) at $(date)"
scontrol show job "$SLURM_JOB_ID"
module restore "$MODULE_SAVELIST_NAME"
source "$PYTHON_ENV_PATH/bin/activate"
cd "$RUN_DIR"
time python tuning_runner.py "${RUNNER_ARGUMENTS[@]}"
echo "Job ended at: $(date)"
