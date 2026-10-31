#!/bin/bash
# Run the lightweight Optuna/Slurm orchestrator on a compute node.
# Edit the configuration variables below, then submit from a login node as:
#   sbatch runner_job.sh

#SBATCH --time=7-00:00:00   # walltime
#SBATCH --cpus-per-task=1   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --job-name='autorec_tuning_runner'   # job name
#SBATCH --account=rrg-j3goals
#SBATCH --constraint='turin'
#SBATCH --output=%x_slurm-%A.out   # output and error log

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# USER CONFIGURATION
LOGIN_NODE='login1'
TARGET_DIR='./hyperparameter_tuning_results'
BASE_CONFIG='./base_config.yaml'
SEARCH_SPACE='./search_space.yaml'
NUM_INITIAL=5
NUM_ITERATIONS=10
BATCH_SIZE=5
MAX_TRIALS=10000
SESSION_TRIALS=1000
GRACE_TRIALS=2000
REDUCTION_FACTOR=2

# ENVIRONMENT CONFIGURATION
MODULE_SAVELIST_NAME=${MODULE_SAVELIST_NAME:-default}
PYTHON_ENV_PATH=${PYTHON_ENV_PATH:-"$HOME/autorec_env"}

RUNNER_ARGUMENTS=(
    --login-node "$LOGIN_NODE"
    --target-dir "$TARGET_DIR"
    --base-config "$BASE_CONFIG"
    --search-space "$SEARCH_SPACE"
    --num-initial "$NUM_INITIAL"
    --num-iterations "$NUM_ITERATIONS"
    --batch-size "$BATCH_SIZE"
    --max-trials "$MAX_TRIALS"
    --session-trials "$SESSION_TRIALS"
    --grace-trials "$GRACE_TRIALS"
    --reduction-factor "$REDUCTION_FACTOR"
)

echo '------------------------------------------------------------'
echo "Running job $SLURM_JOB_NAME"
echo "Allocated node: $(hostname)"
echo "Job started at: $(date)"
echo '------------------------------------------------------------'

# Print all SBATCH parameters for this job
echo 'Printing SBATCH parameters...'
scontrol show job "$SLURM_JOB_ID"
echo '------------------------------------------------------------'

# Load necessary modules and activate environment
module restore "$MODULE_SAVELIST_NAME"
source "$PYTHON_ENV_PATH/bin/activate"

# RUN THE HYPERPARAMETER-TUNING ORCHESTRATOR
cd "$SCRIPT_DIR"
echo 'Running script: runner.py'
echo "Login node: $LOGIN_NODE"
echo "Arguments: ${RUNNER_ARGUMENTS[*]}"
echo

time python runner.py "${RUNNER_ARGUMENTS[@]}"

echo 'All done!'
echo "Job ended at: $(date)"
echo '------------------------------------------------------------'
