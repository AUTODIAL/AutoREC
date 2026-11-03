#!/bin/bash

# Submit from this directory with: sbatch submit_job.sh

#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name=progressive_simplified_reward
#SBATCH --constraint=turin
#SBATCH --account=rrg-j3goals
#SBATCH --output=%x_slurm-%j.out

echo "------------------------------------------------------------"
echo "Running job ${SLURM_JOB_NAME}"
echo "Allocated node: $(hostname)"
echo "Job started at: $(date)"
echo "------------------------------------------------------------"

echo "SLURM job parameters:"
scontrol show job "${SLURM_JOB_ID}"
echo "------------------------------------------------------------"

module restore basic
source "$HOME/eis_env/bin/activate"
export OMP_NUM_THREADS=1

echo "Running script: main_factory.py"
echo "Configuration: ~/modules/AutoREC/example/progressive_exploration/configs/pipeline_config.yaml"
echo "Output directory: calculation_output"

time python main_factory.py \
    --config ~/modules/AutoREC/example/progressive_exploration/configs/pipeline_config.yaml \
    --output-dir calculation_output

echo "Archiving model and fit-plot outputs..."
if [ -d calculation_output/models ]; then
    tar -czf calculation_output/models.tar.gz -C calculation_output models &&
        rm -rf calculation_output/models
fi
for fit_plots_dir in calculation_output/results/fit_plots_*; do
    if [ -d "$fit_plots_dir" ]; then
        tar -czf "${fit_plots_dir}.tar.gz" -C calculation_output/results \
            "$(basename "$fit_plots_dir")" && rm -rf "$fit_plots_dir"
    fi
done

echo "------------------------------------------------------------"
echo "All done!"
echo "Job ended at: $(date)"
echo "------------------------------------------------------------"
