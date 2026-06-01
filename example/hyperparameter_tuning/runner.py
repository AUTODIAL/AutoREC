"""Here we try hyperparameter tuning using Optuna. Note that we already have some
precomputed scores that we can use to initially train Optuna.

Usage:
```bash
    $ python runner.py \
        --target-dir ./hyperparameter_tuning_results \
        --base-config ./base_config.yaml \
        --num-initial 5 \
        --num-iterations 10 \
        --batch-size 5 \
```

Arguments:
    --target-dir: Target directory to store results
    --base-config: Base configuration file to use for generating new configs
    --num-initial: Number of initial trials
    --num-iterations: Number of iterations to run
    --batch-size: Batch size for each iteration
"""

from pathlib import Path
import argparse
from pprint import pprint

import numpy as np
import optuna

from tuning_utils import (
    ConfigurationHandler,
    write_job_script,
    block_until_completed,
    compute_score,
)

# For result visualization analysis
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis_matplotlib


seed = 42
np.random.seed(seed)

# Directories
WORK_DIR = Path(__file__).parent


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--target-dir",
    type=str,
    default="./hyperparameter_tuning_results",
    help="Target directory to store results",
)
parser.add_argument(
    "--base-config",
    type=str,
    default="./base_config.yaml",
    help="Base configuration file to use for generating new configs",
)
parser.add_argument("--num-initial", type=int, default=10, help="Number of initial trials")
parser.add_argument(
    "--num-iterations", type=int, default=5, help="Number of iterations to run"
)
parser.add_argument("--batch-size", type=int, default=5, help="Batch size for each iteration")
args = parser.parse_args()


# Hyperparameter tuning settings
RESULTS_DIR = Path(args.target_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
configs_dir = RESULTS_DIR / "configs"
num_initial = args.num_initial
batch_size = args.batch_size
num_iterations = args.num_iterations
# Other variables not exposed to command line
nlast = 1000  # How many last episodes to average over for mean reward
ndata_per_circuit = 270  # Number of data points per circuit in the dataset
python_file = "main_factory.py"  # The training script to use
base_config_file = Path(args.base_config)
base_sbatch_options = {"time": "72:00:00", "mem_per_cpu": "12G"}

# Instantiate configuration handler
config_handler = ConfigurationHandler(base_config=base_config_file, configs_dir=configs_dir)


# Print out some information
print("=====================================")
print("# HYPERPARAMETER TUNING WITH OPTUNA #")
print("=====================================\n")

print("Results will be stored in:", RESULTS_DIR)
print("Configuration files will be stored in:", configs_dir, "\n")

print("Number of initial trials:", num_initial)
print("Number of iterations:", num_iterations)
print("Batch size per iteration:", batch_size, "\n")

print("Using base configuration file:", base_config_file)
pprint(config_handler.base_config, sort_dicts=False)
print()
print("Hyperparameter search space bounds:")
pprint(config_handler.search_space_bounds, sort_dicts=False)
print()


##########################################################################################
# HYPERPARAMETER TUNING WITH OPTUNA: INITIAL TRIALS
##########################################################################################
print("Starting initial trials of hyperparameter tuning with Optuna...")

# For storing the results
all_params = {}
all_scores = {}

# Search space
search_space = {
    name: (
        optuna.distributions.IntDistribution(*bounds)
        if name in config_handler.integer_variables
        else optuna.distributions.FloatDistribution(*bounds)
    )
    for name, bounds in config_handler.search_space_bounds.items()
}
# Create an Optuna study
sampler = optuna.samplers.TPESampler(n_startup_trials=num_initial, seed=seed)
study = optuna.create_study(sampler=sampler, direction="maximize")

# Check files to skip running jobs if these results already exist
# These files are for computing the scores
check_files = ["dqn_model.keras", "statistical_analysis.pkl", "eval_results.pkl"]

# MAKE SURE TO INCLUDE THE ORIGINAL HYPERPARAMETER SET
print("We make sure that the original hyperparameter set is included in the initial trials.")
config_id, config_file, config = config_handler.write_config(
    config_handler.base_config, RESULTS_DIR
)
SAMPLE_DIR = Path(config["agent"]["save_dir"])
# Write and submit job script
training_done = all([(SAMPLE_DIR / ff).exists() for ff in check_files])
if not training_done:
    print("Submitting job for the original hyperparameter set...")
    sbatch_options = base_sbatch_options.copy()
    sbatch_options["job_name"] = f"hyperparam_tuning_init_{config_id}"
    jobid = write_job_script(SAMPLE_DIR, python_file, config_file, sbatch_options, submit=True)
    # Block until all jobs are finished
    print("Waiting for all initial jobs to finish...")
    block_until_completed([jobid])
# Compute the score
avg_score, score_elements = compute_score(SAMPLE_DIR, nlast)
# Tell Optuna about the result
print("Inserting the original hyperparameter set into the study...")
params = config_handler.config_to_sample(config)
assert set(params.keys()) == set(search_space.keys())
study.add_trial(
    optuna.trial.create_trial(params=params, distributions=search_space, value=avg_score)
)
print(f"Inserted original hyperparameter set with score {avg_score:.4f}\n")
# Store the resuts
all_params[config_id] = params
all_scores[config_id] = np.append([avg_score], score_elements)


# INITIAL TRIALS
num_initial -= 1  # Already did one above using the original hyperparameter set
print(f"Performing initial trials for {num_initial} configurations...")
trials = [study.ask(search_space) for _ in range(num_initial)]
config_id_list = []
jobid_list = []
for trial in trials:
    # Write the configuration file
    config_id, config_file, config = config_handler.write_config(
        config_handler.sample_to_config(trial.params), RESULTS_DIR
    )
    config_id_list.append(config_id)
    SAMPLE_DIR = Path(config["agent"]["save_dir"])
    # Write and submit job script
    training_done = all([(SAMPLE_DIR / ff).exists() for ff in check_files])
    if not training_done:
        sbatch_options = base_sbatch_options.copy()
        sbatch_options["job_name"] = f"hyperparam_tuning_init_{config_id}"
        jobid = write_job_script(
            SAMPLE_DIR, python_file, config_file, sbatch_options, submit=True
        )
        jobid_list.append(jobid)
if len(jobid_list) > 0:
    # Block until all jobs are finished
    print("Waiting for all initial jobs to finish...")
    block_until_completed(jobid_list)

# Collect results from initial trials
print("Collecting results from initial trials...")
score_list = []
for ii, config_id in enumerate(config_id_list):
    SAMPLE_DIR = config_handler.configs_list[config_id]["agent"]["save_dir"]
    avg_score, score_elements = compute_score(SAMPLE_DIR, nlast)
    print("Config ID:", config_id, "Score:", avg_score)
    score_list.append(avg_score)
    # Add to the list of results
    all_params[config_id] = trials[ii].params
    all_scores[config_id] = np.append([avg_score], score_elements)

# Tell Optuna about the results from initial trials
for trial, score in zip(trials, score_list):
    study.tell(trial, score)


##########################################################################################
# HYPERPARAMETER TUNING WITH OPTUNA: CONTINUED TRIALS
##########################################################################################
print("Continuing hyperparameter tuning with Optuna...")

for iteration in range(num_iterations - 1):
    print(f"Starting iteration {iteration + 2}/{num_iterations}...")
    trials = [study.ask(search_space) for _ in range(batch_size)]
    config_id_list = []
    jobid_list = []
    for trial in trials:
        # Write the configuration file
        config_id, config_file, config = config_handler.write_config(
            config_handler.sample_to_config(trial.params), RESULTS_DIR
        )
        config_id_list.append(config_id)
        SAMPLE_DIR = Path(config["agent"]["save_dir"])
        # Write and submit job script
        training_done = all([(SAMPLE_DIR / ff).exists() for ff in check_files])
        if not training_done:
            sbatch_options = base_sbatch_options.copy()
            sbatch_options["job_name"] = f"hyperparam_tuning_{config_id}"
            jobid = write_job_script(
                SAMPLE_DIR, python_file, config_file, sbatch_options, submit=True
            )
            jobid_list.append(jobid)
    if len(jobid_list) > 0:
        # Block until all jobs are finished
        print(f"Waiting for all jobs in iteration {iteration + 2} to finish...")
        block_until_completed(jobid_list)
    # Collect results from trials
    print(f"Collecting results from iteration {iteration + 2}...")
    score_list = []
    for ii, config_id in enumerate(config_id_list):
        SAMPLE_DIR = config_handler.configs_list[config_id]["agent"]["save_dir"]
        avg_score, score_elements = compute_score(SAMPLE_DIR, nlast)
        print("Config ID:", config_id, "Score:", avg_score)
        score_list.append(avg_score)
        # Add to the list of results
        all_params[config_id] = trials[ii].params
        all_scores[config_id] = np.append([avg_score], score_elements)
    # Tell Optuna about the results from trials
    for trial, score in zip(trials, score_list):
        study.tell(trial, score)

print("Hyperparameter tuning with Optuna completed.")


# Final step: Print the best hyperparameter set found
best_trial = study.best_trial
print("\nBest hyperparameter set found:")
best_params = best_trial.params
best_config = config_handler.sample_to_config(best_params)
best_config_id = config_handler.get_config_id(best_config)
print(f"Configuration ID: {best_config_id}")
pprint(best_params, sort_dicts=False)
# pprint(best_config, sort_dicts=False)
print(f"\nWith score: {best_trial.value:.4f}")


# Final step: Print the best hyperparameter set found
best_trial = study.best_trial
print("\nBest hyperparameter set found:")
best_params = best_trial.params
best_config = config_handler.sample_to_config(best_params)
best_config_id = config_handler.get_config_id(best_config)
print(f"Configuration ID: {best_config_id}")
pprint(best_params, sort_dicts=False)
# pprint(best_config, sort_dicts=False)
print(f"\nWith score: {best_trial.value:.4f}")

# Export the study results
study_df = study.trials_dataframe()
study_df.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
# Visualization analysis
print("Generating visualization analysis")
print("- Optimization history")
ax = vis_matplotlib.plot_optimization_history(study)
fig = ax.figure
fig.tight_layout()
fig.savefig(RESULTS_DIR / "optimization_history.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("- Parameter importances")
ax = vis_matplotlib.plot_param_importances(study)
fig = ax.figure
fig.tight_layout()
fig.savefig(RESULTS_DIR / "param_importances.png", dpi=300, bbox_inches="tight")
plt.close(fig)
