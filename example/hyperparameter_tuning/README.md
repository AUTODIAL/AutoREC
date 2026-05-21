# Hyperparameter Tuning Example for Training an RL Agent

This example demonstrates how to perform hyperparameter tuning for training an AutoREC reinforcement learning agent using [`Optuna`](https://optuna.org/).
The example provides scripts for defining the hyperparameter search space, submitting training jobs on an HPC cluster, and collecting the tuning results.



## Prerequisites

The hyperparameter tuning script in this example is intended to be run on an HPC cluster using the Slurm workload manager.

The dependencies for this example can be installed with:

```bash
pip install -r requirements.txt
```

Depending on your cluster environment, you may also need to load the appropriate Python, Julia, CUDA, or MPI modules before installing the dependencies or running the scripts.



## How to run

To run the hyperparameter tuning workflow, execute:

```bash
python runner.py \
  --target-dir ./hyperparameter_tuning_results \
  --base-config ./base_config.yaml \
  --num-initial 5 \
  --num-iterations 10 \
  --batch-size 5
```

Arguments:

- `--target-dir`: Directory where the tuning results will be saved.
- `--base-config`: Path to the base configuration file for training the RL agent.
- `--num-initial`: Number of initial random configurations to evaluate before starting the optimization process.
- `--num-iterations`: Number of optimization iterations to perform.
- `--batch-size`: Number of configurations to evaluate in parallel during each optimization iteration.



## Contents

- `runner.py`: Main script that orchestrates the hyperparameter tuning workflow.
  This script creates an Optuna study, writes and submits Slurm jobs for evaluating different hyperparameter configurations, and collects the results.

- `tuning_utils.py`: Utility functions for the tuning process.
  This file includes functions for writing Slurm job scripts, submitting jobs, computing the objective function to maximize, and defining the hyperparameter search space.

  **Note:** To run this example on a different HPC cluster, you may need to modify the job script template in the `write_job_script` function to match the requirements of your cluster's job submission system.

- `main_factory.py`: Script that trains the RL agent for a given set of hyperparameters.

- `base_config.yaml`: Base configuration file for training the RL agent.
  This file serves three purposes:

  1. Provides values for hyperparameters that are not tuned.
  2. Provides a template for the configuration files generated during tuning.
  3. Provides baseline values for the hyperparameters that are tuned and later overridden by the tuning process.

  **Note:** The tuning workflow in `runner.py` first runs training using the baseline configuration from `base_config.yaml`.
  This provides a reference point before evaluating optimized hyperparameter configurations.



## Expected workflow

The hyperparameter tuning workflow is handled by `runner.py`.
When executed, the workflow proceeds as follows:

1. A baseline training run is submitted using the hyperparameter values specified in `base_config.yaml`.
2. An initial set of random hyperparameter configurations is generated and evaluated.
3. For each tuning iteration, `runner.py` uses Optuna to propose a batch of new hyperparameter configurations.
4. For each configuration in the batch, `runner.py` generates a corresponding configuration file and Slurm job script.
5. The Slurm jobs for the current batch are submitted to the HPC cluster.
6. `runner.py` waits until all jobs in the current batch are completed.
7. After the batch is completed, the results from each training run are collected and used to compute the objective values.
8. The objective values are reported back to the Optuna study.
9. Optuna uses the completed trials to propose the next batch of hyperparameter configurations.
10. Steps 3--9 are repeated until the requested number of tuning iterations is completed.



## Outputs

The tuning results are saved in the directory specified by `--target-dir`.
Each training run is stored in a separate subdirectory inside the target directory.
The directory name follows the format:

```text
run_0000/  # Using baseline configuration from base_config.yaml
run_0001/
run_0002/
...
```

where the index corresponds to the hyperparameter configuration used for that run.
The corresponding configuration file for each run is stored in:

```text
configs/config_0000.yaml  # Baseline configuration from base_config.yaml
configs/config_0001.yaml
configs/config_0002.yaml
...
```

For example, the training result in `run_0005/` was generated using the hyperparameter configuration stored in `configs/config_0005.yaml`

Each `run_XXXX/` directory contains the outputs from training and evaluation.
The main output files are:

- `dqn_model.keras`: Final trained DQN model from the training run.
- `statistical_analysis.pkl`: Training statistics, including reward information collected during training.
- `eval_results.pkl`: Evaluation results obtained by evaluating the final model on the training set. These results are used to compute metrics such as the success rate and exact rate.
- Other intermediate outputs, such as models saved during training, neural-network loss values, and additional training logs.

The best-performing hyperparameter configuration is not exported as a separate configuration file by default.
Instead, it can be identified from the collected tuning results by finding the run with the best objective value.
The corresponding hyperparameter configuration can then be retrieved from the matching file in the `configs/` directory.
