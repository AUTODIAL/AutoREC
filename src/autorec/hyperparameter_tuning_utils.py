"""Collection of utility functions and variables for hyperparameter tuning and
configuration management.
"""

from pathlib import Path
from glob import glob
import copy
import yaml
import re
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import autoeis as ae


# Hyperparameter search bounds
search_space_bounds = {
    # NN hyperparameters
    "batch_size_x50": [1, 10],
    "buffer_capacity_x1000": [1, 20],
    "train_frequency": [1, 100],
    "update_target_frequency_x100": [1, 50],
    # Dynamic variables
    "initial_epsilon": [0.5, 1.0],
    "epsilon_decay": [0.8, 0.9999],
    "epsilon_min": [0.0, 0.1],
    # Prioritized replay parameters
    "prioritized_replay_alpha": [0.5, 1.5],
    "initial_beta": [0.01, 0.6],
    "final_p": [0.7, 1.3],
}
# Some hyperparameters are constrained to be integers
integer_variables = [
    "batch_size_x50",
    "buffer_capacity_x1000",
    "train_frequency",
    "update_target_frequency_x100",
]
# Convert the search space bounds into a N-by-2 array - Sometimes it is easier to work
# with arrays than with dictionaries.
search_space_bounds_array = np.array(
    [list(bounds) for bounds in search_space_bounds.values()]
)


def sample_to_config(row, base_config):
    """Convert a sample into the configuration dictionary format.

    Note that the environment variables don't need to be changed.

    Parameters
    ----------
    row : array-like
        A single sample from the hyperparameter search space.
    base_config : dict
        The base configuration dictionary to modify.

    Returns
    -------
    config : dict
        The modified configuration dictionary with hyperparameters set.
    """
    config = copy.deepcopy(base_config)
    for name, value in zip(list(search_space_bounds), row):
        if name in integer_variables:
            # Sometimes we also need to scale the integer variable
            match = re.match(r"^(.*)_x(\d+)$", name)
            if match:
                name, scale = match.groups()
                value = int(scale * round(value))
            config["agent"][name] = int(round(value))
        else:
            config["agent"][name] = float(value)
    return config


class CustomLogScaler:
    def __init__(self, integer_features, continuous_features):
        self.integer_features = integer_features
        self.continuous_features = continuous_features
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        # Log-transform continuous features (log1p handles 0 safely)
        log_transformed = np.log(df[self.continuous_features].values)
        # Fit internal StandardScaler
        self.scaler.fit(log_transformed)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise ValueError("CustomLogScaler must be fitted before calling transform().")
        # Copy to avoid modifying original data
        df_new = df.copy()
        # Apply log + standardize to continuous features
        df_new[self.continuous_features] = self.scaler.transform(
            np.log(df[self.continuous_features].values)
        )
        # Leave integer features untouched
        # (If you want, you can also scale them—just tell me)
        return df_new

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def inverse_transform(self, df_scaled):
        df = df_scaled.copy()
        # Undo standardization → log-space
        log_vals = self.scaler.inverse_transform(df[self.continuous_features].values)
        # Undo log → original space
        df[self.continuous_features] = np.exp(log_vals)
        # Integers remain unchanged
        return df


class ConfigurationHandler:
    """Class to handle configuration conversion, writing, comparison and storage."""

    def __init__(
        self,
        base_config,
        configs_dir,
        search_space_bounds=search_space_bounds,
        integer_variables=integer_variables,
        basename="config",
    ):
        if isinstance(base_config, (str, Path)):
            with open(base_config, "r") as f:
                self.base_config = yaml.safe_load(f)
        elif isinstance(base_config, dict):
            self.base_config = base_config
        else:
            raise ValueError("base_config must be a dict or a path to a YAML file.")

        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(exist_ok=True, parents=True)
        self.search_space_bounds = search_space_bounds
        self.integer_variables = integer_variables
        self.basename = basename

        # Collect existing configurations to avoid duplicates
        self.configs_list, self.configs_ids = self._load_existing_configs()
        #  Find missing config IDs to fill gaps - Config IDs should be consecutive numbers
        self.missing_config_ids = sorted(
            set(range(max(self.configs_ids) + 1)) - set(self.configs_ids)
            if self.configs_ids
            else []
        )

    def _load_existing_configs(self):
        """Load existing configurations from the configs directory."""
        config_file_pattern = str(self.configs_dir / "config_[0-9][0-9][0-9][0-9].yaml")
        all_config_files = sorted(glob(config_file_pattern))
        existing_configs = {}
        for config_file in all_config_files:
            match = re.search(r"config_(\d+)\.yaml", Path(config_file).name)
            config_number = int(match.group(1))
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                existing_configs[config_number] = config
        # Get the existing config IDs to continue numbering
        existing_configs_ids = list(existing_configs.keys())
        return existing_configs, existing_configs_ids

    def get_config_id(self, config):
        """Get a unique configuration ID for a new configuration."""
        # Check if the configuration already exists
        for existing_id, existing_config in self.configs_list.items():
            if self.configs_equal(config, existing_config):
                return existing_id
        # If not, assign a new ID
        if self.missing_config_ids:
            return self.missing_config_ids.pop(0)
        else:
            return max(self.configs_ids) + 1 if self.configs_ids else 0

    def configs_equal(self, config1, config2):
        """Check if two configurations are equal, ignoring non-essential fields."""
        config1_copy = copy.deepcopy(config1)
        config2_copy = copy.deepcopy(config2)
        # Ignore save_dir when comparing
        config1_copy["agent"]["save_dir"] = None
        config2_copy["agent"]["save_dir"] = None
        return config1_copy == config2_copy

    def sample_to_config(self, sample):
        """Convert a sample configuration dictionary into the full configuration format.

        Parameters
        ----------
        sample : dict
            A dictionary containing hyperparameter values.

        Returns
        -------
        config : dict
            The modified configuration dictionary with hyperparameters set.
        """
        config = copy.deepcopy(self.base_config)
        for name, value in sample.items():
            # Some variables need to be integers
            if name in self.integer_variables:
                # And sometimes we also need to scale the integer variable
                match = re.match(r"^(.*)_x(\d+)$", name)
                if match:
                    name, scale = match.groups()
                    value = int(scale) * int(round(value))
            config["agent"][name] = value
        return config

    def config_to_sample(self, config):
        """Convert a full configuration dictionary into a sample configuration dictionary.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Returns
        -------
        sample : dict
            A dictionary containing only the hyperparameter values.
        """
        sample = {}
        for name in self.search_space_bounds.keys():
            if name in self.integer_variables:  # Some variables are integers
                # And some of them need to be scaled down
                match = re.match(r"^(.*)_x(\d+)$", name)
                if match:
                    orig_name, scale = match.groups()
                    value = int(round(config["agent"][orig_name] / int(scale)))
            else:
                value = config["agent"][name]
            sample[name] = value
        return sample

    def add_config(self, config, config_id=None):
        """Add a configuration to the internal storage without writing to file.

        Parameters
        ----------
        config : dict
            The configuration dictionary to add.
        config_id : int, optional
            The unique ID to assign to the configuration. If None, a new ID will be
            generated.

        Returns
        -------
        config_id : int
            The unique ID assigned to the configuration.
        """
        if config_id is None:
            config_id = self.get_config_id(config)
        if config_id in self.configs_ids:
            # Configuration already exists, no need to add
            return config_id
        # Store the new configuration
        self.configs_list[config_id] = config
        self.configs_ids.append(config_id)
        return config_id

    def remove_config(self, item):
        """Remove a configuration from the internal storage.

        Parameters
        ----------
        item : int or dict
            The configuration ID or configuration dictionary to remove.
        """
        if isinstance(item, dict):
            config_id = self.get_config_id(item)
        elif isinstance(item, int):
            config_id = item
        else:
            raise ValueError("item must be an int (config ID) or a dict (configuration).")

        if config_id in self.configs_ids:
            self.configs_ids.remove(config_id)
            self.configs_list.pop(config_id, None)
            # Update the list of missing IDs
            self.missing_config_ids.append(config_id)
            # Optionally, remove the file as well
            config_filename = self.configs_dir / f"{self.basename}_{config_id:04d}.yaml"
            if config_filename.exists():
                config_filename.unlink()

    def write_config(self, config, results_dir=None):
        """Write a configuration to a YAML file.

        Parameters
        ----------
        config : dict
            The configuration dictionary to write.
        results_dir : str or Path, optional
            If provided, update the save_dir in the configuration to point to a
            subdirectory in results_dir.

        Returns
        -------
        config_id : int
            The unique ID assigned to the configuration.
        """
        config_id = self.get_config_id(config)
        config_filename = self.configs_dir / f"{self.basename}_{config_id:04d}.yaml"
        if config_id in self.configs_ids:
            # Configuration already exists, no need to write
            return config_id, config_filename, self.configs_list[config_id]
        else:
            if results_dir is not None:
                config["agent"]["save_dir"] = str(results_dir / f"run_{config_id:04d}")
            self.add_config(config)

        with open(config_filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        return config_id, config_filename, config


def write_job_script(save_dir, python_file, config_file, sbatch_options, submit=False):
    time = sbatch_options.get("time", "2:00:00")
    cpus_per_task = sbatch_options.get("cpus_per_task", 2)
    nodes = sbatch_options.get("nodes", 1)
    mem_per_cpu = sbatch_options.get("mem_per_cpu", "4G")
    job_name = sbatch_options.get("job_name", "hyperparameter_tuning")

    submit_script = "\n".join(
        [
            "#!/bin/bash",
            "",
            f"#SBATCH --time={time}   # walltime",
            f"#SBATCH --cpus-per-task={cpus_per_task}   # number of processor cores",
            f"#SBATCH --nodes={nodes}   # number of nodes",
            f"#SBATCH --mem-per-cpu={mem_per_cpu}   # memory per CPU core",
            f"#SBATCH --job-name='{job_name}'   # job name",
            "#SBATCH --account=rrg-j3goals",
            "#SBATCH --constraint='turin'",
            f"#SBATCH --output={save_dir}/%x_slurm-%A-%a.out   # output and error log",
            "",
            "echo '------------------------------------------------------------'",
            "echo 'Running job $SLURM_JOB_NAME'",
            "echo 'Allocated node: `hostname`'",
            "echo 'Job started at: `date`'",
            "echo '------------------------------------------------------------'",
            ""
            "# Print all SBATCH parameters for this job"
            "echo 'Printing SBATCH parameters...'",
            "scontrol show job $SLURM_JOB_ID",
            "echo '------------------------------------------------------------'",
            "",
            "# DEFINE YOUR ENVIRONMENT VARIABLES HERE",
            "MODULE_SAVELIST_NAME='basic'",
            "PYTHON_ENV_PATH=$HOME/eis_env",
            "",
            "# Load necessary modules and activate environment",
            "export OMP_NUM_THREADS=1",
            "module restore ${MODULE_SAVELIST_NAME}",
            "source ${PYTHON_ENV_PATH}/bin/activate",
            "",
            "# RUN YOUR PYTHON SCRIPT",
            "# DEFINE YOUR PYTHON SCRIPT PARAMETERS HERE",
            "SCRIPTS='main_factory.py'",
            f"ARGS='-c {config_file}'",
            "",
            "# Run calculation",
            "echo 'Running script: ${SCRIPTS}'",
            "echo 'Arguments: ${ARGS}'",
            "",
            "time python ${SCRIPTS} ${ARGS}",
            "",
            "# TAR INTERMEDIATE MODELS",
            "echo 'Archiving intermediate models...'",
            f"cd {save_dir}",
            "tar -czvf models.tar.gz models",
            "rm -rf models",
            "cd -",
            "",
            "echo 'All done!'",
            "echo 'Job ended at: `date`'",
            "echo '------------------------------------------------------------'",
        ]
    )
    # Write to file
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(save_dir / "submit.sh", "w") as f:
        f.write(submit_script)

    if submit:
        # Submit the job
        submit_command = f"sbatch {save_dir / 'submit.sh'}"
        submit_process = subprocess.run(
            submit_command, shell=True, capture_output=True, text=True
        )
        output = submit_process.stdout.strip()
        print(f"Submitted job with command: {submit_command}")
        print(f"Output: {output}")
        # Get the job ID from the output
        match = re.search(r"Submitted batch job (\d+)", output)
        if match:
            job_id = match.group(1)
            print(f"Job ID: {job_id}")
            return job_id


# def block_until_completed(jobid_list):
#     """Block progress until all jobs in jobid_list are completed."""
#     if len(jobid_list) == 0:  # In case if there is no job submitted
#         return

#     completed = [False] * len(jobid_list)
#     while True:
#         if len(jobid_list) == 0:
#             # This is when all jobs were previously completed, so that we have no jobs to
#             # wait for anymore.
#             break

#         # Check the status of each job
#         for ii, jobid in enumerate(jobid_list):
#             if not completed[ii]:
#                 check_command = f"squeue -j {jobid}"
#                 check_process = subprocess.run(
#                     check_command, shell=True, capture_output=True, text=True
#                 )
#                 output = check_process.stdout.strip()
#                 lines = output.splitlines()
#                 if len(lines) <= 1:
#                     completed[ii] = True
#                     print(f"Job {jobid} has completed.")
#         if all(completed):
#             print("All jobs have completed.")
#             time.sleep(60)  # Wait a bit to ensure all files are written
#             break
#         else:
#             time.sleep(60)  # Wait for 60 seconds before checking again


def block_until_completed(jobid_list, poll_interval=60):
    """
    Block until all jobs in jobid_list reach a terminal Slurm state.
    """
    if not jobid_list:
        return

    # Convert all job IDs to strings
    jobid_list = list(map(str, jobid_list))
    jobid_str = ",".join(jobid_list)  # This is so that we can check all at once

    # Track completion status of each job
    completed = {jobid: False for jobid in jobid_list}

    # These are a list of terminal states in Slurm
    terminal_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
    }

    while True:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                jobid_str,
                "--format=JobIDRaw,State",
                "--parsable2",
                "--noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Track the *best* (final) state per job
        job_states = {}
        for line in result.stdout.splitlines():
            jobid_raw, state = line.split("|")[:2]
            # Ignore job steps (e.g. 12345.batch)
            if "." in jobid_raw:
                continue
            job_states[jobid_raw] = state

        for jobid in completed:
            if not completed[jobid]:
                state = job_states.get(jobid)
                # sacct can lag briefly right after submission
                if state is None:
                    continue

                if state in terminal_states:
                    completed[jobid] = True
                    print(f"Job {jobid} finished with state: {state}")

        if all(completed.values()):
            print("All jobs have completed.")
            time.sleep(60)  # ensure files are flushed
            break

        time.sleep(poll_interval)


def compute_score(
    results_dir,
    reward_sample_size,
    reward_range=[-12, 12],
    include_exact_rate=True,
    simplify_redundancy=False,
):
    """Compute the score to maximize for hyperparameter tuning based on the results
    in results_dir.

    The score is computed as the average of three components:
    1. Average mean reward over the last `nlast` episodes.
       QUESTION: How should we normalize this component?
    2. Average success rate over all episodes.
    3. Average exact success rate over all episodes, i.e., the fraction of episodes where
       the predicted circuit is exactly equivalent to the ground truth circuit.

    Parameters
    ----------
    results_dir : Path
        The directory where results are stored.
    reward_sample_size : int
        The number of last episodes to consider for mean reward calculation.
    reward_range : list, optional
        The range [min, max] of rewards for normalization, by default [-12, 12].
    include_exact_rate : bool, optional
        Whether to include the exact success rate in the score calculation,
        by default True.
    simplify_redundancy : bool, optional
        Whether to simplify circuits by removing redundant resistors before checking
        equivalence, by default False.

    Returns
    -------
    float
        The computed score for the given configuration.
    """
    # Compute the contribution from the mean reward
    results_dir = Path(results_dir)
    statistical_analysis = pd.read_pickle(results_dir / "statistical_analysis.pkl")
    episodic_cumul_rewards = statistical_analysis["episodic_cumul_reward"].values
    mean_reward_last = np.mean(episodic_cumul_rewards[-reward_sample_size:])
    # Normalize the mean reward to [0, 1]
    min_reward, max_reward = reward_range
    score_mean_reward = (mean_reward_last - min_reward) / (max_reward - min_reward)

    # Compute the contributions based on agent success
    eval_results = pd.read_pickle(results_dir / "eval_results.pkl")
    score_success_rate = eval_results["found_solution"].mean()
    # Success exact
    success_exact_count = 0
    for idx, row in eval_results.iterrows():
        found_solution = row["found_solution"]
        if found_solution:
            ground_truth = row["ground_truth_circuit"]
            circuit_string = row["best_circuit"]
            params = row["best_param"]
            if simplify_redundancy:
                simplified_circuit = simplify_circuit(circuit_string, params)
            else:
                simplified_circuit = simplify_circuit_P(circuit_string, params)
            if are_circuit_equivalent(simplified_circuit, ground_truth):
                success_exact_count += 1
    score_success_exact_rate = success_exact_count / len(eval_results)

    if include_exact_rate:
        avg_score = np.mean(
            [score_mean_reward, score_success_rate, score_success_exact_rate]
        )
    else:
        avg_score = np.mean([score_mean_reward, score_success_rate])
    return avg_score, (score_mean_reward, score_success_rate, score_success_exact_rate)


def simplify_circuit_P(circuit_string, params, verbose=False):
    """Simplify the circuit by replacing P with either R or C based on Pn value."""
    circuit_string = copy.deepcopy(circuit_string)
    # Replace P with R or C
    params_copy = {}
    for name, value in params.items():
        # Parameter name that needs attention is "PXn", where "X" indicates the index of
        # the component.
        match = re.match(r"P(\d+)n", name)
        if match:
            index = int(match.group(1))
            if value < 0.1:
                # Replace P with R
                circuit_string = re.sub(f"P{index}", f"R{index}", circuit_string)
                params_copy[f"R{index}"] = 1 / params[f"P{index}w"]
                if f"P{index}w" in params_copy:
                    del params_copy[f"P{index}w"]
                if verbose:
                    print(f"Replaced P{index} → R{index} (R{index}={value:.4f})")
            elif value > 0.9:
                # Replace P with C
                circuit_string = re.sub(f"P{index}", f"C{index}", circuit_string)
                params_copy[f"C{index}"] = params[f"P{index}w"]
                if f"P{index}w" in params_copy:
                    del params_copy[f"P{index}w"]
                if verbose:
                    print(f"Replaced P{index} → C{index} (C{index}={value:.4f})")
            else:
                params_copy[name] = value
        else:
            params_copy[name] = value
    return circuit_string, params_copy


def simplify_circuit_R(circuit_string, params, verbose=False):
    """Simplify the resistors in series or parallel."""
    circuit_string = copy.deepcopy(circuit_string)
    params = copy.deepcopy(params)
    while True:
        nomore_redundancy = True
        # R in series
        match = re.findall(r"R\d+-R\d+", circuit_string)
        if match:
            nomore_redundancy = False
            index1 = int(re.findall(r"R(\d+)-R(\d+)", circuit_string)[0][0])
            index2 = int(re.findall(r"R(\d+)-R(\d+)", circuit_string)[0][1])
            circuit_string = circuit_string.replace(match[0], f"R{index1}")
            params[f"R{index1}"] = params[f"R{index1}"] + params[f"R{index2}"]
            del params[f"R{index2}"]
            if verbose:
                print(
                    f"Combined series R{index1} + R{index2} → R{index1} "
                    f"(R{index1}={params[f'R{index1}']:.4f})"
                )
        # R in parallel
        match = re.findall(r"\[R\d+,R\d+\]", circuit_string)
        if match:
            nomore_redundancy = False
            index1 = int(re.findall(r"\[R(\d+),R(\d+)\]", circuit_string)[0][0])
            index2 = int(re.findall(r"\[R(\d+),R(\d+)\]", circuit_string)[0][1])
            circuit_string = circuit_string.replace(match[0], f"R{index1}")
            params[f"R{index1}"] = 1 / (
                1 / params[f"R{index1}"] + 1 / params[f"R{index2}"]
            )
            del params[f"R{index2}"]
            if verbose:
                print(
                    f"Combined parallel R{index1} + R{index2} → R{index1} "
                    f"(R{index1}={params[f'R{index1}']:.4f})"
                )
        if nomore_redundancy:
            break
    return circuit_string, params


def simplify_circuit_C(circuit_string, params, verbose=False):
    """Simplify the capacitors in series or parallel."""
    circuit_string = copy.deepcopy(circuit_string)
    params = copy.deepcopy(params)
    while True:
        nomore_redundancy = True
        # C in series
        match = re.findall(r"C\d+-C\d+", circuit_string)
        if match:
            nomore_redundancy = False
            index1 = int(re.findall(r"C(\d+)-C(\d+)", circuit_string)[0][0])
            index2 = int(re.findall(r"C(\d+)-C(\d+)", circuit_string)[0][1])
            circuit_string = circuit_string.replace(match[0], f"C{index1}")
            params[f"C{index1}"] = 1 / (
                1 / params[f"C{index1}"] + 1 / params[f"C{index2}"]
            )
            del params[f"C{index2}"]
            if verbose:
                print(
                    f"Combined series C{index1} + C{index2} → C{index1} "
                    f"(C{index1}={params[f'C{index1}']:.4f})"
                )
        # C in parallel
        match = re.findall(r"\[C\d+,C\d+\]", circuit_string)
        if match:
            nomore_redundancy = False
            index1 = int(re.findall(r"\[C(\d+),C(\d+)\]", circuit_string)[0][0])
            index2 = int(re.findall(r"\[C(\d+),C(\d+)\]", circuit_string)[0][1])
            circuit_string = circuit_string.replace(match[0], f"C{index1}")
            params[f"C{index1}"] = params[f"C{index1}"] + params[f"C{index2}"]
            del params[f"C{index2}"]
            if verbose:
                print(
                    f"Combined parallel C{index1} + C{index2} → C{index1} "
                    f"(C{index1}={params[f'C{index1}']:.4f})"
                )
        if nomore_redundancy:
            break
    return circuit_string, params


def simplify_circuit_L(circuit_string, params, verbose=False):
    """Simplify the inductors in series or parallel."""
    circuit_string = copy.deepcopy(circuit_string)
    params = copy.deepcopy(params)
    while True:
        nomore_redundancy = True
        # L in series
        match = re.findall(r"L\d+-L\d+", circuit_string)
        if match:
            nomore_redundancy = False
            index1 = int(re.findall(r"L(\d+)-L(\d+)", circuit_string)[0][0])
            index2 = int(re.findall(r"L(\d+)-L(\d+)", circuit_string)[0][1])
            circuit_string = circuit_string.replace(match[0], f"L{index1}")
            params[f"L{index1}"] = params[f"L{index1}"] + params[f"L{index2}"]
            del params[f"L{index2}"]
            if verbose:
                print(
                    f"Combined series L{index1} + L{index2} → L{index1} "
                    f"(L{index1}={params[f'L{index1}']:.4f})"
                )
        # L in parallel
        match = re.findall(r"\[L\d+,L\d+\]", circuit_string)
        if match:
            nomore_redundancy = False
            index1 = int(re.findall(r"\[L(\d+),L(\d+)\]", circuit_string)[0][0])
            index2 = int(re.findall(r"\[L(\d+),L(\d+)\]", circuit_string)[0][1])
            circuit_string = circuit_string.replace(match[0], f"L{index1}")
            params[f"L{index1}"] = 1 / (
                1 / params[f"L{index1}"] + 1 / params[f"L{index2}"]
            )
            del params[f"L{index2}"]
            if verbose:
                print(
                    f"Combined parallel L{index1} + L{index2} → L{index1} "
                    f"(L{index1}={params[f'L{index1}']:.4f})"
                )
        if nomore_redundancy:
            break
    return circuit_string, params


def simplify_circuit(
    circuit_string,
    params,
    simplify_elements=["R", "C", "P", "L"],
    verbose=False,
):
    """Replace P with R or C based on Pn value and remove redundant resistors."""
    circuit_string = copy.deepcopy(circuit_string)
    params = copy.deepcopy(params)
    # Replace P with R or C
    if "P" in simplify_elements:
        circuit_string, params = simplify_circuit_P(circuit_string, params, verbose)
    # Remove redundant
    while True:
        circuit_string_mod = copy.deepcopy(circuit_string)
        if "R" in simplify_elements:
            circuit_string_mod, params = simplify_circuit_R(
                circuit_string_mod, params, verbose
            )
        if "C" in simplify_elements:
            circuit_string_mod, params = simplify_circuit_C(
                circuit_string_mod, params, verbose
            )
        if "L" in simplify_elements:
            circuit_string_mod, params = simplify_circuit_L(
                circuit_string_mod, params, verbose
            )
        if circuit_string_mod == circuit_string:
            break
        else:
            circuit_string = circuit_string_mod
    return circuit_string, params


def are_circuit_equivalent(circuit1, circuit2):
    """Check if two circuit strings are equivalent using autoeis utility."""
    return ae.utils.are_circuits_equivalent(circuit1, circuit2)
