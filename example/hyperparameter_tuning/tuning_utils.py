"""Utilities for hyperparameter tuning and configuration management.

Notes
-----
The saved module list used on the cluster is assumed to be ``"default"``, and
the Python environment is assumed to be created with ``virtualenv`` and located
at ``~/autorec_env``. If your setup differs, modify ``MODULE_SAVELIST_NAME`` and
``PYTHON_ENV_PATH`` accordingly.
"""

from pathlib import Path
from glob import glob
from typing import Optional, Sequence, Tuple, Union
import copy
import json
import math
import os
import yaml
import re
import shlex
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import autoeis as ae

from autorec.parser import _simplify_P, simplify


# Variables to specify the presetup environments
MODULE_SAVELIST_NAME = "default"
PYTHON_ENV_PATH = "$HOME/autorec_env"


class SlurmJobsFailed(RuntimeError):
    """Report terminal Slurm failures while retaining every observed final result."""

    def __init__(self, message: str, results: dict[str, tuple[str, str]]):
        super().__init__(message)
        self.results = results


class SlurmJobRegistry:
    """Atomically persist submitted jobs so a restart cannot duplicate them."""

    VERSION = 1

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Slurm job registry is unreadable: {self.path}") from error
            if (
                type(payload) is not dict
                or set(payload) != {"version", "jobs"}
                or payload["version"] != self.VERSION
                or type(payload["jobs"]) is not dict
            ):
                raise ValueError(f"Slurm job registry has an unsupported schema: {self.path}")
            self.jobs = payload["jobs"]
        elif self.path.exists():
            raise ValueError(f"Slurm job registry is not a regular file: {self.path}")
        else:
            self.jobs = {}

    @staticmethod
    def key(kind: str, config_id: int, target_trials: int) -> str:
        return f"{kind}:{config_id:04d}:{target_trials:08d}"

    def current(self, kind: str, config_id: int, target_trials: int) -> dict | None:
        record = self.jobs.get(self.key(kind, config_id, target_trials))
        return None if record is None else record["attempts"][-1]

    def record_submission(
        self,
        *,
        kind: str,
        config_id: int,
        target_trials: int,
        job_id: str,
        save_dir: Union[str, Path],
        script: Union[str, Path],
        log: Union[str, Path],
    ) -> None:
        key = self.key(kind, config_id, target_trials)
        metadata = {
            "kind": kind,
            "config_id": config_id,
            "target_trials": target_trials,
            "save_dir": str(Path(save_dir)),
        }
        record = self.jobs.setdefault(key, {**metadata, "attempts": []})
        if {name: record.get(name) for name in metadata} != metadata:
            raise ValueError(f"Slurm job registry metadata conflicts for {key}.")
        record["attempts"].append(
            {
                "job_id": str(job_id),
                "state": "SUBMITTED",
                "exit_code": None,
                "script": str(Path(script)),
                "log": str(Path(log)),
            }
        )
        self._write()

    def record_results(self, results: dict[str, tuple[str, str]]) -> None:
        pending = dict(results)
        for record in self.jobs.values():
            for attempt in record["attempts"]:
                result = pending.pop(attempt["job_id"], None)
                if result is not None:
                    attempt["state"], attempt["exit_code"] = result
        if pending:
            raise ValueError(f"Slurm results contain unregistered jobs: {sorted(pending)}")
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w") as stream:
            json.dump({"version": self.VERSION, "jobs": self.jobs}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)


def cumulative_trial_targets(max_trials: int, session_trials: int) -> list[int]:
    """Return cumulative evaluation/checkpoint targets for fixed-size sessions."""
    for value, name in ((max_trials, "max_trials"), (session_trials, "session_trials")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_trials % session_trials:
        raise ValueError("max_trials must be divisible by session_trials.")
    return list(range(session_trials, max_trials + 1, session_trials))


def promotion_trial_targets(
    max_trials: int,
    grace_trials: int,
    reduction_factor: int,
) -> list[int]:
    """Return geometric cumulative budgets where synchronous pruning occurs."""
    if type(grace_trials) is not int or grace_trials < 1:
        raise ValueError("grace_trials must be a positive integer.")
    if type(reduction_factor) is not int or reduction_factor < 2:
        raise ValueError("reduction_factor must be an integer of at least 2.")
    if grace_trials >= max_trials:
        raise ValueError("grace_trials must be smaller than max_trials.")

    targets = []
    target = grace_trials
    while target < max_trials:
        targets.append(target)
        target *= reduction_factor
    return targets


def stage_output_paths(save_dir: Union[str, Path], target_trials: int) -> dict[str, Path]:
    """Return the required checkpoint and evaluation artifacts for one fidelity stage."""
    save_dir = Path(save_dir)
    stem = f"eval_results_trial-{target_trials:08d}"
    return {
        "checkpoint manifest": (
            save_dir / "checkpoints" / f"trial-{target_trials:08d}" / "manifest.json"
        ),
        "evaluation pickle": save_dir / f"{stem}.pkl",
        "evaluation CSV": save_dir / f"{stem}.csv",
    }


def missing_stage_outputs(save_dir: Union[str, Path], target_trials: int) -> list[Path]:
    """Return required stage artifacts that have not been published."""
    return [
        path
        for path in stage_output_paths(save_dir, target_trials).values()
        if not path.is_file()
    ]


def select_promoted_config_ids(scores: dict[int, float], reduction_factor: int) -> set[int]:
    """Select the highest-scoring fraction with deterministic config-ID tie breaking."""
    if not scores:
        return set()
    if type(reduction_factor) is not int or reduction_factor < 2:
        raise ValueError("reduction_factor must be an integer of at least 2.")
    if not all(np.isfinite(score) for score in scores.values()):
        raise ValueError("Promotion scores must be finite.")
    count = max(1, math.ceil(len(scores) / reduction_factor))
    ranked = sorted(scores, key=lambda config_id: (-scores[config_id], config_id))
    return set(ranked[:count])


class CustomLogScaler:
    """Log-transform and normalize continuous features.

    Integer features are left unchanged.

    Parameters
    ----------
    integer_features : list of str or str
        List of feature names that should be treated as integers.
    continuous_features : list of str or str
        List of feature names that should be treated as continuous values.
    """

    def __init__(
        self, integer_features: Union[list, str], continuous_features: Union[list, str]
    ):
        self.integer_features = integer_features
        self.continuous_features = continuous_features
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "CustomLogScaler":
        """Fit the scaler to continuous features.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing the continuous features to log-transform and
            standardize.

        Returns
        -------
        CustomLogScaler
            The fitted scaler.
        """
        # Log-transform continuous features
        log_transformed = np.log(df[self.continuous_features].values)
        # Fit internal StandardScaler
        self.scaler.fit(log_transformed)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform continuous features and leave integer features unchanged.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` with transformed continuous features.

        Raises
        ------
        ValueError
            If the scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("CustomLogScaler must be fitted before calling transform().")
        # Copy to avoid modifying original data
        df_new = df.copy()
        # Apply log + standardize to continuous features
        df_new[self.continuous_features] = self.scaler.transform(
            np.log(df[self.continuous_features].values)
        )
        # Leave integer features untouched
        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler and transform the data frame.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame to fit and transform.

        Returns
        -------
        pandas.DataFrame
            A transformed copy of ``df``.
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """Undo the scaling and log transformation.

        Parameters
        ----------
        df_scaled : pandas.DataFrame
            Data frame containing scaled continuous features.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df_scaled`` with continuous features restored to their
            original scale.
        """
        df = df_scaled.copy()
        # Undo standardization in log space
        log_vals = self.scaler.inverse_transform(df[self.continuous_features].values)
        # Undo log transformation
        df[self.continuous_features] = np.exp(log_vals)
        # Integers remain unchanged
        return df


class ConfigurationHandler:
    """Handle configuration conversion, writing, comparison, and storage.

    Parameters
    ----------
    base_config : dict or str or Path
        The base configuration dictionary or the path to a YAML file containing the base
        configuration.
    configs_dir : str or Path
        The directory where configuration files will be stored.
    search_space_bounds : dict
        A dictionary specifying the bounds of the search space for each hyperparameter.
    integer_variables : list of str
        A list of hyperparameter names that should be treated as integers.
    basename : str, optional
        The base name for configuration files, by default "config". Configuration files will
        be named as ``"{basename}_{config_id:04d}.yaml"``.
    """

    def __init__(
        self,
        base_config: Union[dict, str, Path],
        configs_dir: Union[str, Path],
        search_space_bounds: dict,
        integer_variables: list,
        basename: str = "config",
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

    def _load_existing_configs(self) -> Tuple[dict, list]:
        """Load existing configurations from the configs directory.

        Returns
        -------
        existing_configs : dict
            Dictionary mapping configuration IDs to configuration dictionaries.
        existing_configs_ids : list of int
            Existing configuration IDs.
        """
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

    def get_config_id(self, config: dict) -> int:
        """Get a unique configuration ID for a new configuration.

        Parameters
        ----------
        config : dict
            The configuration dictionary for which to get the ID.

        Returns
        -------
        config_id : int
            The unique ID assigned to the configuration. If the configuration already exists,
            the existing ID will be returned. Otherwise, a new ID will be generated.
        """
        # Check if the configuration already exists
        for existing_id, existing_config in self.configs_list.items():
            if self.configs_equal(config, existing_config):
                return existing_id
        # If not, assign a new ID
        if self.missing_config_ids:
            return self.missing_config_ids.pop(0)
        else:
            return max(self.configs_ids) + 1 if self.configs_ids else 0

    def configs_equal(self, config1: dict, config2: dict) -> bool:
        """Check whether two configurations are equal.

        The comparison ignores non-essential fields such as ``save_dir``.

        Parameters
        ----------
        config1 : dict
            First configuration dictionary.
        config2 : dict
            Second configuration dictionary.

        Returns
        -------
        bool
            Whether the two configurations are equal after ignored fields are
            removed.
        """
        config1_copy = copy.deepcopy(config1)
        config2_copy = copy.deepcopy(config2)
        # Ignore save_dir when comparing
        config1_copy["agent"]["save_dir"] = None
        config2_copy["agent"]["save_dir"] = None
        return config1_copy == config2_copy

    def sample_to_config(self, sample: dict) -> dict:
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

    def config_to_sample(self, config: dict) -> dict:
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
                    # No scaling, just convert to int
                    value = int(config["agent"][name])
            else:
                value = config["agent"][name]
            sample[name] = value
        return sample

    def add_config(self, config: dict, config_id: Optional[int] = None) -> int:
        """Add a configuration to the internal storage without writing to file.

        Parameters
        ----------
        config : dict
            The configuration dictionary to add.
        config_id : int, optional
            The unique ID to assign to the configuration. If ``None``, a new ID will be
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

    def remove_config(self, item: Union[int, dict]) -> None:
        """Remove a configuration from the internal storage.

        Parameters
        ----------
        item : int or dict
            The configuration ID or configuration dictionary to remove.

        Raises
        ------
        ValueError
            If ``item`` is neither an integer nor a dictionary.
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

    def write_config(
        self, config: dict, results_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[int, Path, dict]:
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
        config_filename : pathlib.Path
            Path to the configuration file.
        config : dict
            The configuration dictionary that was written or retrieved.
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


def write_job_script(
    save_dir: Union[str, Path],
    python_file: Union[str, Path],
    config_file: Union[str, Path],
    sbatch_options: dict,
    script_args: Optional[Sequence[Union[str, int, Path]]] = None,
    submit: bool = False,
) -> Optional[str]:
    """Write a Slurm job submission script with the specified options and parameters.

    Parameters
    ----------
    save_dir : str or Path
        The directory where the job script and output logs will be saved.
    python_file : str or Path
        The path to the Python script that will be executed in the job.
    config_file : str or Path
        The path to the configuration file that will be passed as an argument to the Python
        script.
    sbatch_options : dict
        A dictionary containing Slurm options such as time, cpus_per_task, nodes,
        mem_per_cpu, and job_name.
    script_args : sequence, optional
        Additional command-line arguments passed to the training script after ``--config``.
    submit : bool, optional
        Whether to submit the job immediately after writing the script, by default False.

    Returns
    -------
    job_id : str or None
        The Slurm job ID if the job was submitted, otherwise None.
    """
    save_dir = Path(save_dir)
    python_file = Path(python_file).resolve()
    time = sbatch_options.get("time", "2:00:00")
    cpus_per_task = sbatch_options.get("cpus_per_task", 2)
    nodes = sbatch_options.get("nodes", 1)
    mem_per_cpu = sbatch_options.get("mem_per_cpu", "4G")
    job_name = sbatch_options.get("job_name", "hyperparameter_tuning")
    arguments = ["--config", str(config_file), *(script_args or ())]
    argument_array = " ".join(shlex.quote(str(value)) for value in arguments)

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
            f"#SBATCH --output={save_dir}/%x_slurm-%A.out   # output and error log",
            "",
            "set -euo pipefail",
            "",
            "echo '------------------------------------------------------------'",
            "echo 'Running job $SLURM_JOB_NAME'",
            "echo 'Allocated node: `hostname`'",
            "echo 'Job started at: `date`'",
            "echo '------------------------------------------------------------'",
            "",
            "# Print all SBATCH parameters for this job",
            "echo 'Printing SBATCH parameters...'",
            "scontrol show job $SLURM_JOB_ID",
            "echo '------------------------------------------------------------'",
            "",
            "# DEFINE YOUR ENVIRONMENT VARIABLES HERE",
            f"MODULE_SAVELIST_NAME='{MODULE_SAVELIST_NAME}'",
            f"PYTHON_ENV_PATH={PYTHON_ENV_PATH}",
            "",
            "# Load necessary modules and activate environment",
            "export CUDA_VISIBLE_DEVICES=-1",
            "export PYTHONHASHSEED=0",
            "export TF_NUM_INTRAOP_THREADS=1",
            "export TF_NUM_INTEROP_THREADS=1",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export MPLBACKEND=Agg",
            "module restore ${MODULE_SAVELIST_NAME}",
            "source ${PYTHON_ENV_PATH}/bin/activate",
            "",
            "# RUN YOUR PYTHON SCRIPT",
            "# DEFINE YOUR PYTHON SCRIPT PARAMETERS HERE",
            f"WORK_DIR={shlex.quote(str(python_file.parent))}",
            f"SCRIPTS={shlex.quote(str(python_file))}",
            f"ARGS=({argument_array})",
            "",
            "# Run calculation",
            'cd "$WORK_DIR"',
            "echo 'Running script: ${SCRIPTS}'",
            'echo "Arguments: ${ARGS[*]}"',
            "",
            'time python "$SCRIPTS" "${ARGS[@]}"',
            "",
            "echo 'All done!'",
            "echo 'Job ended at: `date`'",
            "echo '------------------------------------------------------------'",
        ]
    )
    # Write to file
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "submit.sh", "w") as f:
        f.write(submit_script)

    if submit:
        return _submit_script(save_dir / "submit.sh")


def write_archive_job_script(
    save_dir: Union[str, Path],
    target_trials: int,
    sbatch_options: dict,
    submit: bool = False,
) -> Optional[str]:
    """Write a restart-safe Slurm job that archives terminal run outputs."""
    save_dir = Path(save_dir).resolve()
    time_limit = sbatch_options.get("time", "2:00:00")
    cpus_per_task = sbatch_options.get("cpus_per_task", 1)
    nodes = sbatch_options.get("nodes", 1)
    mem_per_cpu = sbatch_options.get("mem_per_cpu", "1G")
    job_name = sbatch_options.get("job_name", "archive_hyperparameter_tuning")
    quoted_save_dir = shlex.quote(str(save_dir))

    archive_script = "\n".join(
        [
            "#!/bin/bash",
            "",
            f"#SBATCH --time={time_limit}   # walltime",
            f"#SBATCH --cpus-per-task={cpus_per_task}   # number of processor cores",
            f"#SBATCH --nodes={nodes}   # number of nodes",
            f"#SBATCH --mem-per-cpu={mem_per_cpu}   # memory per CPU core",
            f"#SBATCH --job-name='{job_name}'   # job name",
            "#SBATCH --account=rrg-j3goals",
            "#SBATCH --constraint='turin'",
            f"#SBATCH --output={save_dir}/%x_slurm-%A.out   # output and error log",
            "",
            "set -euo pipefail",
            f"SAVE_DIR={quoted_save_dir}",
            f"TARGET_TRIALS={target_trials}",
            'COMPLETE_MARKER="$SAVE_DIR/.archive_complete"',
            'IN_PROGRESS_MARKER="$SAVE_DIR/.archive_in_progress"',
            'MARKER_TEMP="$SAVE_DIR/.archive_complete.tmp"',
            "",
            'if [[ -f "$COMPLETE_MARKER" ]]; then',
            '    if [[ "$(cat "$COMPLETE_MARKER")" != "$TARGET_TRIALS" ]]; then',
            '        echo "Archive marker target does not match $TARGET_TRIALS." >&2',
            "        exit 1",
            "    fi",
            '    echo "Run outputs are already archived."',
            "    exit 0",
            "fi",
            "",
            'if [[ ! -f "$IN_PROGRESS_MARKER" ]]; then',
            "    for name in models fit_plots; do",
            '        if [[ -d "$SAVE_DIR/$name" && -e "$SAVE_DIR/$name.tar.gz" ]]; then',
            '            echo "Refusing ambiguous archive state for $name." >&2',
            "            exit 1",
            "        fi",
            "    done",
            '    printf "%s\\n" "$TARGET_TRIALS" > "$IN_PROGRESS_MARKER.tmp"',
            '    mv "$IN_PROGRESS_MARKER.tmp" "$IN_PROGRESS_MARKER"',
            "fi",
            "",
            "cleanup() {",
            '    rm -f "$SAVE_DIR/.models.tar.gz.tmp"',
            '    rm -f "$SAVE_DIR/.fit_plots.tar.gz.tmp" "$MARKER_TEMP"',
            "}",
            "trap cleanup EXIT",
            "",
            "archive_directory() {",
            '    local name="$1"',
            '    local source="$SAVE_DIR/$name"',
            '    local archive="$SAVE_DIR/$name.tar.gz"',
            '    local temporary="$SAVE_DIR/.$name.tar.gz.tmp"',
            "",
            '    if [[ -f "$archive" ]]; then',
            '        tar -tzf "$archive" >/dev/null',
            '        if [[ -d "$source" ]]; then',
            '            rm -rf "$source"',
            "        fi",
            '    elif [[ -d "$source" ]]; then',
            '        tar -czf "$temporary" -C "$SAVE_DIR" "$name"',
            '        tar -tzf "$temporary" >/dev/null',
            '        mv "$temporary" "$archive"',
            '        rm -rf "$source"',
            '    elif [[ -e "$source" || -e "$archive" ]]; then',
            '        echo "Unsupported archive input type for $name." >&2',
            "        exit 1",
            "    else",
            '        echo "No $name directory to archive."',
            "    fi",
            "}",
            "",
            "archive_directory models",
            "archive_directory fit_plots",
            'printf "%s\\n" "$TARGET_TRIALS" > "$MARKER_TEMP"',
            'mv "$MARKER_TEMP" "$COMPLETE_MARKER"',
            'rm -f "$IN_PROGRESS_MARKER"',
            'echo "Archived terminal outputs for trial $TARGET_TRIALS."',
        ]
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    script_path = save_dir / "archive.sh"
    script_path.write_text(archive_script)
    if submit:
        return _submit_script(script_path)


def _submit_script(script_path: Path) -> str:
    """Submit one generated Slurm script and return its numeric job ID."""
    submit_command = ["sbatch", str(script_path.resolve())]
    submit_process = subprocess.run(
        submit_command, capture_output=True, text=True, check=False
    )
    output = submit_process.stdout.strip()
    print(f"Submitted job with command: {shlex.join(submit_command)}")
    print(f"Output: {output}")
    if submit_process.returncode:
        raise RuntimeError(f"Slurm submission failed: {submit_process.stderr.strip()}")
    match = re.search(r"Submitted batch job (\d+)", output)
    if match:
        job_id = match.group(1)
        print(f"Job ID: {job_id}")
        return job_id
    raise RuntimeError(f"Could not parse a Slurm job ID from: {output!r}")


def block_until_completed(
    jobid_list: list,
    poll_interval: int = 60,
    accounting_timeout: int = 3_600,
) -> dict[str, tuple[str, str]]:
    """Block until all jobs reach a terminal Slurm state.

    Parameters
    ----------
    jobid_list : list of str
        A list of Slurm job IDs to monitor.
    poll_interval : int, optional
        The interval (in seconds) at which to poll the job status, by default 60 seconds.
    accounting_timeout : int, optional
        Maximum continuous ``sacct`` outage tolerated before failing, by default one hour.

    Returns
    -------
    dict
        Final ``(state, exit_code)`` pair for every requested job ID.
    """
    if not jobid_list:
        return {}
    if type(poll_interval) is not int or poll_interval < 1:
        raise ValueError("poll_interval must be a positive integer.")
    if type(accounting_timeout) is not int or accounting_timeout < 1:
        raise ValueError("accounting_timeout must be a positive integer.")

    # Convert all job IDs to strings
    jobid_list = list(map(str, jobid_list))
    jobid_str = ",".join(jobid_list)  # This is so that we can check all at once

    terminal_results = {}
    accounting_failure_started = None
    missing_since = {}

    # These are a list of terminal states in Slurm
    terminal_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "DEADLINE",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "BOOT_FAIL",
    }

    while True:
        query_command = [
            "sacct",
            "-j",
            jobid_str,
            "--format=JobIDRaw,State,ExitCode",
            "--parsable2",
            "--noheader",
        ]
        result = subprocess.run(
            query_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            now = time.monotonic()
            if accounting_failure_started is None:
                accounting_failure_started = now
            elapsed = now - accounting_failure_started
            remaining = [jobid for jobid in jobid_list if jobid not in terminal_results]
            queue_result = subprocess.run(
                [
                    "squeue",
                    "--jobs",
                    ",".join(remaining),
                    "--noheader",
                    "--format=%A|%T",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            active_text = (
                queue_result.stdout.strip()
                if queue_result.returncode == 0 and queue_result.stdout.strip()
                else "active state unavailable"
            )
            error = result.stderr.strip() or result.stdout.strip() or "unknown sacct error"
            print(
                f"Slurm accounting is unavailable ({elapsed:.0f}s): {error}. "
                f"squeue: {active_text}"
            )
            if elapsed >= accounting_timeout:
                ids = ", ".join(remaining)
                raise RuntimeError(
                    f"Slurm accounting remained unavailable for {elapsed:.0f}s; "
                    f"job status is unknown for: {ids}. Last sacct error: {error}"
                )
            time.sleep(poll_interval)
            continue
        accounting_failure_started = None

        # Track the *best* (final) state per job
        observed = set()
        for line in result.stdout.splitlines():
            columns = line.split("|")
            if len(columns) < 3:
                continue
            jobid_raw, state, exit_code = columns[:3]
            # Ignore job steps (e.g. 12345.batch)
            if "." in jobid_raw or jobid_raw not in jobid_list:
                continue
            observed.add(jobid_raw)
            state = state.split()[0].rstrip("+")
            if state in terminal_states and jobid_raw not in terminal_results:
                terminal_results[jobid_raw] = (state, exit_code)
                print(f"Job {jobid_raw} finished with state: {state}, exit code: {exit_code}")

        now = time.monotonic()
        remaining = [jobid for jobid in jobid_list if jobid not in terminal_results]
        for jobid in remaining:
            if jobid in observed:
                missing_since.pop(jobid, None)
            else:
                missing_since.setdefault(jobid, now)
        expired = [
            jobid
            for jobid, started in missing_since.items()
            if now - started >= accounting_timeout
        ]
        if expired:
            raise RuntimeError(
                "Slurm accounting returned no record within the status timeout for: "
                + ", ".join(expired)
            )

        if len(terminal_results) == len(jobid_list):
            print("All jobs have completed.")
            failed = {
                jobid: result
                for jobid, result in terminal_results.items()
                if result != ("COMPLETED", "0:0")
            }
            if failed:
                details = ", ".join(
                    f"{jobid}={state}/{exit_code}"
                    for jobid, (state, exit_code) in failed.items()
                )
                raise SlurmJobsFailed(
                    f"One or more Slurm jobs failed: {details}", terminal_results
                )
            return terminal_results

        time.sleep(poll_interval)


def compute_score(
    results_dir: Union[str, Path],
    reward_sample_size: int,
    reward_range: list = [-12, 12],
    include_exact_rate: bool = True,
    simplify_redundancy: bool = False,
    evaluation_results: Optional[Union[str, Path]] = None,
) -> Tuple[float, Tuple[float, float, float]]:
    """Compute the score to maximize for hyperparameter tuning.

    The score is computed as the average of three components:
    1. Average mean reward over the last ``reward_sample_size`` episodes.
    2. Average success rate over all episodes.
    3. Average exact success rate over all episodes, i.e., the fraction of episodes where
       the predicted circuit is exactly equivalent to the ground truth circuit.

    Parameters
    ----------
    results_dir : str or Path
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
    evaluation_results : str or Path, optional
        Rung-specific evaluation pickle. Defaults to ``eval_results.pkl`` in
        ``results_dir`` for backward compatibility.

    Returns
    -------
    avg_score : float
        The computed average score for the given configuration.
    score_components : tuple of float
        The mean reward score, success rate score, and exact success rate score.
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
    evaluation_results = (
        results_dir / "eval_results.pkl"
        if evaluation_results is None
        else Path(evaluation_results)
    )
    eval_results = pd.read_pickle(evaluation_results)
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
                simplified_circuit = simplify(circuit_string, params)[0]
            else:
                simplified_circuit = _simplify_P(circuit_string, params)[0]
            if are_circuit_equivalent(simplified_circuit, ground_truth):
                success_exact_count += 1
    score_success_exact_rate = success_exact_count / len(eval_results)

    if include_exact_rate:
        avg_score = np.mean([score_mean_reward, score_success_rate, score_success_exact_rate])
    else:
        avg_score = np.mean([score_mean_reward, score_success_rate])
    return avg_score, (score_mean_reward, score_success_rate, score_success_exact_rate)


def are_circuit_equivalent(circuit1: str, circuit2: str) -> bool:
    """Check whether two circuit strings are equivalent.

    Parameters
    ----------
    circuit1 : str
        First circuit string.
    circuit2 : str
        Second circuit string.

    Returns
    -------
    bool
        Whether the circuits are equivalent according to the ``autoeis`` utility.
    """
    return ae.utils.are_circuits_equivalent(circuit1, circuit2)
