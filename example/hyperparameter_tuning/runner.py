"""Run full-budget batch hyperparameter tuning from a Slurm login node."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from pprint import pprint

# Keep the login-node orchestrator lightweight. These limits must be applied before
# importing numerical libraries, AutoEIS, or TensorFlow through tuning_utils.
THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "JULIA_NUM_THREADS",
)
for variable in THREAD_LIMIT_VARIABLES:
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis_matplotlib
import pandas as pd
import yaml

from tuning_utils import (
    ConfigurationHandler,
    block_until_completed,
    compute_score,
    write_job_script,
)


SEED = 42
WORK_DIR = Path(__file__).resolve().parent
REWARD_SAMPLE_SIZE = 1_000


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("./hyperparameter_tuning"),
        help="Directory in which configurations and results are stored.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=WORK_DIR / "base_config.yaml",
    )
    parser.add_argument(
        "--search-space",
        "-s",
        type=Path,
        default=WORK_DIR / "search_space.yaml",
    )
    parser.add_argument("--num-initial", type=_positive_integer, default=5)
    parser.add_argument("--num-iterations", type=_positive_integer, default=10)
    parser.add_argument("--batch-size", type=_positive_integer, default=5)
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_integer,
        default=1_000,
        help="Save each candidate every N trials (default: 1000).",
    )
    return parser.parse_args()


def _load_search_space(path: Path) -> tuple[dict, list[str]]:
    with path.open() as stream:
        definitions = yaml.safe_load(stream)
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError(f"Search space {path} must define a non-empty mapping.")
    for name, specification in definitions.items():
        if (
            not isinstance(specification, dict)
            or set(specification) != {"bounds", "type"}
            or not isinstance(specification["bounds"], list)
            or len(specification["bounds"]) != 2
            or str(specification["type"]).lower() not in {"int", "float"}
        ):
            raise ValueError(f"Invalid search-space entry: {name}")
    bounds = {name: specification["bounds"] for name, specification in definitions.items()}
    integers = [
        name
        for name, specification in definitions.items()
        if str(specification["type"]).lower() == "int"
    ]
    return bounds, integers


def _optuna_distributions(handler: ConfigurationHandler) -> dict:
    return {
        name: (
            optuna.distributions.IntDistribution(*bounds)
            if name in handler.integer_variables
            else optuna.distributions.FloatDistribution(*bounds)
        )
        for name, bounds in handler.search_space_bounds.items()
    }


def _candidate_complete(save_dir: Path, target_trials: int) -> bool:
    required = (
        save_dir / "checkpoints" / f"trial-{target_trials:08d}" / "manifest.json",
        save_dir / "statistical_analysis.pkl",
        save_dir / "eval_results.pkl",
        save_dir / "eval_results.csv",
    )
    return all(path.is_file() for path in required)


def _write_csv_atomic(rows: list[dict], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(destination)


def _export_study(study: optuna.Study, results_dir: Path) -> None:
    study.trials_dataframe().to_csv(results_dir / "optuna_trials.csv", index=False)
    plotters = (
        (vis_matplotlib.plot_optimization_history, "optimization_history.png"),
        (vis_matplotlib.plot_param_importances, "param_importances.png"),
    )
    for plotter, filename in plotters:
        try:
            axes = plotter(study)
        except (RuntimeError, ValueError, ZeroDivisionError) as error:
            print(f"Skipping {filename}: {error}")
            continue
        figure = axes.figure
        figure.tight_layout()
        figure.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)


def _candidate(
    handler: ConfigurationHandler,
    config: dict,
    results_dir: Path,
    trial: optuna.trial.Trial | None = None,
) -> dict:
    config_id, config_file, written_config = handler.write_config(config, results_dir)
    return {
        "trial": trial,
        "config_id": config_id,
        "config_file": config_file.resolve(),
        "save_dir": Path(written_config["agent"]["save_dir"]).resolve(),
    }


def _submit_candidates(
    candidates: list[dict],
    target_trials: int,
    checkpoint_every: int,
    python_file: Path,
    job_name_prefix: str,
    sbatch_options: dict,
) -> None:
    job_ids = []
    submitted_config_ids = set()
    for candidate in candidates:
        config_id = candidate["config_id"]
        save_dir = candidate["save_dir"]
        if config_id in submitted_config_ids or _candidate_complete(save_dir, target_trials):
            continue
        submitted_config_ids.add(config_id)
        options = dict(sbatch_options)
        options["job_name"] = f"{job_name_prefix}_{config_id}"
        job_id = write_job_script(
            save_dir,
            python_file,
            candidate["config_file"],
            options,
            script_args=("--checkpoint-every", checkpoint_every),
            submit=True,
        )
        if job_id is None:
            raise RuntimeError(f"Slurm did not return a job ID for config {config_id}.")
        job_ids.append(job_id)

    block_until_completed(job_ids)
    incomplete = [
        candidate["config_id"]
        for candidate in candidates
        if not _candidate_complete(candidate["save_dir"], target_trials)
    ]
    if incomplete:
        raise RuntimeError(f"Candidates did not produce complete results: {incomplete}")


def _score_candidate(candidate: dict):
    return compute_score(candidate["save_dir"], REWARD_SAMPLE_SIZE)


def _record_score(rows: list[dict], group: str, candidate: dict, score, components) -> None:
    trial = candidate["trial"]
    rows.append(
        {
            "group": group,
            "trial_number": 0 if trial is None else trial.number,
            "config_id": candidate["config_id"],
            "score": score,
            "mean_reward_score": components[0],
            "success_rate": components[1],
            "exact_rate": components[2],
        }
    )


def _run_baseline(
    study: optuna.Study,
    handler: ConfigurationHandler,
    distributions: dict,
    results_dir: Path,
    target_trials: int,
    checkpoint_every: int,
    python_file: Path,
    job_name_prefix: str,
    sbatch_options: dict,
    score_rows: list[dict],
) -> None:
    candidate = _candidate(
        handler, copy.deepcopy(handler.base_config), results_dir, trial=None
    )
    _submit_candidates(
        [candidate],
        target_trials,
        checkpoint_every,
        python_file,
        job_name_prefix,
        sbatch_options,
    )
    score, components = _score_candidate(candidate)
    params = handler.config_to_sample(handler.configs_list[candidate["config_id"]])
    study.add_trial(
        optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=float(score),
        )
    )
    _record_score(score_rows, "baseline", candidate, score, components)
    _write_csv_atomic(score_rows, results_dir / "candidate_scores.csv")


def _run_candidate_group(
    group_name: str,
    trials: list[optuna.trial.Trial],
    study: optuna.Study,
    handler: ConfigurationHandler,
    results_dir: Path,
    target_trials: int,
    checkpoint_every: int,
    python_file: Path,
    job_name_prefix: str,
    sbatch_options: dict,
    score_rows: list[dict],
) -> None:
    candidates = [
        _candidate(
            handler,
            handler.sample_to_config(trial.params),
            results_dir,
            trial=trial,
        )
        for trial in trials
    ]
    _submit_candidates(
        candidates,
        target_trials,
        checkpoint_every,
        python_file,
        job_name_prefix,
        sbatch_options,
    )
    for candidate in candidates:
        score, components = _score_candidate(candidate)
        study.tell(candidate["trial"], float(score))
        _record_score(score_rows, group_name, candidate, score, components)
    _write_csv_atomic(score_rows, results_dir / "candidate_scores.csv")


def main() -> None:
    args = _parse_args()
    results_dir = args.target_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    bounds, integer_variables = _load_search_space(args.search_space.resolve())
    handler = ConfigurationHandler(
        base_config=args.base_config.resolve(),
        configs_dir=results_dir / "configs",
        search_space_bounds=bounds,
        integer_variables=integer_variables,
    )
    target_trials = handler.base_config["agent"]["num_trials"]
    if isinstance(target_trials, bool) or not isinstance(target_trials, int):
        raise ValueError("agent.num_trials must be a positive integer.")
    if target_trials < 1:
        raise ValueError("agent.num_trials must be a positive integer.")

    distributions = _optuna_distributions(handler)
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.num_initial, seed=SEED)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    python_file = WORK_DIR / "main_factory.py"
    job_name_prefix = results_dir.name
    sbatch_options = {"time": "72:00:00", "mem_per_cpu": "12G"}
    score_rows = []

    print("=====================================")
    print("# FULL-BUDGET HYPERPARAMETER TUNING #")
    print("=====================================")
    print(f"Results: {results_dir}")
    print(f"Training budget per candidate: {target_trials} trials")
    print(f"Checkpoint interval: {args.checkpoint_every} trials")
    print("Run this orchestrator on the Slurm login node.")
    pprint(handler.search_space_bounds, sort_dicts=False)

    _run_baseline(
        study,
        handler,
        distributions,
        results_dir,
        target_trials,
        args.checkpoint_every,
        python_file,
        job_name_prefix,
        sbatch_options,
        score_rows,
    )
    _export_study(study, results_dir)

    group_sizes = [args.num_initial - 1] + [
        args.batch_size for _ in range(args.num_iterations - 1)
    ]
    for group_index, group_size in enumerate(group_sizes, start=1):
        if group_size == 0:
            continue
        group_name = "initial" if group_index == 1 else f"iteration_{group_index}"
        print(f"Starting {group_name} with {group_size} candidates...")
        trials = [study.ask(fixed_distributions=distributions) for _ in range(group_size)]
        _run_candidate_group(
            group_name,
            trials,
            study,
            handler,
            results_dir,
            target_trials,
            args.checkpoint_every,
            python_file,
            job_name_prefix,
            sbatch_options,
            score_rows,
        )
        _export_study(study, results_dir)

    best_trial = study.best_trial
    best_config = handler.sample_to_config(best_trial.params)
    print("Hyperparameter tuning completed.")
    print(f"Best score: {best_trial.value:.6f}")
    print(f"Best configuration ID: {handler.get_config_id(best_config)}")
    pprint(best_trial.params, sort_dicts=False)


if __name__ == "__main__":
    main()
