"""Run synchronous multi-fidelity hyperparameter tuning on a Slurm cluster."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from pprint import pprint

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
    cumulative_trial_targets,
    promotion_trial_targets,
    select_promoted_config_ids,
    write_job_script,
)


SEED = 42
WORK_DIR = Path(__file__).resolve().parent


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
        default=Path("./hyperparameter_tuning_results"),
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
    parser.add_argument("--num-initial", type=_positive_integer, default=10)
    parser.add_argument("--num-iterations", type=_positive_integer, default=5)
    parser.add_argument("--batch-size", type=_positive_integer, default=5)
    parser.add_argument(
        "--max-trials",
        type=_positive_integer,
        default=10_000,
        help="Maximum cumulative training budget per promoted candidate.",
    )
    parser.add_argument(
        "--session-trials",
        type=_positive_integer,
        default=1_000,
        help="Trials added before each checkpoint and validation score.",
    )
    parser.add_argument(
        "--grace-trials",
        type=_positive_integer,
        default=2_000,
        help="First cumulative trial count at which candidates may be pruned.",
    )
    parser.add_argument(
        "--reduction-factor",
        type=_positive_integer,
        default=2,
        help="Keep the best ceil(1 / factor) candidates at each promotion rung.",
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


def _stage_paths(save_dir: Path, target_trials: int) -> tuple[Path, Path]:
    checkpoint = save_dir / "checkpoints" / f"trial-{target_trials:08d}"
    evaluation = save_dir / f"eval_results_trial-{target_trials:08d}.pkl"
    return checkpoint, evaluation


def _stage_complete(save_dir: Path, target_trials: int) -> bool:
    checkpoint, evaluation = _stage_paths(save_dir, target_trials)
    return (
        (checkpoint / "manifest.json").is_file()
        and evaluation.is_file()
        and evaluation.with_suffix(".csv").is_file()
    )


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


def _submit_stage(
    candidates: list[dict],
    target_trials: int,
    max_trials: int,
    python_file: Path,
    job_name_prefix: str,
    sbatch_options: dict,
) -> None:
    job_ids = []
    submitted_config_ids = set()
    for candidate in candidates:
        config_id = candidate["config_id"]
        save_dir = candidate["save_dir"]
        if config_id in submitted_config_ids or _stage_complete(save_dir, target_trials):
            continue
        submitted_config_ids.add(config_id)
        options = dict(sbatch_options)
        options["job_name"] = f"{job_name_prefix}_{config_id}_{target_trials}"
        job_id = write_job_script(
            save_dir,
            python_file,
            candidate["config_file"],
            options,
            script_args=("--target-trials", target_trials),
            archive_models=target_trials == max_trials,
            submit=True,
        )
        if job_id is None:
            raise RuntimeError(f"Slurm did not return a job ID for config {config_id}.")
        job_ids.append(job_id)
    block_until_completed(job_ids)


def _score_candidate(candidate: dict, target_trials: int, reward_sample_size: int):
    _, evaluation = _stage_paths(candidate["save_dir"], target_trials)
    return compute_score(
        candidate["save_dir"],
        reward_sample_size,
        evaluation_results=evaluation,
    )


def _run_baseline(
    study: optuna.Study,
    handler: ConfigurationHandler,
    distributions: dict,
    results_dir: Path,
    targets: list[int],
    python_file: Path,
    job_name_prefix: str,
    sbatch_options: dict,
    reward_sample_size: int,
    rung_rows: list[dict],
) -> None:
    config_id, config_file, config = handler.write_config(
        copy.deepcopy(handler.base_config), results_dir
    )
    candidate = {
        "config_id": config_id,
        "config_file": config_file.resolve(),
        "save_dir": Path(config["agent"]["save_dir"]).resolve(),
    }
    intermediate_values = {}
    final_score = None
    for target in targets:
        _submit_stage(
            [candidate],
            target,
            targets[-1],
            python_file,
            job_name_prefix,
            sbatch_options,
        )
        final_score, components = _score_candidate(candidate, target, reward_sample_size)
        intermediate_values[target] = float(final_score)
        rung_rows.append(
            {
                "group": "baseline",
                "trial_number": 0,
                "config_id": config_id,
                "target_trials": target,
                "score": final_score,
                "mean_reward_score": components[0],
                "success_rate": components[1],
                "exact_rate": components[2],
                "status": "complete" if target == targets[-1] else "continued",
            }
        )
        _write_csv_atomic(rung_rows, results_dir / "rung_scores.csv")
    params = handler.config_to_sample(config)
    study.add_trial(
        optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=float(final_score),
            intermediate_values=intermediate_values,
        )
    )


def _run_candidate_group(
    group_name: str,
    trials: list[optuna.trial.Trial],
    study: optuna.Study,
    handler: ConfigurationHandler,
    results_dir: Path,
    targets: list[int],
    promotion_targets: set[int],
    reduction_factor: int,
    python_file: Path,
    job_name_prefix: str,
    sbatch_options: dict,
    reward_sample_size: int,
    rung_rows: list[dict],
) -> None:
    candidates = []
    for trial in trials:
        config_id, config_file, config = handler.write_config(
            handler.sample_to_config(trial.params), results_dir
        )
        candidates.append(
            {
                "trial": trial,
                "config_id": config_id,
                "config_file": config_file.resolve(),
                "save_dir": Path(config["agent"]["save_dir"]).resolve(),
            }
        )

    active = candidates
    for target in targets:
        _submit_stage(
            active,
            target,
            targets[-1],
            python_file,
            job_name_prefix,
            sbatch_options,
        )
        scores = {}
        components_by_trial = {}
        for candidate in active:
            score, components = _score_candidate(candidate, target, reward_sample_size)
            trial = candidate["trial"]
            trial.report(float(score), step=target)
            scores[trial.number] = float(score)
            components_by_trial[trial.number] = components

        promoted = set(scores)
        if target in promotion_targets:
            promoted = select_promoted_config_ids(scores, reduction_factor)

        next_active = []
        for candidate in active:
            trial = candidate["trial"]
            components = components_by_trial[trial.number]
            if target == targets[-1]:
                status = "complete"
                study.tell(trial, scores[trial.number])
            elif trial.number in promoted:
                status = "promoted" if target in promotion_targets else "continued"
                next_active.append(candidate)
            else:
                status = "pruned"
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rung_rows.append(
                {
                    "group": group_name,
                    "trial_number": trial.number,
                    "config_id": candidate["config_id"],
                    "target_trials": target,
                    "score": scores[trial.number],
                    "mean_reward_score": components[0],
                    "success_rate": components[1],
                    "exact_rate": components[2],
                    "status": status,
                }
            )
        _write_csv_atomic(rung_rows, results_dir / "rung_scores.csv")
        active = next_active


def main() -> None:
    args = _parse_args()
    if args.reduction_factor < 2:
        raise ValueError("--reduction-factor must be at least 2.")
    targets = cumulative_trial_targets(args.max_trials, args.session_trials)
    promotion_targets = promotion_trial_targets(
        args.max_trials, args.grace_trials, args.reduction_factor
    )
    if not set(promotion_targets).issubset(targets):
        raise ValueError("Every promotion budget must align with a session boundary.")

    results_dir = args.target_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    bounds, integer_variables = _load_search_space(args.search_space.resolve())
    handler = ConfigurationHandler(
        base_config=args.base_config.resolve(),
        configs_dir=results_dir / "configs",
        search_space_bounds=bounds,
        integer_variables=integer_variables,
    )
    if handler.base_config["agent"]["num_trials"] != args.session_trials:
        raise ValueError("agent.num_trials must equal --session-trials.")

    distributions = _optuna_distributions(handler)
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.num_initial, seed=SEED)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    python_file = WORK_DIR / "main_factory.py"
    job_name_prefix = results_dir.name
    sbatch_options = {"time": "72:00:00", "mem_per_cpu": "12G"}
    reward_sample_size = args.session_trials
    rung_rows = []

    print("============================================")
    print("# MULTI-FIDELITY HYPERPARAMETER TUNING #")
    print("============================================")
    print(f"Results: {results_dir}")
    print(f"Cumulative targets: {targets}")
    print(f"Promotion targets: {promotion_targets}")
    print("The baseline is protected from pruning and runs to the maximum budget.")
    pprint(handler.search_space_bounds, sort_dicts=False)

    _run_baseline(
        study,
        handler,
        distributions,
        results_dir,
        targets,
        python_file,
        job_name_prefix,
        sbatch_options,
        reward_sample_size,
        rung_rows,
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
            targets,
            set(promotion_targets),
            args.reduction_factor,
            python_file,
            job_name_prefix,
            sbatch_options,
            reward_sample_size,
            rung_rows,
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
