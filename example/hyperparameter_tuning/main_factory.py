"""Train or resume one cumulative hyperparameter-tuning fidelity stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


DIRECTORY = Path(__file__).resolve().parent


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _read_config(config_file: Path) -> dict:
    with config_file.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("agent"), dict):
        raise ValueError("Configuration must contain an 'agent' mapping.")
    num_trials = config["agent"].get("num_trials")
    if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError("agent.num_trials must be a positive integer.")
    return config


def _checkpoint_trial(path: Path) -> int | None:
    prefix = "trial-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _latest_checkpoint(checkpoint_dir: Path, target_trials: int) -> Path | None:
    candidates = [
        (trial, path)
        for path in checkpoint_dir.glob("trial-*")
        if (trial := _checkpoint_trial(path)) is not None and trial <= target_trials
    ]
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _evaluation_paths(save_dir: Path, target_trials: int) -> tuple[Path, Path]:
    stem = f"eval_results_trial-{target_trials:08d}"
    return save_dir / f"{stem}.csv", save_dir / f"{stem}.pkl"


def _write_evaluation_results(results, save_dir: Path, target_trials: int) -> None:
    """Atomically publish rung-specific results and refresh legacy latest-result paths."""
    save_dir.mkdir(parents=True, exist_ok=True)
    rung_csv, rung_pickle = _evaluation_paths(save_dir, target_trials)
    destinations = (
        (rung_csv, "csv"),
        (rung_pickle, "pickle"),
        (save_dir / "eval_results.csv", "csv"),
        (save_dir / "eval_results.pkl", "pickle"),
    )
    for destination, file_type in destinations:
        temporary = destination.with_name(f".{destination.name}.tmp")
        if file_type == "csv":
            results.to_csv(temporary, index=False)
        else:
            results.to_pickle(temporary)
        temporary.replace(destination)


def _configure_checkpoint_tensorflow() -> None:
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def _set_evaluation_seed(seed: int) -> None:
    """Reset evaluation RNGs without reconfiguring an initialized TensorFlow runtime."""
    import random

    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_stage(config_file: Path, target_trials: int | None, evaluation_seed: int) -> None:
    config = _read_config(config_file)
    session_trials = config["agent"]["num_trials"]
    target_trials = session_trials if target_trials is None else target_trials
    save_dir = Path(config["agent"]["save_dir"])
    checkpoint_dir = save_dir / "checkpoints"
    target_checkpoint = checkpoint_dir / f"trial-{target_trials:08d}"
    rung_csv, rung_pickle = _evaluation_paths(save_dir, target_trials)

    if target_checkpoint.is_dir() and rung_csv.is_file() and rung_pickle.is_file():
        print(f"Cumulative trial {target_trials} is already trained and evaluated.")
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    from autorec.agent import DDQN_ECM
    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    previous_checkpoint = _latest_checkpoint(checkpoint_dir, target_trials)
    previous_trials = (
        0 if previous_checkpoint is None else int(_checkpoint_trial(previous_checkpoint))
    )
    additional_trials = target_trials - previous_trials

    print("#" * 55)
    print("TRAINING RL AGENT FOR GENERATING ECM FROM EIS DATA")
    print("#" * 55)
    print(f"Configuration file: {config_file}")
    print(f"Cumulative target: {target_trials} trials")

    if previous_checkpoint is None:
        set_global_seed(config["agent"].get("random_seed", 42), True)
        _, _, _, _, agent = pipeline_builder(config_file)
    else:
        # Match the initial session's deterministic runtime without reseeding. The
        # checkpoint loader restores its saved RNG streams last.
        _configure_checkpoint_tensorflow()
        print(f"Restoring checkpoint: {previous_checkpoint}")
        agent = DDQN_ECM.from_checkpoint(
            previous_checkpoint,
            save_dir=save_dir,
            num_trials=session_trials,
        )

    if additional_trials:
        print(f"Training {additional_trials} additional trials...")
        agent.train(additional_trials=additional_trials, checkpoint_dir=checkpoint_dir)
    if not target_checkpoint.is_dir():
        raise RuntimeError(f"Expected checkpoint was not created: {target_checkpoint}")

    # Save before evaluation. The next stage restores this immutable checkpoint, so
    # validation RNG, environment, and cache changes cannot affect continued training.
    print(f"Evaluating cumulative trial {target_trials} on the validation dataset...")
    _set_evaluation_seed(evaluation_seed)
    results = agent.eval_all_eis(max_actions=24, verbose=True, use_eval_env=True)
    _write_evaluation_results(results, save_dir, target_trials)
    print(f"Completed cumulative trial {target_trials}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DIRECTORY / "base_config.yaml",
        help="Configuration for one hyperparameter candidate.",
    )
    parser.add_argument(
        "--target-trials",
        type=_positive_integer,
        default=None,
        help="Cumulative trial target; defaults to agent.num_trials.",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=42,
        help="Seed used only for validation evaluation.",
    )
    args = parser.parse_args()
    run_stage(args.config.resolve(), args.target_trials, args.evaluation_seed)


if __name__ == "__main__":
    main()
