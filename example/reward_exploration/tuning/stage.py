"""Train or resume one full-budget reward-exploration tuning candidate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


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


def _write_evaluation_results(results, save_dir: Path) -> None:
    """Atomically publish the candidate's validation results."""
    save_dir.mkdir(parents=True, exist_ok=True)
    destinations = (
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


def _configure_deterministic_tensorflow() -> None:
    """Match the TensorFlow runtime recorded in fresh and resumed checkpoints."""
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


def run_candidate(
    config_file: Path,
    evaluation_seed: int,
    checkpoint_every: int,
    pipeline_builder_function=None,
    prepare_runtime_function=None,
) -> None:
    config = _read_config(config_file)
    target_trials = config["agent"]["num_trials"]
    save_dir = Path(config["agent"]["save_dir"])
    checkpoint_dir = save_dir / "checkpoints"
    target_checkpoint = checkpoint_dir / f"trial-{target_trials:08d}"
    result_paths = (save_dir / "eval_results.csv", save_dir / "eval_results.pkl")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)
    # Runtime compatibility is validated while discovering the latest checkpoint, so
    # determinism and thread settings must be active before recovery planning begins.
    _configure_deterministic_tensorflow()

    if prepare_runtime_function is not None:
        prepare_runtime_function()

    from autorec.agent import DDQN_ECM
    from autorec.checkpoint import plan_training_recovery
    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    if pipeline_builder_function is None:
        pipeline_builder_function = pipeline_builder

    recovery = plan_training_recovery(checkpoint_dir, target_trials)
    if not recovery.additional_trials and all(path.is_file() for path in result_paths):
        print(f"Candidate is already trained and evaluated through trial {target_trials}.")
        return

    print("#" * 55)
    print("TRAINING RL AGENT FOR GENERATING ECM FROM EIS DATA")
    print("#" * 55)
    print(f"Configuration file: {config_file}")
    print(f"Full training budget: {target_trials} trials")

    if recovery.checkpoint is None:
        set_global_seed(config["agent"].get("random_seed", 42), True)
        _, _, _, _, agent = pipeline_builder_function(config_file)
    else:
        # Do not reseed: the checkpoint loader restores its saved RNG streams last.
        print(f"Restoring checkpoint: {recovery.checkpoint}")
        agent = DDQN_ECM.from_checkpoint(
            recovery.checkpoint,
            save_dir=save_dir,
            num_trials=target_trials,
        )

    if recovery.additional_trials:
        print(f"Training {recovery.additional_trials} remaining trials...")
        agent.train(
            additional_trials=recovery.additional_trials,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
        )
    completed = plan_training_recovery(checkpoint_dir, target_trials)
    if completed.additional_trials:
        raise RuntimeError(f"Expected checkpoint was not created: {target_checkpoint}")

    # Checkpoint before evaluation so validation RNG, environment, and cache changes
    # cannot affect a later restoration of the completed candidate.
    print(f"Evaluating trial {target_trials} on the validation dataset...")
    _set_evaluation_seed(evaluation_seed)
    results = agent.eval_all_eis(max_actions=24, verbose=True, use_eval_env=True)
    _write_evaluation_results(results, save_dir)
    print(f"Completed candidate at trial {target_trials}.")


def main(
    pipeline_builder_function=None,
    prepare_runtime_function=None,
    default_config: Path | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=default_config,
        required=default_config is None,
        help="Configuration for one hyperparameter candidate.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_integer,
        default=1_000,
        help="Save a complete checkpoint every N trials (default: 1000).",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=42,
        help="Seed used only for validation evaluation.",
    )
    args = parser.parse_args()
    run_candidate(
        args.config.resolve(),
        args.evaluation_seed,
        args.checkpoint_every,
        pipeline_builder_function,
        prepare_runtime_function,
    )


if __name__ == "__main__":
    main()
