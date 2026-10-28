"""Restore one final checkpoint and evaluate fixed samples from its training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import random

from checkpoint_runtime import (
    configure_cache_environment,
    configure_tensorflow_for_checkpoint,
)


def _variable_records(variables):
    import numpy as np

    return [
        {
            "name": variable.name,
            "trainable": bool(variable.trainable),
            "value": np.array(variable.numpy(), copy=True),
        }
        for variable in variables
    ]


def _assert_variables_unchanged(before, variables, label):
    after = _variable_records(variables)
    if len(before) != len(after):
        raise RuntimeError(f"{label} variable count changed during evaluation.")
    for old, new in zip(before, after):
        if (
            old["name"] != new["name"]
            or old["trainable"] != new["trainable"]
            or old["value"].dtype != new["value"].dtype
            or old["value"].shape != new["value"].shape
            or old["value"].tobytes() != new["value"].tobytes()
        ):
            raise RuntimeError(f"{label} variables changed during evaluation.")


def _parse_indices(raw_indices):
    if raw_indices is None:
        return None
    indices = [int(value.strip()) for value in raw_indices.split(",") if value.strip()]
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("--indices must contain unique, nonnegative row positions.")
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", help="Comma-separated training dataset row positions.")
    selection.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--max-actions", type=int)
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_actions is not None and args.max_actions < 1:
        parser.error("--max-actions must be positive")
    try:
        requested_indices = _parse_indices(args.indices)
    except ValueError as error:
        parser.error(str(error))

    configure_cache_environment()

    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    import tensorflow as tf
    import pandas as pd

    from autorec.agent import DDQN_ECM

    configure_tensorflow_for_checkpoint(tf)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = DDQN_ECM.from_checkpoint(args.checkpoint.resolve(), save_dir=output_dir)

    dataset_size = len(agent.training_env.dataset)
    if requested_indices is None:
        sample_size = min(args.num_samples, dataset_size)
        indices = sorted(
            random.Random(args.sample_seed).sample(range(dataset_size), sample_size)
        )
    else:
        indices = requested_indices
    if max(indices) >= dataset_size:
        raise IndexError(
            f"Evaluation index {max(indices)} is outside the {dataset_size}-row training dataset."
        )

    online_variables = _variable_records(agent.model.variables)
    target_variables = _variable_records(agent.target_model.variables)
    optimizer_variables = _variable_records(agent.optimizer.variables)
    max_actions = agent.action_cap if args.max_actions is None else args.max_actions

    # Call the single-EIS API so failures are not converted to placeholder rows and the
    # comparison does not spend time generating plots or GIFs.
    results = pd.DataFrame(
        [
            agent.generate_ECM(
                EIS_i=index,
                max_actions=max_actions,
                verbose=False,
                get_fit_plots=False,
                gif_generation=False,
                use_eval_env=False,
            )
            for index in indices
        ]
    )

    _assert_variables_unchanged(online_variables, agent.model.variables, "Online network")
    _assert_variables_unchanged(
        target_variables, agent.target_model.variables, "Target network"
    )
    _assert_variables_unchanged(optimizer_variables, agent.optimizer.variables, "Optimizer")

    payload = {
        "indices": indices,
        "max_actions": max_actions,
        "online_variables": online_variables,
        "target_variables": target_variables,
        "optimizer_variables": optimizer_variables,
        "results": results,
    }
    temporary = output_dir / "evaluation.pkl.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    temporary.replace(output_dir / "evaluation.pkl")
    results.to_csv(output_dir / "evaluation.csv", index=False)
    print(f"Saved evaluation for rows {indices} to {output_dir}")


if __name__ == "__main__":
    main()
