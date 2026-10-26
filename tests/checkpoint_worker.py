"""Fresh-process checkpoint acceptance worker; controlled fitting, real RL/TF updates."""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from test_training_baseline import controlled_fit, environment_factory
from autorec.agent import DDQN_ECM
from autorec.checkpoint import _capture, load_checkpoint, save_checkpoint
from autorec.environment import EIS_ECM_Env
from autorec.utils import set_global_seed


def exact(value):
    """Stable comparison form, including ordered containers and raw numerical bytes."""
    if isinstance(value, np.ndarray):
        return (
            value.dtype.descr if value.dtype.names else value.dtype.str,
            value.shape,
            value.tobytes(),
        )
    if isinstance(value, np.generic):
        return exact(np.asarray(value))
    if isinstance(value, pd.DataFrame):
        return (
            tuple(value.columns),
            tuple(value.index),
            tuple(str(dtype) for dtype in value.dtypes),
            tuple(exact(row) for row in value.itertuples(index=False, name=None)),
        )
    if isinstance(value, dict):
        return tuple((key, exact(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(exact(item) for item in value)
    return value


def signature(agent):
    payloads = _capture(agent)
    payloads["config.json"].pop("output")
    return exact(payloads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cache", default="lru")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    patches = pytest.MonkeyPatch()
    make_env = environment_factory.__wrapped__(patches)
    patches.setattr(
        EIS_ECM_Env,
        "_evaluate_circuit",
        lambda self, circuit, components_number: controlled_fit(circuit, components_number),
    )
    patches.setattr(DDQN_ECM, "_plot_training_metrics", lambda *args: None)
    patches.setattr(tf.keras.Sequential, "save", lambda *args, **kwargs: None)
    set_global_seed(42)
    traces = []
    interact = DDQN_ECM._interact_update

    def traced(self, *args, **kwargs):
        result = interact(self, *args, **kwargs)
        traces.append(exact(result))
        return result

    patches.setattr(DDQN_ECM, "_interact_update", traced)
    if args.resume:
        agent = load_checkpoint(args.resume, save_dir=args.directory)
    else:
        agent = DDQN_ECM(
            make_env(args.cache),
            save_dir=args.directory,
            num_trials=args.horizon,
            hidden_layers=[{"type": "Dense", "units": 4, "activation": "relu"}],
            action_cap=2,
            batch_size=2,
            buffer_capacity=7,
            NN_sleep=3,
            train_frequency=3 if args.horizon < 1000 else 197,
            update_target_frequency=5 if args.horizon < 1000 else 137,
            epsilon_decay=0.995,
            start_decay=1,
            save_start=args.horizon + 1,
            optimizer={
                "type": "Adam",
                "learning_rate": {
                    "type": "ExponentialDecay",
                    "initial_learning_rate": 0.001,
                    "decay_steps": 3,
                    "decay_rate": 0.9,
                },
            },
        )
    agent.train(additional_trials=args.count)
    state = signature(agent)
    save_checkpoint(agent, args.directory / "checkpoint")
    assert signature(agent) == state, "Checkpoint saving changed training state"
    with (args.directory / "result.pkl").open("wb") as stream:
        pickle.dump({"state": state, "trace": traces}, stream, protocol=5)
    print("State digest:", hashlib.sha256(pickle.dumps(state)).hexdigest())


if __name__ == "__main__":
    main()
