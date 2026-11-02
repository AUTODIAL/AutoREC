"""Fresh-process checkpoint acceptance worker; controlled fitting, real RL/TF updates."""

import argparse
import hashlib
import pickle
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from test_training_baseline import controlled_fit, environment_factory
from autorec.agent import DDQN_ECM
from autorec import cli
from autorec.checkpoint import _capture
from autorec.environment import EIS_ECM_Env
from autorec.utils import set_global_seed


def exact(value):
    """Stable comparison form, including ordered containers and raw numerical bytes."""
    if type(value) is float:
        return ("float64", struct.pack("!d", value))
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
    # Session length is intentionally mutable and does not affect cumulative training state.
    payloads["config.json"]["agent"].pop("num_trials")
    return exact(payloads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--schedule-trials", type=int, default=12)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cache", default="lru")
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--auto-target", type=int)
    parser.add_argument("--interrupt-at", type=int)
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
    sample = DDQN_ECM._sample_experience

    def sampled(self, *args, **kwargs):
        result = sample(self, *args, **kwargs)
        traces.append(("sample", exact(result)))
        return result

    patches.setattr(DDQN_ECM, "_sample_experience", sampled)
    interact = DDQN_ECM._interact_update

    def traced(self, *args, **kwargs):
        result = interact(self, *args, **kwargs)
        traces.append(exact(result))
        return result

    patches.setattr(DDQN_ECM, "_interact_update", traced)

    def make_agent(count):
        return DDQN_ECM(
            make_env(args.cache),
            save_dir=args.directory,
            num_trials=count,
            hidden_layers=[{"type": "Dense", "units": 4, "activation": "relu"}],
            action_cap=2,
            batch_size=2,
            buffer_capacity=7,
            NN_sleep=3,
            train_frequency=3 if args.schedule_trials < 1000 else 197,
            update_target_frequency=5 if args.schedule_trials < 1000 else 137,
            epsilon_decay=0.995 if args.schedule_trials < 1000 else 0.99999,
            start_decay=1,
            start_jump=(2 / 5) * args.schedule_trials,
            beta_anneal_trials=0.595 * args.schedule_trials,
            save_start=args.schedule_trials + 1,
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

    if args.auto_target is not None:
        config = {
            "environment": {"cache": args.cache},
            "agent": {
                "num_trials": args.auto_target,
                "schedule_trials": args.schedule_trials,
            },
        }
        patches.setattr(cli, "_configure_runtime", lambda command_args: None)
        patches.setattr(cli, "_read_config_with_overrides", lambda *unused: config)
        patches.setattr(
            "autorec.factory.pipeline_builder",
            lambda unused: (None, None, None, None, make_agent(args.auto_target)),
        )

        class SimulatedInterruption(RuntimeError):
            pass

        if args.interrupt_at is not None:
            save_checkpoint = DDQN_ECM.save_checkpoint

            def interrupt_after_save(self, directory):
                path = save_checkpoint(self, directory)
                if self.completed_trials == args.interrupt_at:
                    raise SimulatedInterruption("simulated process interruption")
                return path

            patches.setattr(DDQN_ECM, "save_checkpoint", interrupt_after_save)

        command_args = SimpleNamespace(
            resume_latest=True,
            resume=None,
            config=Path("controlled-config.yaml"),
            checkpoint_dir=args.directory / "periodic",
            checkpoint_every=args.save_every,
            additional_trials=None,
            output_dir=args.directory,
            seed=42,
            non_deterministic=False,
        )
        try:
            cli._run_train(command_args)
        except SimulatedInterruption:
            with (args.directory / "result.pkl").open("wb") as stream:
                pickle.dump({"trace": traces}, stream, protocol=5)
            raise SystemExit(75) from None
        final = args.directory / "periodic" / f"trial-{args.auto_target:08d}"
        agent = DDQN_ECM.from_checkpoint(final, save_dir=args.directory)
    elif args.resume:
        agent = DDQN_ECM.from_checkpoint(
            args.resume, save_dir=args.directory, num_trials=args.count
        )
        agent.train(
            checkpoint_dir=args.directory / "periodic" if args.save_every else None,
            checkpoint_every=args.save_every,
        )
    else:
        agent = make_agent(args.count)
        agent.train(
            checkpoint_dir=args.directory / "periodic" if args.save_every else None,
            checkpoint_every=args.save_every,
        )
    state = signature(agent)
    if args.auto_target is None:
        agent.save_checkpoint(args.directory / "checkpoint")
        assert signature(agent) == state, "Checkpoint saving changed training state"
    with (args.directory / "result.pkl").open("wb") as stream:
        pickle.dump({"state": state, "trace": traces}, stream, protocol=5)
    print("State digest:", hashlib.sha256(pickle.dumps(state)).hexdigest())


if __name__ == "__main__":
    main()
