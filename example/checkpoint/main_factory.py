"""Run one configured initial or checkpoint-resumed training session."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path

import yaml


CHECKPOINT_SECTION = "checkpoint"
_RUNTIME_CACHE = Path(__file__).resolve().parent / "outputs/runtime_cache"


def _configure_cache_environment():
    """Keep plotting and Julia metadata caches with this experiment."""
    _RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUTOREC_ROOT", str(Path(__file__).resolve().parents[2]))
    os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE))
    os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE / "matplotlib"))
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(_RUNTIME_CACHE / "julia_env"))
    os.environ.setdefault("JULIA_DEPOT_PATH", str(_RUNTIME_CACHE / "julia_depot"))


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _read_and_validate(config_path: Path):
    config = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a dictionary at the top level.")
    try:
        checkpoint_config = config[CHECKPOINT_SECTION]
        agent_config = config["agent"]
    except KeyError as error:
        raise ValueError(f"Missing configuration section: {error.args[0]}") from error
    if not isinstance(checkpoint_config, dict):
        raise TypeError(f"{CHECKPOINT_SECTION} must be a dictionary.")
    if not isinstance(agent_config, dict):
        raise TypeError("agent must be a dictionary.")

    _positive_integer(agent_config.get("num_trials"), "agent.num_trials")
    for name in ("save_dir", "generated_config_dir"):
        if not checkpoint_config.get(name):
            raise ValueError(f"{CHECKPOINT_SECTION}.{name} is required.")
    resume_from = checkpoint_config.get("resume_from")
    if resume_from is not None and not isinstance(resume_from, str):
        raise TypeError(f"{CHECKPOINT_SECTION}.resume_from must be a path string or null.")
    return config, checkpoint_config


def _training_environment_config(config):
    environment = config.get("environment", {})
    return environment.get("training", environment)


def _validate_resumed_configuration(config, agent):
    """Reject changed settings that the checkpoint, rather than YAML, will own."""
    attribute_names = {
        "save_start": "_save_start",
        "save_frequency": "_save_frequency",
        "optimizer": "optimizer_config",
    }
    for name, configured in config["agent"].items():
        if name in {"save_dir", "num_trials"}:
            continue
        if name == "model":
            if configured is not None:
                raise ValueError(
                    "A resumed configuration cannot replace the checkpoint model."
                )
            continue
        attribute = attribute_names.get(name, name)
        if not hasattr(agent, attribute):
            raise ValueError(f"Unsupported resumed agent setting: {name}")
        expected = configured
        if name == "action_cap" and configured is None:
            expected = (
                agent.training_env.chromosome_HEAD_len
                + agent.training_env.chromosome_TAIL_len
                + 3
            )
        if getattr(agent, attribute) != expected:
            raise ValueError(f"Cannot change agent.{name} when resuming a checkpoint.")

    environment = _training_environment_config(config)
    environment_attributes = {
        "elements": "ELEMENTS",
        "initial_state": "_all_start_states",
        "seed": "seed",
        "chromosome_HEAD_len": "chromosome_HEAD_len",
        "cache_enabled": "cache_enabled",
        "cache_capacity": "cache_capacity",
        "cache_type": "cache_type",
    }
    for name, attribute in environment_attributes.items():
        if name in environment and getattr(agent.training_env, attribute) != environment[name]:
            raise ValueError(f"Cannot change environment.{name} when resuming a checkpoint.")


def _write_next_config(config, completed_trials: int) -> Path:
    next_config = deepcopy(config)
    checkpoint_config = next_config[CHECKPOINT_SECTION]
    checkpoint_config["resume_from"] = str(
        Path(checkpoint_config["save_dir"]) / f"trial-{completed_trials:08d}"
    )
    output_dir = Path(checkpoint_config["generated_config_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"resume-from-trial-{completed_trials:08d}.yaml"
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(next_config, sort_keys=False))
    temporary.replace(destination)
    return destination


def run_session(config_path: Path) -> Path:
    config, checkpoint_config = _read_and_validate(config_path)
    _configure_cache_environment()

    from autorec.runtime import configure_autorec_runtime

    # Configure JuliaCall/AutoEIS and numerical runtimes before their dependent modules.
    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    import tensorflow as tf

    from autorec.agent import DDQN_ECM
    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    resume_from = checkpoint_config.get("resume_from")
    session_trials = config["agent"]["num_trials"]
    if resume_from is None:
        set_global_seed(config["agent"]["random_seed"], deterministic_ops=True)
        # pipeline_builder ignores the checkpoint section and deep-copies the input.
        _, _, _, _, agent = pipeline_builder(config)
    else:
        # Enable deterministic kernels without replacing the checkpoint RNG streams.
        tf.config.experimental.enable_op_determinism()
        agent = DDQN_ECM.from_checkpoint(
            Path(resume_from),
            save_dir=Path(config["agent"]["save_dir"]),
            num_trials=session_trials,
        )
        _validate_resumed_configuration(config, agent)

    start = agent.completed_trials
    agent.train(checkpoint_dir=Path(checkpoint_config["save_dir"]))
    expected_stop = start + session_trials
    if agent.completed_trials != expected_stop:
        raise RuntimeError(
            f"Training stopped at trial {agent.completed_trials}; expected {expected_stop}."
        )

    next_config = _write_next_config(config, agent.completed_trials)
    print(f"Completed session at cumulative trial {agent.completed_trials}.")
    print(f"Next configuration, if more training is needed: {next_config}")
    return next_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, required=True)
    args = parser.parse_args()
    run_session(args.config.resolve())


if __name__ == "__main__":
    main()
