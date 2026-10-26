"""Versioned, immutable training checkpoints. Only load checkpoints you trust.

Checksums detect incomplete/corrupt files, not malicious pickle payloads. Component
snapshots are explicit dictionaries; live agents and environments are never pickled.
"""

from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import config as tf_config

from autorec.neural_state import neural_state_dict
from autorec.runtime_state import environment_state_dict, rng_state_dict


_CONFIG_FIELDS = frozenset(
    "random_seed action_cap episodes_trial num_trials gamma continuous_deadloop "
    "latent_deadloop invalid_terminals initial_epsilon epsilon_min epsilon_decay start_decay "
    "batch_size train_frequency update_target_frequency gradient_steps NN_sleep buffer_capacity "
    "prioritized_replay_alpha prioritized_replay_eps initial_beta beta_jump start_jump "
    "anneal_fraction final_beta optimizer_config bayesian _save_start _save_frequency hidden_layers".split()
)
_DYNAMIC_FIELDS = frozenset("epsilon prioritized_replay_beta epsilon_list beta_list".split())
_COMPONENTS = {"config.json", "training.pkl", "environments.pkl", "neural.pkl", "replay.npy"}


def _runtime():
    """Conservative compatibility: identical source, numerical stack and CPU settings."""
    if tf.config.get_visible_devices("GPU"):
        raise ValueError("Training checkpoints currently support CPU execution only.")
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        source.update(str(path.relative_to(Path(__file__).parent)).encode())
        source.update(path.read_bytes())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_sha256": source.hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "tensorflow",
                "keras",
                "scipy",
                "jax",
                "jaxlib",
                "autoeis",
            )
        },
        "deterministic_ops": tf_config.is_op_determinism_enabled(),
        "intra_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "TF_ENABLE_ONEDNN_OPTS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _capture(agent):
    from autorec.agent import DDQN_ECM

    expected = (
        _CONFIG_FIELDS
        | _DYNAMIC_FIELDS
        | {
            "training_env",
            "eval_env",
            "_active_env",
            "save_dir",
            "model_dir",
            "model_name",
            "model",
            "target_model",
            "optimizer",
            "_replay_buffer",
            "_training_state",
            "_continuation_mode",
        }
    )
    if type(agent) is not DDQN_ECM or set(vars(agent)) != expected:
        raise ValueError("Checkpoints require an unmodified DDQN_ECM agent.")
    state = agent._training_state
    if state is None or agent._replay_buffer is None:
        raise ValueError("Initialize training with train(additional_trials=0) before saving.")
    if not state.at_trial_boundary:
        raise ValueError("Checkpoints require a completed-trial boundary.")
    if state.schedule_config != agent._training_schedule_config():
        raise ValueError("The training horizon or schedules have changed.")
    replay = agent._replay_buffer.state_dict()
    config = {
        "agent": {key: getattr(agent, key) for key in sorted(_CONFIG_FIELDS)},
        "output": {"save_dir": str(agent.save_dir), "model_name": str(agent.model_name)},
    }
    training = {
        "attributes": {key: getattr(agent, key) for key in sorted(_DYNAMIC_FIELDS)},
        "progress": {field.name: getattr(state, field.name) for field in fields(state)},
        "replay": {key: value for key, value in replay.items() if key != "buffer"},
        "rng": rng_state_dict(),
    }
    return {
        "config.json": config,
        "training.pkl": training,
        "environments.pkl": environment_state_dict(
            agent.training_env, agent.eval_env, agent._active_env
        ),
        "neural.pkl": neural_state_dict(agent.model, agent.target_model, agent.optimizer),
        "replay.npy": replay["buffer"],
    }


def _digest(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def _sync_directory(path):
    # POSIX directory fsync makes publication durable on local filesystems.
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def save_checkpoint(agent, destination) -> Path:
    """Write a new checkpoint directory atomically; never replace an existing path.

    Call synchronously at a completed-trial boundary. No concurrent training or writers
    to the same destination are supported. Filesystem durability depends on fsync/rename.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    runtime = _runtime()
    payloads = _capture(agent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name, payload in payloads.items():
            with (temporary / name).open("wb") as stream:
                if name.endswith(".json"):
                    stream.write(json.dumps(payload, indent=2, allow_nan=False).encode())
                elif name.endswith(".npy"):
                    np.save(stream, payload, allow_pickle=False)
                else:
                    pickle.dump(payload, stream, protocol=5)
                stream.flush()
                os.fsync(stream.fileno())
        manifest = {
            "format_version": 1,
            "completed_trials": agent.completed_trials,
            "runtime": runtime,
            "files": {name: _digest(temporary / name) for name in sorted(_COMPONENTS)},
        }
        with (temporary / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        temporary.rename(destination)
        _sync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _read_checkpoint(directory):
    """Verify all files before deserializing any trusted pickle payload."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Checkpoint manifest must not be a symlink.")
    manifest = json.loads(manifest_path.read_text())
    if (
        set(manifest) != {"format_version", "completed_trials", "runtime", "files"}
        or manifest["format_version"] != 1
    ):
        raise ValueError("Unsupported checkpoint manifest/version.")
    if manifest["runtime"] != _runtime():
        raise ValueError(
            "Checkpoint runtime/source differs; exact continuation is unsupported."
        )
    if set(manifest["files"]) != _COMPONENTS:
        raise ValueError("Incomplete checkpoint component list.")
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Missing or corrupt checkpoint component: {name}")
    payloads = {}
    for name in sorted(_COMPONENTS):
        with (directory / name).open("rb") as stream:
            if name.endswith(".json"):
                payloads[name] = json.load(stream)
            elif name.endswith(".npy"):
                payloads[name] = np.load(stream, allow_pickle=False)
            else:
                payloads[name] = pickle.load(stream)
    progress = payloads["training.pkl"]["progress"]
    if (
        progress["at_trial_boundary"] is not True
        or progress["completed_trials"] != manifest["completed_trials"]
    ):
        raise ValueError("Checkpoint progress does not match its boundary/manifest.")
    return payloads
