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
import re
import tempfile
import uuid

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import config as tf_config

from autorec.neural_state import neural_state_dict, restore_neural_state
from autorec.optimized_data_structures.circular_buffer import CircularBuffer
from autorec.runtime_state import (
    environment_state_dict,
    restore_environment_state,
    restore_rng_state,
    rng_state_dict,
)


_CONFIG_FIELDS = frozenset(
    "random_seed action_cap episodes_trial num_trials gamma continuous_deadloop "
    "latent_deadloop invalid_terminals initial_epsilon epsilon_min epsilon_decay start_decay "
    "batch_size train_frequency update_target_frequency gradient_steps NN_sleep buffer_capacity "
    "prioritized_replay_alpha prioritized_replay_eps initial_beta beta_jump start_jump "
    "anneal_fraction beta_anneal_trials final_beta optimizer_config bayesian _save_start "
    "_save_frequency hidden_layers".split()
)
_DYNAMIC_FIELDS = frozenset("epsilon prioritized_replay_beta epsilon_list beta_list".split())
_RECONSTRUCTIBLE_FIELDS = frozenset(
    "_flatten_z_matrices _action_index_maps _compiled_train_step_signature "
    "_compiled_train_step".split()
)
_COMPONENTS = {"config.json", "training.pkl", "environments.pkl", "neural.pkl", "replay.npy"}
_CHECKPOINT_NAME = re.compile(r"trial-(\d{8})")


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
            "_checkpoint_series_id",
        }
    )
    # Performance caches are derived from checkpoint-owned environments, models, and
    # configuration. They are rebuilt lazily after restoration and must not be serialized.
    if type(agent) is not DDQN_ECM or set(vars(agent)) - _RECONSTRUCTIBLE_FIELDS != expected:
        raise ValueError("Checkpoints require an unmodified DDQN_ECM agent.")
    state = agent._training_state
    if state is None or agent._replay_buffer is None:
        raise ValueError("Initialize training with train(additional_trials=0) before saving.")
    if not state.at_trial_boundary:
        raise ValueError("Checkpoints require a completed-trial boundary.")
    if state.schedule_config != agent._training_schedule_config():
        raise ValueError("The epsilon/beta schedules have changed.")
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
    if agent._checkpoint_series_id is None:
        agent._checkpoint_series_id = uuid.uuid4().hex
    if not isinstance(agent._checkpoint_series_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", agent._checkpoint_series_id
    ):
        raise ValueError("Checkpoint series ID is invalid.")
    runtime = _runtime()
    payloads = _capture(agent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
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
            "format_version": 3,
            "series_id": agent._checkpoint_series_id,
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
    return destination


def _read_manifest(directory):
    """Read and validate checkpoint metadata without deserializing payloads."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Checkpoint manifest must not be a symlink.")
    if not manifest_path.is_file():
        raise ValueError(f"Checkpoint manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Checkpoint manifest is unreadable: {manifest_path}") from error
    if (
        set(manifest)
        != {"format_version", "series_id", "completed_trials", "runtime", "files"}
        or manifest["format_version"] != 3
    ):
        raise ValueError("Unsupported checkpoint manifest/version.")
    if not isinstance(manifest["series_id"], str) or not re.fullmatch(
        r"[0-9a-f]{32}", manifest["series_id"]
    ):
        raise ValueError("Checkpoint manifest has an invalid series ID.")
    completed_trials = manifest["completed_trials"]
    if isinstance(completed_trials, bool) or not isinstance(completed_trials, int):
        raise ValueError("Checkpoint manifest has an invalid completed-trial count.")
    if completed_trials < 0:
        raise ValueError("Checkpoint manifest has a negative completed-trial count.")
    return manifest


def _validate_checkpoint_files(directory, manifest):
    """Validate runtime, component inventory, sizes, and hashes without unpickling."""
    directory = Path(directory)
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


def find_latest_checkpoint(directory, *, target_trials=None):
    """Return the newest valid immutable checkpoint at or below a cumulative target.

    Missing or empty directories return ``None``. Atomic-write temporary directories and
    unrelated entries are ignored. Checkpoint-like entries that are malformed, corrupt,
    incompatible, or beyond ``target_trials`` fail clearly instead of causing a silent
    restart.
    """
    if target_trials is not None:
        if isinstance(target_trials, bool) or not isinstance(target_trials, int):
            raise TypeError("target_trials must be a nonnegative integer or None.")
        if target_trials < 0:
            raise ValueError("target_trials must be nonnegative.")

    directory = Path(directory)
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Checkpoint root must be a regular directory: {directory}")

    candidates = []
    series_id = None
    for path in directory.iterdir():
        if path.name.startswith("."):
            continue
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            if path.name.startswith("trial-"):
                raise ValueError(f"Malformed checkpoint directory name: {path.name}")
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Checkpoint path must be a regular directory: {path}")
        named_trials = int(match.group(1))
        manifest = _read_manifest(path)
        if series_id is None:
            series_id = manifest["series_id"]
        elif manifest["series_id"] != series_id:
            raise ValueError("Checkpoint root contains multiple training series.")
        if manifest["completed_trials"] != named_trials:
            raise ValueError(f"Checkpoint directory and manifest trial counts differ: {path}")
        if target_trials is not None and named_trials > target_trials:
            raise ValueError(
                f"Checkpoint trial {named_trials} exceeds target trial {target_trials}."
            )
        candidates.append((named_trials, path, manifest))

    if not candidates:
        return None
    _, path, manifest = max(candidates, key=lambda candidate: candidate[0])
    _validate_checkpoint_files(path, manifest)
    return path


def _read_checkpoint(directory):
    """Verify all files before deserializing any trusted pickle payload."""
    directory = Path(directory)
    manifest = _read_manifest(directory)
    _validate_checkpoint_files(directory, manifest)
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
    payloads["manifest.json"] = manifest
    return payloads


def load_checkpoint(directory, *, save_dir=None, num_trials=None):
    """Build a fresh agent from a trusted checkpoint and restore RNG streams LAST.

    ``save_dir`` may relocate outputs and ``num_trials`` may select the next session length.
    All training-critical configuration remains checkpoint-owned. Failed loads restore the
    caller's supported RNG streams and never modify an existing agent.
    """
    from autorec.agent import DDQN_ECM, _TrainingState

    if num_trials is not None:
        if isinstance(num_trials, bool) or not isinstance(num_trials, int):
            raise TypeError("num_trials must be a positive integer or None.")
        if num_trials < 1:
            raise ValueError("num_trials must be positive.")
    previous_rng = rng_state_dict()
    try:
        payloads = _read_checkpoint(directory)
        manifest = payloads.pop("manifest.json")
        config = payloads["config.json"]
        training = payloads["training.pkl"]
        if set(config) != {"agent", "output"} or set(config["agent"]) != _CONFIG_FIELDS:
            raise ValueError("Invalid checkpoint agent configuration.")
        if set(config["output"]) != {"save_dir", "model_name"}:
            raise ValueError("Invalid checkpoint output configuration.")
        if (
            set(training) != {"attributes", "progress", "replay", "rng"}
            or set(training["attributes"]) != _DYNAMIC_FIELDS
        ):
            raise ValueError("Invalid checkpoint training attributes.")
        if set(training["progress"]) != {field.name for field in fields(_TrainingState)}:
            raise ValueError("Invalid checkpoint progress fields.")
        agent = DDQN_ECM.__new__(DDQN_ECM)
        agent.__dict__.update(config["agent"])
        agent.__dict__.update(training["attributes"])
        if num_trials is not None:
            agent.num_trials = num_trials
        agent._training_state = _TrainingState(**training["progress"])
        progress = agent._training_state
        for name in (
            "iteration",
            "target_freq",
            "memory",
            "success_count",
            "completed_trials",
        ):
            value = getattr(progress, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid checkpoint counter: {name}")
        if type(agent.num_trials) is not int or agent.num_trials < 1:
            raise ValueError("Invalid num_trials session length.")
        if progress.schedule_config != agent._training_schedule_config():
            raise ValueError("Checkpoint schedules are inconsistent.")
        agent.training_env, agent.eval_env, agent._active_env = restore_environment_state(
            payloads["environments.pkl"]
        )
        agent._replay_buffer = CircularBuffer.from_state_dict(
            dict(training["replay"], buffer=payloads["replay.npy"])
        )
        expected_replay = agent._create_replay_buffer()
        if (
            agent._replay_buffer.capacity != expected_replay.capacity
            or agent._replay_buffer.buffer.dtype != expected_replay.buffer.dtype
        ):
            raise ValueError("Checkpoint replay does not match the agent/environment.")
        neural = payloads["neural.pkl"]
        agent.model = tf.keras.Sequential.from_config(neural["online"]["config"])
        agent.target_model = tf.keras.Sequential.from_config(neural["target"]["config"])
        agent.optimizer = restore_neural_state(neural, agent.model, agent.target_model)
        env = agent.training_env
        input_size = env.chromosome_len * len(env.ELEMENTS_EXTENDED) + env.EIS_INPUT_SIZE
        for model in (agent.model, agent.target_model):
            if model.input_shape != (None, input_size) or model.output_shape != (
                None,
                len(env.ACTIONS_LIST),
            ):
                raise ValueError("Checkpoint network dimensions do not match the environment.")
        agent._continuation_mode = True
        agent._checkpoint_series_id = manifest["series_id"]
        output = config["output"]
        agent.save_dir = Path(output["save_dir"] if save_dir is None else save_dir)
        agent.model_dir = agent.save_dir / "models"
        agent.model_name = (
            Path(output["model_name"])
            if save_dir is None
            else agent.save_dir / Path(output["model_name"]).name
        )
        agent.model_dir.mkdir(parents=True, exist_ok=True)
        restore_rng_state(training["rng"])
        return agent
    except BaseException:
        restore_rng_state(previous_rng)
        raise
