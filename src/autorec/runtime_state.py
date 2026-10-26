"""RNG and prepared-environment snapshots for paused, supported AutoREC training.

These are trusted in-memory payloads, not a portable or safe unpickling API. TensorFlow
legacy random-op counters are not recoverable; AutoREC's supported training path uses
NumPy sampling and separately captured Keras layer-local generators instead.
"""

import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import stateful_random_ops

from autorec.environment import EIS_ECM_Env
from autorec.optimized_data_structures.circuit_cache import HistoryCache


_ENV_FIELDS = frozenset(
    "dataset ELEMENTS ELEMENTS_EXTENDED _operator _terminal seed _all_start_states "
    "EIS_measurement_size EIS_INPUT_SIZE chromosome_HEAD_len chromosome_TAIL_len "
    "chromosome_len cache_enabled cache_capacity cache_type start_state_list ACTIONS_LIST "
    "current_eis_index encoded_state START_STATE state Z freq Z_norm_flatten true_circuit "
    "chi_threshold r2_threshold".split()
)


def _copy_graph(value):
    # DataFrame.copy/deepcopy does not copy arrays inside object cells. One pickle graph
    # preserves these values and their shared references without serializing live objects.
    return pickle.loads(pickle.dumps(value, protocol=5))


def rng_state_dict() -> dict:
    """Capture Python, NumPy (including Gaussian caches), and an existing TF generator.

    Inspect the pinned TensorFlow singleton without initializing a new generator, which
    the public getter would do. Keras layer generators belong to neural_state instead.
    """
    generator = stateful_random_ops.global_generator
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "tensorflow": None
        if generator is None
        else {"algorithm": generator.algorithm, "state": generator.state.numpy().copy()},
    }


def restore_rng_state(state: dict) -> None:
    """Restore streams last, after reconstruction. Validate before changing any stream."""
    if set(state) != {"version", "python", "numpy", "tensorflow"} or state["version"] != 1:
        raise ValueError("Invalid RNG state schema/version.")
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    saved = state["tensorflow"]
    generator = None
    if saved is not None:
        if set(saved) != {"algorithm", "state"}:
            raise ValueError("Invalid TensorFlow generator state.")
        generator = tf.random.Generator.from_state(saved["state"], saved["algorithm"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # The public setter cannot represent the original uninitialized singleton.
    stateful_random_ops.global_generator = generator


def environment_state_dict(training_env, eval_env, active_env) -> dict:
    """Capture prepared data, caches, current observations, and environment identities."""
    environments = [training_env] if training_env is eval_env else [training_env, eval_env]
    if not any(active_env is env for env in environments):
        raise ValueError("Active environment must be training_env or eval_env.")
    records = []
    for env in environments:
        if type(env) is not EIS_ECM_Env or set(vars(env)) != _ENV_FIELDS | {"cache"}:
            raise ValueError("Environment snapshots require an unmodified EIS_ECM_Env.")
        records.append(
            {
                "attributes": {key: getattr(env, key) for key in sorted(_ENV_FIELDS)},
                "cache": None if env.cache is None else env.cache.state_dict(),
            }
        )
    return _copy_graph(
        {"version": 1, "environments": records, "active": int(active_env is not training_env)}
    )


def restore_environment_state(state: dict):
    """Return training/evaluation/active environments without reset, fitting, or seeding."""
    state = _copy_graph(state)
    if set(state) != {"version", "environments", "active"} or state["version"] != 1:
        raise ValueError("Invalid environment state schema/version.")
    records = state["environments"]
    if (
        len(records) not in (1, 2)
        or type(state["active"]) is not int
        or not 0 <= state["active"] < len(records)
    ):
        raise ValueError("Invalid environment identities.")
    environments = []
    for record in records:
        if set(record) != {"attributes", "cache"} or set(record["attributes"]) != _ENV_FIELDS:
            raise ValueError("Invalid environment attributes.")
        env = EIS_ECM_Env.__new__(EIS_ECM_Env)
        env.__dict__.update(record["attributes"])
        env.cache = (
            None if record["cache"] is None else HistoryCache.from_state_dict(record["cache"])
        )
        if env.cache_enabled != (env.cache is not None):
            raise ValueError("Environment cache configuration does not match its state.")
        environments.append(env)
    return environments[0], environments[-1], environments[state["active"]]
