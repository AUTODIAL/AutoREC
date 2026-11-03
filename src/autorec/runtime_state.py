"""RNG and prepared-environment snapshots for paused, supported AutoREC training.

These are trusted in-memory payloads, not a portable or safe unpickling API. TensorFlow
legacy random-op counters are not recoverable; AutoREC's supported training path uses
NumPy sampling and separately captured Keras layer-local generators instead.
"""

import importlib
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import stateful_random_ops

from autorec.environment import EIS_ECM_Env
from autorec.optimized_data_structures.circuit_cache import HistoryCache


_ENV_FIELDS = frozenset(
    "dataset dataprep ELEMENTS ELEMENTS_EXTENDED _operator _terminal seed _all_start_states "
    "EIS_measurement_size EIS_INPUT_SIZE chromosome_HEAD_len chromosome_TAIL_len "
    "chromosome_len cache_enabled cache_capacity cache_type start_state_list ACTIONS_LIST "
    "current_eis_index active_spectrum_view current_spectrum_view_id encoded_state "
    "START_STATE state Z freq Z_norm_flatten true_circuit chi_threshold r2_threshold".split()
)


def _checkpoint_fields(environment_class) -> frozenset[str]:
    if not issubclass(environment_class, EIS_ECM_Env):
        raise ValueError("Checkpoint environments must inherit from EIS_ECM_Env.")
    if environment_class is EIS_ECM_Env:
        fields = environment_class.CHECKPOINT_FIELDS
    else:
        fields = environment_class.__dict__.get("CHECKPOINT_FIELDS")
        if fields is None:
            raise ValueError("Environment subclasses must declare CHECKPOINT_FIELDS.")
    if (
        type(fields) is not frozenset
        or not all(type(name) is str and name for name in fields)
        or fields & (_ENV_FIELDS | {"cache"})
    ):
        raise ValueError("Environment CHECKPOINT_FIELDS are invalid.")
    return fields


def environment_source_modules(*environments) -> tuple[str, ...]:
    """Return custom environment and declared dependency modules to fingerprint."""
    modules = set()
    for env in environments:
        environment_class = type(env)
        _checkpoint_fields(environment_class)
        if environment_class is EIS_ECM_Env:
            continue
        dependencies = environment_class.__dict__.get("CHECKPOINT_SOURCE_MODULES")
        if type(dependencies) is not tuple or not all(
            type(name) is str and name for name in dependencies
        ):
            raise ValueError(
                "Environment subclasses must declare CHECKPOINT_SOURCE_MODULES as a tuple."
            )
        modules.add(environment_class.__module__)
        modules.update(dependencies)
    return tuple(sorted(modules))


def _environment_class_record(environment_class) -> dict:
    fields = _checkpoint_fields(environment_class)
    return {
        "module": environment_class.__module__,
        "qualname": environment_class.__qualname__,
        "fields": sorted(fields),
    }


def _restore_environment_class(record: dict):
    if set(record) != {"module", "qualname", "fields"}:
        raise ValueError("Invalid environment class record.")
    module = importlib.import_module(record["module"])
    environment_class = module
    for name in record["qualname"].split("."):
        environment_class = getattr(environment_class, name)
    fields = _checkpoint_fields(environment_class)
    if record["fields"] != sorted(fields):
        raise ValueError("Environment checkpoint fields do not match the class.")
    return environment_class, fields


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
    if (
        training_env is not eval_env
        and training_env.cache is not None
        and training_env.cache is eval_env.cache
    ):
        raise ValueError("Distinct environments must not share a mutable circuit cache.")
    if not any(active_env is env for env in environments):
        raise ValueError("Active environment must be training_env or eval_env.")
    records = []
    for env in environments:
        environment_class = type(env)
        checkpoint_fields = _checkpoint_fields(environment_class)
        attributes = _ENV_FIELDS | checkpoint_fields
        if set(vars(env)) != attributes | {"cache"}:
            raise ValueError("Environment snapshot attributes are incomplete or undeclared.")
        records.append(
            {
                "class": _environment_class_record(environment_class),
                "attributes": {key: getattr(env, key) for key in sorted(attributes)},
                "cache": None if env.cache is None else env.cache.state_dict(),
            }
        )
    return _copy_graph(
        {"version": 2, "environments": records, "active": int(active_env is not training_env)}
    )


def restore_environment_state(state: dict):
    """Return training/evaluation/active environments without reset, fitting, or seeding."""
    state = _copy_graph(state)
    if set(state) != {"version", "environments", "active"} or state["version"] != 2:
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
        if set(record) != {"class", "attributes", "cache"}:
            raise ValueError("Invalid environment record.")
        environment_class, checkpoint_fields = _restore_environment_class(record["class"])
        if set(record["attributes"]) != _ENV_FIELDS | checkpoint_fields:
            raise ValueError("Invalid environment attributes.")
        env = environment_class.__new__(environment_class)
        env.__dict__.update(record["attributes"])
        env.cache = (
            None if record["cache"] is None else HistoryCache.from_state_dict(record["cache"])
        )
        if env.cache_enabled != (env.cache is not None):
            raise ValueError("Environment cache configuration does not match its state.")
        environments.append(env)
    return environments[0], environments[-1], environments[state["active"]]
