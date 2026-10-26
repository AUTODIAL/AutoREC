"""Neural component snapshots; no agent reconstruction, file I/O, or global RNG handling.

Restoration uses existing built networks and returns a new optimizer. TensorFlow/Keras
versions and supported configurations are checked before modifying either network.
"""

from copy import deepcopy
import json

import numpy as np
import tensorflow as tf

from autorec.factory.optimizer import create_optimizer, get_optimizer_config


_LAYERS = (
    tf.keras.layers.Dense,
    tf.keras.layers.BatchNormalization,
    tf.keras.layers.Dropout,
    tf.keras.layers.Activation,
    tf.keras.layers.Flatten,
)
_SCHEDULES = (
    tf.keras.optimizers.schedules.ExponentialDecay,
    tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    tf.keras.optimizers.schedules.CosineDecay,
)


def _keys(value, required, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dictionary.")
    if value.keys() != set(required):
        raise ValueError(f"Invalid {label} fields; expected {sorted(required)}.")


def _builtin_config(value):
    """Reject serialized custom objects rather than claiming to capture their hidden state."""
    if isinstance(value, dict):
        module = value.get("module")
        if module is not None and module != "keras" and not module.startswith("keras."):
            raise ValueError("Neural snapshots support built-in Keras objects only.")
        for item in value.values():
            _builtin_config(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _builtin_config(item)


def _model_config(model):
    if type(model) is not tf.keras.Sequential or not model.built:
        raise ValueError("Neural snapshots require built Keras Sequential models.")
    if tf.keras.backend.backend() != "tensorflow":
        raise ValueError("Neural snapshots require the TensorFlow Keras backend.")
    if model.compiled:
        raise ValueError("Compiled models are not supported by neural snapshots.")
    if model.dtype_policy.name not in ("float32", "float64"):
        raise ValueError("Neural snapshots support float32/float64 policies only.")
    for layer in model.layers:
        if type(layer) not in _LAYERS:
            raise ValueError(f"Unsupported neural snapshot layer: {type(layer).__name__}.")
        if layer.dtype_policy.name not in ("float32", "float64"):
            raise ValueError("Neural snapshots support float32/float64 policies only.")
        if getattr(layer, "quantization_mode", None) is not None:
            raise ValueError("Quantized layers are not supported by neural snapshots.")
        if getattr(layer, "lora_enabled", False):
            raise ValueError("LoRA layers are not supported by neural snapshots.")
    config = deepcopy(model.get_config())
    _builtin_config(config)
    try:
        json.dumps(config)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Neural snapshots require plain serializable model configurations."
        ) from error
    return config


def _architecture(config):
    """Sequential topology is positional; generated model/layer names may differ."""
    if not isinstance(config, dict) or not isinstance(config.get("layers"), list):
        raise ValueError("Invalid Sequential model configuration.")
    config = deepcopy(config)
    config.pop("name", None)
    for layer in config["layers"]:
        if not isinstance(layer, dict) or not isinstance(layer.get("config"), dict):
            raise ValueError("Invalid layer configuration.")
        layer["config"].pop("name", None)
    return json.dumps(config, sort_keys=True)


def _optimizer_config(optimizer, model):
    if type(optimizer) not in (tf.keras.optimizers.Adam, tf.keras.optimizers.SGD):
        raise ValueError("Neural snapshots support built-in Adam and SGD only.")
    # These private Keras attributes are qualified against the recorded, pinned version.
    rate = optimizer._learning_rate
    if isinstance(rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        if type(rate) not in _SCHEDULES:
            raise ValueError("Unsupported neural snapshot learning-rate schedule.")
    elif callable(rate):
        raise ValueError("Callable learning rates are not supported by neural snapshots.")
    config = get_optimizer_config(optimizer)
    try:
        json.dumps(config)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Neural snapshots require constant, serializable optimizer/schedule configuration."
        ) from error
    if (
        getattr(optimizer, "_exclude_from_weight_decay", None)
        or getattr(optimizer, "_exclude_from_weight_decay_pattern", None) is not None
    ):
        raise ValueError("Weight-decay exclusions are not supported by neural snapshots.")
    if (
        config.get("use_ema")
        or config.get("gradient_accumulation_steps") is not None
        or config.get("loss_scale_factor") is not None
    ):
        raise ValueError("EMA, gradient accumulation, and loss scaling are not supported.")
    if optimizer.built:
        bound = optimizer._trainable_variables
        expected = model.trainable_variables
        if len(bound) != len(expected) or any(a is not b for a, b in zip(bound, expected)):
            raise ValueError("Optimizer must be bound to the online variables in model order.")
    return deepcopy(config)


def _distinct_models(model, target_model):
    if not isinstance(model, tf.keras.Model) or not isinstance(target_model, tf.keras.Model):
        raise TypeError("Online and target must be Keras models.")
    if model is target_model or {id(v) for v in model.variables} & {
        id(v) for v in target_model.variables
    }:
        raise ValueError("Online and target networks must have independent variables.")


def _model_state(model):
    return {
        "config": _model_config(model),
        "variables": [
            {
                "name": variable.name,
                "trainable": bool(variable.trainable),
                "value": np.array(variable.numpy(), copy=True),
            }
            for variable in model.variables
        ],
    }


def neural_state_dict(model, target_model, optimizer) -> dict:
    """Export independent online, target, and optimizer component state.

    Includes nontrainable variables, layer-local seed counters, optimizer slots/iterations,
    mutable learning rate, schedule configuration, and whether slots have been built.
    Export does not build an optimizer, run a network, or consume random draws.
    Use only while training is paused. This is not a complete training checkpoint.
    """
    _distinct_models(model, target_model)
    config = _optimizer_config(optimizer, model)
    return {
        "version": 1,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "online": _model_state(model),
        "target": _model_state(target_model),
        "optimizer": {
            "config": config,
            "built": bool(optimizer.built),
            "variables": [
                np.array(variable.numpy(), copy=True) for variable in optimizer.variables
            ],
        },
    }


def _array(value, variable, label):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array.")
    if value.dtype != np.dtype(variable.dtype) or value.shape != tuple(variable.shape):
        raise ValueError(f"{label} dtype/shape does not match the destination variable.")


def _validate_model_state(state, model, label):
    _keys(state, ("config", "variables"), label)
    if _architecture(state["config"]) != _architecture(_model_config(model)):
        raise ValueError(f"{label} architecture/configuration does not match.")
    records = state["variables"]
    if not isinstance(records, list) or len(records) != len(model.variables):
        raise ValueError(f"{label} variable count does not match.")
    for record, variable in zip(records, model.variables):
        _keys(record, ("name", "trainable", "value"), f"{label} variable")
        if (
            record["name"] != variable.name
            or type(record["trainable"]) is not bool
            or record["trainable"] != variable.trainable
        ):
            raise ValueError(f"{label} variable identity/trainability does not match.")
        _array(record["value"], variable, label)


def restore_neural_state(state: dict, model, target_model) -> tf.keras.optimizers.Optimizer:
    """Restore into compatible built networks and RETURN the restored optimizer.

    Assign the return value to ``agent.optimizer``. A new optimizer is constructed with
    existing factory helpers; slots are built only if the snapshot had them, without a
    gradient update. All schema/configuration/array checks finish before network assignment.
    Validation failures leave the supplied networks unchanged. No model is initialized,
    target is never copied from online, and no global RNG state is saved/restored here.

    Supported: uncompiled built-in Sequential Dense, BatchNormalization, Dropout, Activation, Flatten;
    float32/float64; Adam (including AMSGrad) or SGD; constant LR or ExponentialDecay,
    PiecewiseConstantDecay, CosineDecay. Versions must match exactly. Other configurations
    remain usable for ordinary training but are not supported by these component methods.
    """
    _keys(
        state,
        ("version", "tensorflow_version", "keras_version", "online", "target", "optimizer"),
        "neural state",
    )
    if type(state["version"]) is not int or state["version"] != 1:
        raise ValueError("Unsupported neural state version.")
    if (
        state["tensorflow_version"] != tf.__version__
        or state["keras_version"] != tf.keras.__version__
    ):
        raise ValueError("Neural state TensorFlow/Keras versions must match exactly.")
    _distinct_models(model, target_model)
    _validate_model_state(state["online"], model, "online")
    _validate_model_state(state["target"], target_model, "target")
    saved = state["optimizer"]
    _keys(saved, ("config", "built", "variables"), "optimizer state")
    if type(saved["built"]) is not bool:
        raise TypeError("Optimizer built status must be a boolean.")
    if not isinstance(saved["config"], dict) or saved["config"].get("type") not in (
        "Adam",
        "SGD",
    ):
        raise ValueError("Neural snapshots support built-in Adam and SGD only.")
    config = deepcopy(saved["config"])
    lr = config.get("learning_rate")
    if isinstance(lr, dict) and lr.get("type") not in {item.__name__ for item in _SCHEDULES}:
        raise ValueError("Unsupported neural snapshot learning-rate schedule.")
    optimizer = create_optimizer(config)
    if _optimizer_config(optimizer, model) != config:
        raise ValueError("Optimizer configuration does not round-trip exactly.")
    if saved["built"]:
        optimizer.build(model.trainable_variables)
    arrays = saved["variables"]
    if not isinstance(arrays, list) or len(arrays) != len(optimizer.variables):
        raise ValueError("Optimizer variable count does not match built status/configuration.")
    for value, variable in zip(arrays, optimizer.variables):
        _array(value, variable, "optimizer variable")
    if arrays[0].item() < 0:
        raise ValueError("Optimizer iteration must be nonnegative.")
    for value, variable in zip(arrays, optimizer.variables):
        variable.assign(value)
    for label, destination in (("online", model), ("target", target_model)):
        for record, variable in zip(state[label]["variables"], destination.variables):
            variable.assign(record["value"])
    return optimizer
