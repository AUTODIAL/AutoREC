"""Exact neural snapshots and subsequent optimizer updates on small CPU models."""

from copy import deepcopy
import pickle
import random

import numpy as np
import pytest
import tensorflow as tf

from autorec.factory.optimizer import create_optimizer
from autorec.neural_state import neural_state_dict, restore_neural_state


@pytest.fixture(autouse=True)
def preserve_random_states():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with tf.device("/CPU:0"):
        yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def models(*, extras=False, dtype="float32"):
    layers = [
        tf.keras.layers.Input((2,), dtype=dtype),
        tf.keras.layers.Dense(3, activation="tanh", dtype=dtype),
    ]
    if extras:
        layers += [
            tf.keras.layers.BatchNormalization(dtype=dtype),
            tf.keras.layers.Dropout(0.25, seed=19, dtype=dtype),
        ]
    layers.append(tf.keras.layers.Dense(1, dtype=dtype))
    online = tf.keras.Sequential(layers)
    target = tf.keras.models.clone_model(online)
    for variable in target.variables:
        if variable.dtype.startswith("float"):
            variable.assign(tf.ones_like(variable) * 0.375)
    return online, target


def arrays(variables):
    return [
        (value.dtype, value.shape, value.tobytes())
        for value in (np.asarray(variable.numpy()) for variable in variables)
    ]


def step(model, optimizer):
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=model.inputs[0].dtype)
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss = tf.reduce_mean(tf.square(prediction - tf.cast(0.5, prediction.dtype)))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return np.asarray(loss.numpy()).tobytes()


OPTIMIZERS = [
    {"type": "Adam", "learning_rate": 0.01},
    {"type": "Adam", "learning_rate": 0.01, "amsgrad": True},
    {"type": "SGD", "learning_rate": 0.01, "momentum": 0.75, "nesterov": True},
    {
        "type": "Adam",
        "learning_rate": {
            "type": "ExponentialDecay",
            "initial_learning_rate": 0.01,
            "decay_steps": 2,
            "decay_rate": 0.5,
            "staircase": True,
        },
    },
    {
        "type": "SGD",
        "momentum": 0.75,
        "learning_rate": {
            "type": "PiecewiseConstantDecay",
            "boundaries": [1, 3],
            "values": [0.01, 0.005, 0.0025],
        },
    },
    {
        "type": "Adam",
        "learning_rate": {
            "type": "CosineDecay",
            "initial_learning_rate": 0.01,
            "decay_steps": 8,
        },
    },
]


@pytest.mark.parametrize("config", OPTIMIZERS)
@pytest.mark.parametrize("phase", ["unbuilt", "built", "trained"])
def test_exact_roundtrip_and_next_updates(config, phase):
    online, target = models()
    optimizer = create_optimizer(config)
    if phase != "unbuilt":
        optimizer.build(online.trainable_variables)
    if phase == "trained":
        for _ in range(3):
            step(online, optimizer)
        if not isinstance(config["learning_rate"], dict):
            optimizer.learning_rate.assign(0.003125)  # Capture the live mutable LR.
    destination, destination_target = models()
    source_before = (
        arrays(online.variables),
        arrays(target.variables),
        arrays(optimizer.variables),
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    snapshot = neural_state_dict(online, target, optimizer)
    wire = pickle.loads(pickle.dumps(snapshot))  # Trusted component dictionary only.
    restored_optimizer = restore_neural_state(wire, destination, destination_target)
    assert restored_optimizer is not optimizer
    assert restored_optimizer.built == optimizer.built == (phase != "unbuilt")
    assert source_before == (
        arrays(online.variables),
        arrays(target.variables),
        arrays(optimizer.variables),
    )
    assert arrays(destination.variables) == arrays(online.variables)
    assert arrays(destination_target.variables) == arrays(target.variables)
    assert arrays(destination.variables) != arrays(destination_target.variables)
    assert arrays(restored_optimizer.variables) == arrays(optimizer.variables)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert len(optimizer.variables) == len(snapshot["optimizer"]["variables"])
    for _ in range(3):
        assert (
            np.asarray(restored_optimizer.learning_rate).tobytes()
            == np.asarray(optimizer.learning_rate).tobytes()
        )
        assert step(destination, restored_optimizer) == step(online, optimizer)
        assert arrays(destination.variables) == arrays(online.variables)
        assert arrays(restored_optimizer.variables) == arrays(optimizer.variables)
        assert arrays(destination_target.variables) == source_before[1]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nontrainable_buffers_and_layer_seed_counters_are_independent(dtype):
    online, target = models(extras=True, dtype=dtype)
    optimizer = tf.keras.optimizers.Adam()
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    online(x, training=True)
    online(x, training=True)
    target(x, training=True)
    step(online, optimizer)
    snapshot = neural_state_dict(online, target, optimizer)
    assert len(online.variables) > len(online.weights)  # Seed state is not in weights.
    destination, destination_target = models(extras=True, dtype=dtype)
    restored_optimizer = restore_neural_state(snapshot, destination, destination_target)
    assert arrays(destination.variables) == arrays(online.variables)
    assert arrays(destination_target.variables) == arrays(target.variables)
    # Exercise the next stochastic call independently for BOTH networks.
    assert (
        destination(x, training=True).numpy().tobytes()
        == online(x, training=True).numpy().tobytes()
    )
    assert (
        destination_target(x, training=True).numpy().tobytes()
        == target(x, training=True).numpy().tobytes()
    )
    assert arrays(destination.variables) == arrays(online.variables)
    assert arrays(destination_target.variables) == arrays(target.variables)
    assert step(destination, restored_optimizer) == step(online, optimizer)
    assert arrays(destination.variables) == arrays(online.variables)


def test_snapshots_are_independent_and_frozen_layers_are_preserved():
    online, target = models()
    online.layers[0].trainable = False
    optimizer = tf.keras.optimizers.Adam()
    step(online, optimizer)
    snapshot = neural_state_dict(online, target, optimizer)
    destination, destination_target = models()
    destination.layers[0].trainable = False
    restored_optimizer = restore_neural_state(snapshot, destination, destination_target)
    saved = snapshot["online"]["variables"][0]["value"].copy()
    online.variables[0].assign(tf.zeros_like(online.variables[0]))
    np.testing.assert_array_equal(snapshot["online"]["variables"][0]["value"], saved)
    assert destination.variables[0].numpy().tobytes() == saved.tobytes()
    snapshot["online"]["variables"][0]["value"].fill(42)
    assert destination.variables[0].numpy().tobytes() == saved.tobytes()
    assert arrays(restored_optimizer.variables) == arrays(optimizer.variables)


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "runtime",
        "missing",
        "model_count",
        "target_dtype",
        "shape",
        "name",
        "trainable",
        "architecture",
        "optimizer_count",
        "optimizer_dtype",
        "built",
        "optimizer_type",
        "iterations",
    ],
)
def test_invalid_state_does_not_partially_restore_networks(damage):
    online, target = models()
    optimizer = tf.keras.optimizers.Adam()
    step(online, optimizer)
    state = neural_state_dict(online, target, optimizer)
    destination, destination_target = models()
    before = arrays(destination.variables), arrays(destination_target.variables)
    if damage == "version":
        state["version"] = 2
    elif damage == "runtime":
        state["keras_version"] = "incompatible"
    elif damage == "missing":
        del state["target"]
    elif damage == "model_count":
        state["online"]["variables"].pop()
    elif damage == "target_dtype":
        record = state["target"]["variables"][0]
        record["value"] = record["value"].astype(np.float64)
    elif damage == "shape":
        state["online"]["variables"][0]["value"] = np.zeros(1, dtype=np.float32)
    elif damage == "name":
        state["target"]["variables"][0]["name"] = "wrong"
    elif damage == "trainable":
        state["online"]["variables"][0]["trainable"] = False
    elif damage == "architecture":
        state["online"]["config"]["layers"][1]["config"]["activation"] = "relu"
    elif damage == "optimizer_count":
        state["optimizer"]["variables"].pop()
    elif damage == "optimizer_dtype":
        state["optimizer"]["variables"][0] = np.array(1, dtype=np.int32)
    elif damage == "built":
        state["optimizer"]["built"] = False
    elif damage == "optimizer_type":
        state["optimizer"]["config"]["type"] = "CustomOptimizer"
    elif damage == "iterations":
        state["optimizer"]["variables"][0][...] = -1
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with pytest.raises((ValueError, TypeError)):
        restore_neural_state(state, destination, destination_target)
    assert (arrays(destination.variables), arrays(destination_target.variables)) == before
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


@pytest.mark.parametrize(
    "kind", ["ema", "accumulation", "scaling", "exclusions", "binding", "callable"]
)
def test_unsupported_optimizer_state_is_rejected(kind):
    online, target = models()
    config = {
        "ema": {"use_ema": True},
        "accumulation": {"gradient_accumulation_steps": 2},
        "scaling": {"loss_scale_factor": 2.0},
        "exclusions": {},
        "binding": {},
        "callable": {"learning_rate": lambda: 0.01},
    }[kind]
    optimizer = tf.keras.optimizers.Adam(**config)
    if kind == "exclusions":
        optimizer.exclude_from_weight_decay(var_list=online.trainable_variables[:1])
    if kind == "binding":
        optimizer.build(target.trainable_variables)
    with pytest.raises(ValueError):
        neural_state_dict(online, target, optimizer)


def test_shared_models_and_unsupported_layer_are_rejected():
    online, target = models()
    with pytest.raises(ValueError, match="independent"):
        neural_state_dict(online, online, tf.keras.optimizers.Adam())
    unsupported = tf.keras.Sequential(
        [tf.keras.layers.Input((2,)), tf.keras.layers.GaussianNoise(0.1)]
    )
    with pytest.raises(ValueError, match="layer"):
        neural_state_dict(online, unsupported, tf.keras.optimizers.Adam())


def test_model_reconstruction_config_is_retained():
    online, target = models(extras=True)
    snapshot = neural_state_dict(online, target, tf.keras.optimizers.Adam())
    destination = tf.keras.Sequential.from_config(deepcopy(snapshot["online"]["config"]))
    destination_target = tf.keras.Sequential.from_config(
        deepcopy(snapshot["target"]["config"])
    )
    restore_neural_state(snapshot, destination, destination_target)
    assert arrays(destination.variables) == arrays(online.variables)
    assert arrays(destination_target.variables) == arrays(target.variables)


def test_activation_and_flatten_layers_preserve_predictions():
    online = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    target = tf.keras.models.clone_model(online)
    optimizer = tf.keras.optimizers.SGD()
    state = neural_state_dict(online, target, optimizer)
    destination = tf.keras.models.clone_model(online)
    destination_target = tf.keras.models.clone_model(target)
    restore_neural_state(state, destination, destination_target)
    x = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    assert destination(x).numpy().tobytes() == online(x).numpy().tobytes()
    assert destination_target(x).numpy().tobytes() == target(x).numpy().tobytes()


def test_compiled_models_are_rejected():
    online, target = models()
    online.compile(optimizer="adam", loss="mse")
    with pytest.raises(ValueError, match="Compiled models"):
        neural_state_dict(online, target, tf.keras.optimizers.Adam())
