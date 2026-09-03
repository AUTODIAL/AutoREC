"""Tests for configuration-driven Keras optimizer construction."""

import pytest
import tensorflow as tf

from autorec.factory import (
    create_lr_schedule,
    create_optimizer,
    get_lr_schedule_config,
    get_optimizer_config,
)


def test_create_optimizer_builds_builtin_optimizer_without_mutating_config():
    config = {"type": "SGD", "learning_rate": 0.01, "momentum": 0.9}

    optimizer = create_optimizer(config)

    assert isinstance(optimizer, tf.keras.optimizers.SGD)
    assert float(optimizer.learning_rate.numpy()) == pytest.approx(0.01)
    assert float(optimizer.momentum) == pytest.approx(0.9)
    assert config == {"type": "SGD", "learning_rate": 0.01, "momentum": 0.9}


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ([], TypeError, "must be a dictionary"),
        ({"learning_rate": 0.001}, ValueError, "must contain a 'type'"),
        ({"type": 42}, ValueError, "must be a non-empty string"),
        ({"type": "adam"}, ValueError, "Unknown Keras optimizer type"),
        ({"type": "NotAnOptimizer"}, ValueError, "Unknown Keras optimizer type"),
        (
            {"type": "Adam", "not_an_argument": 1},
            ValueError,
            "Failed to create Keras optimizer",
        ),
    ],
)
def test_create_optimizer_rejects_invalid_configs(config, error_type, message):
    with pytest.raises(error_type, match=message):
        create_optimizer(config)


def test_optimizer_config_round_trip_preserves_builtin_optimizer_settings():
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.002,
        beta_1=0.8,
        beta_2=0.95,
        amsgrad=True,
    )

    config = get_optimizer_config(optimizer)
    rebuilt = create_optimizer(config)

    assert config["type"] == "Adam"
    assert config["learning_rate"] == pytest.approx(0.002)
    assert config["beta_1"] == pytest.approx(0.8)
    assert config["beta_2"] == pytest.approx(0.95)
    assert config["amsgrad"] is True
    assert isinstance(rebuilt, tf.keras.optimizers.Adam)
    assert float(rebuilt.learning_rate.numpy()) == pytest.approx(0.002)
    assert rebuilt.beta_1 == pytest.approx(0.8)
    assert rebuilt.beta_2 == pytest.approx(0.95)
    assert rebuilt.amsgrad is True


def test_get_optimizer_config_rejects_non_optimizer():
    with pytest.raises(TypeError, match="must be a Keras Optimizer instance"):
        get_optimizer_config(object())


def test_create_lr_schedule_builds_schedule_without_mutating_config():
    config = {
        "type": "ExponentialDecay",
        "initial_learning_rate": 0.01,
        "decay_steps": 10,
        "decay_rate": 0.5,
        "staircase": True,
    }

    schedule = create_lr_schedule(config)

    assert isinstance(schedule, tf.keras.optimizers.schedules.ExponentialDecay)
    assert float(schedule(0).numpy()) == pytest.approx(0.01)
    assert float(schedule(10).numpy()) == pytest.approx(0.005)
    assert config == {
        "type": "ExponentialDecay",
        "initial_learning_rate": 0.01,
        "decay_steps": 10,
        "decay_rate": 0.5,
        "staircase": True,
    }


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ([], TypeError, "must be a dictionary"),
        ({}, ValueError, "must contain a 'type'"),
        ({"type": 42}, ValueError, "must be a non-empty string"),
        ({"type": ""}, ValueError, "must be a non-empty string"),
        ({"type": "NotASchedule"}, ValueError, "Unknown Keras learning-rate schedule"),
        (
            {"type": "LearningRateSchedule"},
            ValueError,
            "Unknown Keras learning-rate schedule",
        ),
        (
            {"type": "ExponentialDecay", "not_an_argument": 1},
            ValueError,
            "Failed to create learning-rate schedule",
        ),
    ],
)
def test_create_lr_schedule_rejects_invalid_configs(config, error_type, message):
    with pytest.raises(error_type, match=message):
        create_lr_schedule(config)


def test_lr_schedule_config_round_trip_preserves_settings():
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[10, 20],
        values=[0.01, 0.005, 0.001],
        name="piecewise_learning_rate",
    )

    config = get_lr_schedule_config(schedule)
    rebuilt = create_lr_schedule(config)

    assert config == {
        "type": "PiecewiseConstantDecay",
        "boundaries": [10, 20],
        "values": [0.01, 0.005, 0.001],
        "name": "piecewise_learning_rate",
    }
    assert isinstance(rebuilt, tf.keras.optimizers.schedules.PiecewiseConstantDecay)
    for step in (0, 10, 20):
        assert float(rebuilt(step).numpy()) == pytest.approx(float(schedule(step).numpy()))


def test_get_lr_schedule_config_rejects_non_schedule():
    with pytest.raises(TypeError, match="must be a Keras LearningRateSchedule instance"):
        get_lr_schedule_config(0.001)


def test_optimizer_config_round_trip_preserves_learning_rate_schedule():
    config = {
        "type": "Adam",
        "learning_rate": {
            "type": "CosineDecay",
            "initial_learning_rate": 0.01,
            "decay_steps": 100,
            "alpha": 0.1,
        },
    }

    optimizer = create_optimizer(config)
    optimizer_config = get_optimizer_config(optimizer)
    rebuilt = create_optimizer(optimizer_config)

    assert config["learning_rate"]["type"] == "CosineDecay"
    assert optimizer_config["learning_rate"]["type"] == "CosineDecay"
    assert optimizer_config["learning_rate"]["initial_learning_rate"] == pytest.approx(0.01)
    assert optimizer_config["learning_rate"]["decay_steps"] == 100
    assert optimizer_config["learning_rate"]["alpha"] == pytest.approx(0.1)
    assert get_optimizer_config(rebuilt)["learning_rate"]["type"] == "CosineDecay"

    optimizer.iterations.assign(50)
    rebuilt.iterations.assign(50)
    assert float(rebuilt.learning_rate.numpy()) == pytest.approx(
        float(optimizer.learning_rate.numpy())
    )
