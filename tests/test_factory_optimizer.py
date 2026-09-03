"""Tests for configuration-driven Keras optimizer construction."""

import pytest
import tensorflow as tf

from autorec.factory import create_optimizer, get_optimizer_config


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
