"""Tests for configuration-driven Keras layer and model construction."""

import pytest
import tensorflow as tf

from autorec.factory import create_model, get_model_config
from autorec.factory.model import create_layer


def layer_types(model):
    """Return stable class names for the layers exposed by a Keras model."""
    return [layer.__class__.__name__ for layer in model.layers]


def test_create_layer_builds_builtin_layer_without_mutating_config():
    config = {"type": "Dropout", "rate": 0.25, "seed": 7}

    layer = create_layer(config)

    assert isinstance(layer, tf.keras.layers.Dropout)
    assert layer.rate == pytest.approx(0.25)
    assert config == {"type": "Dropout", "rate": 0.25, "seed": 7}


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ([], TypeError, "must be a dictionary"),
        ({"units": 4}, ValueError, "must contain a 'type'"),
        ({"type": 42}, ValueError, "must be a non-empty string"),
        ({"type": "NotAKerasLayer"}, ValueError, "Unknown Keras layer type"),
        ({"type": "Dense", "not_an_argument": 1}, ValueError, "Failed to create"),
    ],
)
def test_create_layer_rejects_invalid_configs(config, error_type, message):
    with pytest.raises(error_type, match=message):
        create_layer(config)


def test_create_model_builds_generic_sequential_stack():
    model = create_model(
        input_layer={"type": "InputLayer", "shape": (4,)},
        hidden_layers=[
            {"type": "Dense", "units": 8, "activation": "relu"},
            {"type": "Dropout", "rate": 0.25},
            {"type": "LayerNormalization"},
        ],
        output_layer={"type": "Dense", "units": 2, "activation": "linear"},
        name="configured_model",
    )

    assert model.name == "configured_model"
    assert model.input_shape == (None, 4)
    assert model.output_shape == (None, 2)
    assert layer_types(model) == ["Dense", "Dropout", "LayerNormalization", "Dense"]


def test_model_config_round_trip_preserves_layer_architecture():
    model = create_model(
        input_layer={"type": "InputLayer", "shape": (4,)},
        hidden_layers=[
            {"type": "Dense", "units": 8, "activation": "elu"},
            {"type": "Dropout", "rate": 0.1},
        ],
        output_layer={"type": "Dense", "units": 2, "activation": "linear"},
    )

    config = get_model_config(model, include_input=True, include_output=True)
    rebuilt = create_model(config[0], config[1:-1], config[-1])

    assert [item["type"] for item in config] == ["InputLayer", "Dense", "Dropout", "Dense"]
    assert rebuilt.input_shape == model.input_shape
    assert rebuilt.output_shape == model.output_shape
    assert layer_types(rebuilt) == layer_types(model)
    assert rebuilt.layers[0].units == 8
    assert rebuilt.layers[0].activation.__name__ == "elu"
    assert rebuilt.layers[1].rate == pytest.approx(0.1)


def test_get_model_config_excludes_functional_input_and_output_by_default():
    inputs = tf.keras.layers.Input(shape=(4,))
    hidden = tf.keras.layers.Dense(8, activation="relu")(inputs)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    outputs = tf.keras.layers.Dense(2, activation="linear")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    config = get_model_config(model)

    assert [item["type"] for item in config] == ["Dense", "Dropout"]
    assert config[0]["units"] == 8
    assert config[1]["rate"] == pytest.approx(0.2)
