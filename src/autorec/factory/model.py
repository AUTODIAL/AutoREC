"""This module contains functions for constructing and initializing TensorFlow models from
specified configurations.
"""

from typing import Any, Sequence

import tensorflow as tf
from tensorflow.keras import layers

LayerConfig = dict[str, Any]


def create_layer(layer_config: dict[str, Any]) -> layers.Layer:
    """
    Create a Keras layer from a configuration dictionary.

    Parameters
    ----------
    layer_config : dict
        A dictionary containing the configuration for the layer. It must include a "type" key
        specifying the type of the layer (e.g., "Dense", "Conv2D", etc.) and other keys
        corresponding to the parameters required to initialize that layer.

    Returns
    -------
    layers.Layer
        An instance of the specified Keras layer initialized with the provided configuration.
    """
    if not isinstance(layer_config, dict):
        raise TypeError("Layer configuration must be a dictionary.")

    config = layer_config.copy()

    # layer_config must include 'type' key to specify the layer type
    try:
        layer_type = config.pop("type")
    except KeyError as exc:
        raise ValueError("Layer configuration must contain a 'type' field.") from exc

    if not isinstance(layer_type, str) or not layer_type:
        raise ValueError("Layer configuration 'type' must be a non-empty string.")

    layer_class = getattr(layers, layer_type, None)
    if not isinstance(layer_class, type) or not issubclass(layer_class, layers.Layer):
        raise ValueError(f"Unknown Keras layer type '{layer_type}'.")

    try:
        return layer_class(**config)
    except Exception as exc:
        raise ValueError(
            f"Failed to create Keras layer '{layer_type}' with configuration {config}."
        ) from exc


def get_layer_config(layer: layers.Layer) -> LayerConfig:
    """Convert a Keras layer into an AutoREC layer configuration."""
    return {"type": layer.__class__.__name__, **layer.get_config()}


def create_model(
    input_layer: LayerConfig,
    hidden_layers: Sequence[LayerConfig],
    output_layer: LayerConfig,
    name: str | None = None,
) -> tf.keras.Sequential:
    """
    Build a Keras Sequential model from layer configurations.

    Parameters
    ----------
    input_layer : dict
        A dictionary specifying the input layer of the model.
    hidden_layers : Sequence[dict]
        A sequence of dictionaries, each specifying a hidden layer in the model.
    output_layer : dict
        A dictionary specifying the output layer of the model.
    name : str, optional
        The name of the model. If not provided, a default name will be assigned.

    Returns
    -------
    tf.keras.Sequential
        A Keras Sequential model constructed from the specified layer configurations.
    """

    model = tf.keras.Sequential(name=name)

    model.add(create_layer(input_layer))  # Input
    for config in hidden_layers:  # Hidden layers
        model.add(create_layer(config))
    model.add(create_layer(output_layer))  # Output

    return model


def get_model_config(
    model: tf.keras.Model,
    include_input: bool = False,
    include_output: bool = False,
) -> list[LayerConfig]:
    """
    Convert a Sequential Keras model into layer configurations.

    Parameters
    ----------
    model : tf.keras.Model
        A Keras Sequential model to be converted into layer configurations.
    include_input : bool, optional
        Whether to include the input layer configuration in the output list. Default is False.
    include_output : bool, optional
        Whether to include the output layer configuration in the output list. Default is False.

    Returns
    -------
    list[dict]
        A list of dictionaries, each representing the configuration of a layer in the model.
    """

    configs: list[LayerConfig] = []

    # Input
    if include_input:
        input_shape = tuple(model.input_shape[1:])
        configs.append({"type": "InputLayer", "shape": input_shape})

    # Keras Sequential models omit their InputLayer from model.layers, while Functional
    # models include it. Normalize both representations before selecting the model body.
    model_layers = [
        layer for layer in model.layers if not isinstance(layer, layers.InputLayer)
    ]

    if not include_output:
        model_layers = model_layers[:-1]

    configs.extend(get_layer_config(layer) for layer in model_layers)

    return configs
