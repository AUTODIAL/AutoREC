"""This module provides a utility function to create Keras optimizers from a configuration
dictionary.
"""

from typing import Any

import tensorflow as tf


OptimizerConfig = dict[str, Any]


def create_optimizer(
    optimizer_config: OptimizerConfig,
) -> tf.keras.optimizers.Optimizer:
    """
    Create a Keras optimizer from a configuration dictionary.

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the optimizer configuration. It must include a
        "type" key specifying the optimizer type, with all remaining entries
        passed as keyword arguments to the optimizer.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        Initialized Keras optimizer.
    """
    if not isinstance(optimizer_config, dict):
        raise TypeError("Optimizer configuration must be a dictionary.")

    config = optimizer_config.copy()

    # optimizer_config must include 'type' key to specify the optimizer type
    try:
        optimizer_type = config.pop("type")
    except KeyError as exc:
        raise ValueError("Optimizer configuration must contain a 'type' field.") from exc

    if not isinstance(optimizer_type, str) or not optimizer_type:
        raise ValueError("Optimizer configuration 'type' must be a non-empty string.")

    optimizer_class = getattr(tf.keras.optimizers, optimizer_type, None)
    if not isinstance(optimizer_class, type) or not issubclass(
        optimizer_class,
        tf.keras.optimizers.Optimizer,
    ):
        raise ValueError(f"Unknown Keras optimizer type '{optimizer_type}'.")

    try:
        return optimizer_class(**config)
    except Exception as exc:
        raise ValueError(
            f"Failed to create Keras optimizer '{optimizer_type}' with configuration {config}."
        ) from exc


def get_optimizer_config(
    optimizer: tf.keras.optimizers.Optimizer,
) -> OptimizerConfig:
    """
    Convert a Keras optimizer into an optimizer configuration dictionary.

    Parameters
    ----------
    optimizer : tf.keras.optimizers.Optimizer
        The Keras optimizer to convert.

    Returns
    -------
    dict
        A dictionary containing the optimizer type and all configuration
        parameters needed to reconstruct it.
    """
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise TypeError("optimizer must be a Keras Optimizer instance.")

    return {"type": optimizer.__class__.__name__, **optimizer.get_config()}
