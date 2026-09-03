"""Utilities for creating Keras optimizers and learning-rate schedules from configuration
dictionaries.
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
        Dictionary containing the optimizer configuration. It must include a "type" key
        specifying the optimizer type, with all remaining entries passed as keyword arguments
        to the optimizer.

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

    # Construct learning-rate schedule if one is specified.
    learning_rate = config.get("learning_rate")
    if isinstance(learning_rate, dict):
        config["learning_rate"] = create_lr_schedule(learning_rate)

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

    # We can get the configuration of the learning rate from the serialized optimizer, but we
    # need to convert the format of the learning rate configuration to match the expected
    # format for creating a new optimizer.
    config = optimizer.get_config()
    learning_rate = config.get("learning_rate")
    if (
        isinstance(learning_rate, dict)
        and "class_name" in learning_rate
        and "config" in learning_rate
    ):
        config["learning_rate"] = {
            "type": learning_rate["class_name"],
            **learning_rate["config"],
        }

    return {
        "type": optimizer.__class__.__name__,
        **config,
    }


def create_lr_schedule(
    schedule_config: dict[str, Any],
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Create a Keras learning-rate schedule from a configuration dictionary.

    Parameters
    ----------
    schedule_config : dict
        Dictionary containing the learning-rate schedule configuration. It must include a
        "type" key specifying the schedule type, with all remaining entries passed as
        keyword arguments to the schedule.

    Returns
    -------
    tf.keras.optimizers.schedules.LearningRateSchedule
        Initialized Keras learning-rate schedule.
    """

    if not isinstance(schedule_config, dict):
        raise TypeError("Learning-rate schedule configuration must be a dictionary.")

    config = schedule_config.copy()

    # schedule_config must include 'type' key to specify the schedule type
    try:
        schedule_type = config.pop("type")
    except KeyError as exc:
        raise ValueError(
            "Learning-rate schedule configuration must contain a 'type' field."
        ) from exc

    if not isinstance(schedule_type, str) or not schedule_type:
        raise ValueError(
            "Learning-rate schedule configuration 'type' must be a non-empty string."
        )

    try:
        schedule = tf.keras.optimizers.schedules.deserialize(
            {"class_name": schedule_type, "config": config}
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to create learning-rate schedule "
            f"'{schedule_type}' with configuration {config}."
        ) from exc

    if (
        not isinstance(schedule, tf.keras.optimizers.schedules.LearningRateSchedule)
        or type(schedule) is tf.keras.optimizers.schedules.LearningRateSchedule
    ):
        raise ValueError(f"Unknown Keras learning-rate schedule type '{schedule_type}'.")

    return schedule


def get_lr_schedule_config(
    schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
) -> dict[str, Any]:
    """Convert a Keras learning-rate schedule into a configuration dictionary.

    Parameters
    ----------
    schedule : tf.keras.optimizers.schedules.LearningRateSchedule
        The Keras learning-rate schedule to convert.

    Returns
    -------
    dict
        A dictionary containing the schedule type and all configuration parameters
        needed to reconstruct it.
    """

    if not isinstance(schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
        raise TypeError("schedule must be a Keras LearningRateSchedule instance.")

    return {"type": schedule.__class__.__name__, **schedule.get_config()}
