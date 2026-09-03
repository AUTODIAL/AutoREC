"""Configuration-driven environment, agent, and Keras model factories."""

from autorec.factory.dataprep import dataprep_builder
from autorec.factory.environment import environment_builder
from autorec.factory.agent import agent_builder
from autorec.factory.pipeline import pipeline_builder
from autorec.factory.model import create_model, get_model_config
from autorec.factory.optimizer import create_optimizer, get_optimizer_config
from autorec.factory.utils import AUTOREC_ROOT, config_reader

__all__ = [
    "AUTOREC_ROOT",
    "config_reader",
    "dataprep_builder",
    "agent_builder",
    "environment_builder",
    "pipeline_builder",
    "create_model",
    "get_model_config",
    "create_optimizer",
    "get_optimizer_config",
]
