from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Union

from autorec.factory.utils import config_reader
from autorec.factory.dataprep import dataprep_builder
from autorec.factory.environment import environment_builder
from autorec.factory.agent import agent_builder

if TYPE_CHECKING:
    from autorec.data_preparation import EISDataPrep
    from autorec.environment import EIS_ECM_Env
    from autorec.agent import DDQN_ECM


def pipeline_builder(
    args: Union[str, Path, Dict],
) -> (EISDataPrep, EISDataPrep, EIS_ECM_Env, EIS_ECM_Env, DDQN_ECM):
    """
    Builder for the complete pipeline, including the EISDataPrep data preparations, EIS_ECM_Env
    environments, and a DDQN_ECM agent from a single configuration file or dictionary.

    Notes
    -----
    * The available sections in the configurations are: "dataprep", "environment", and "agent".
    * The "dataprep" section is optional. If not provided, then "dataset" key must be specified
      in the "environment" section.
    * The "dataprep" and "environment" sections can have 2 subsections: "training" and
      "eval", specifying the training and evaluation datasets and environments
      separately. If no subsection is detected, then the training and evaluation
      datasets/environments will be the same.
    * If "dataprep" section is available, then the keyword "dataset" in "environment" section
      is optional. The "dataset" value will be replaced with the output of the "dataprep"
      section, even if this keyword already exists.
    * The "training_env" and "eval_env" keywords in the "agent" section are optional. They will
      be replaced by the output of the "environment" section, even if they already exist.

    Parameters
    ----------
    args : Union[str, Path, Dict]
        Configuration for the environment and agent. Can be a YAML file path or a dictionary.

    Returns
    -------
    (EISDataPrep, EISDataPrep, EIS_ECM_Env, EIS_ECM_Env, DDQN_ECM)
        A tuple containing two instances of the EISDataPrep data preparations and EIS_ECM_Env
        environments (for training and evaluation, in which the latter can be None) and the
        DDQN_ECM agent.
    """
    # Read the configuration
    if isinstance(args, (str, Path)):
        config = config_reader(args)
    elif isinstance(args, dict):
        config = deepcopy(args)
    else:
        raise TypeError("The 'args' parameter must be a str, Path, or dict.")

    # Validate configuration
    _validate_pipeline_config(config)

    # Separate the sections and subsections configurations
    section_configs = {
        "dataprep_training": None,
        "dataprep_eval": None,
        "environment_training": None,
        "environment_eval": None,
        "agent": None,
    }
    if "dataprep" in config:
        if "training" in config["dataprep"]:
            section_configs["dataprep_training"] = config["dataprep"]["training"]
        else:
            section_configs["dataprep_training"] = config["dataprep"]
        if "eval" in config["dataprep"]:
            section_configs["dataprep_eval"] = config["dataprep"]["eval"]
    if "environment" in config:
        if "training" in config["environment"]:
            section_configs["environment_training"] = config["environment"]["training"]
        else:
            section_configs["environment_training"] = config["environment"]
        if "eval" in config["environment"]:
            section_configs["environment_eval"] = config["environment"]["eval"]
    if "agent" in config:
        section_configs["agent"] = config["agent"]

    # Data preparation
    if section_configs["dataprep_training"] is not None:
        dataprep_training, dataset_training = dataprep_builder(
            section_configs["dataprep_training"]
        )
        if section_configs["dataprep_eval"] is not None:
            dataprep_eval, dataset_eval = dataprep_builder(section_configs["dataprep_eval"])
        else:
            dataprep_eval = dataprep_training
            dataset_eval = dataset_training.copy()
    else:
        dataprep_training = None
        dataset_training = None
        dataprep_eval = None
        dataset_eval = None

    # Environment
    if dataprep_training is not None:
        section_configs["environment_training"]["dataset"] = dataset_training
        if section_configs["environment_eval"] is not None:
            section_configs["environment_eval"]["dataset"] = dataset_eval
    env_training = environment_builder(section_configs["environment_training"])
    if section_configs["environment_eval"] is not None:
        env_eval = environment_builder(section_configs["environment_eval"])
    else:
        env_eval = None

    # Agent
    section_configs["agent"]["training_env"] = env_training
    section_configs["agent"]["eval_env"] = env_eval
    agent = agent_builder(section_configs["agent"])

    return (dataprep_training, dataprep_eval, env_training, env_eval, agent)


def _validate_pipeline_config(config):
    """Perform some validation checks on the pipeline configuration dictionary.

    Validation checks include:
    * The configuration must be a dictionary.
    * If the "dataprep" section is included and an "eval" subsection is included, then
      the "training" subsection must also be included.
    * The configuration must include an "environment" section.
    * If the "environment" section is included and an "eval" subsection is included,
      then the "training" subsection must also be included.
    * If the "dataprep" section is not included, then the "environment" section must include a
      "dataset" key.
    * The configuration must include an "agent" section.
    * The "agent" section must not have "training_env" and "eval_env" keys
    """
    # Check that the configuration is a dictionary
    if not isinstance(config, dict):
        raise TypeError("The configuration must be a dictionary.")

    # dataprep section
    if "dataprep" in config:
        if "eval" in config["dataprep"] and "training" not in config["dataprep"]:
            raise KeyError(
                "If the 'dataprep' section includes an 'eval' subsection, "
                "it must also include a 'training' subsection."
            )

    # environment section
    if "environment" not in config:
        raise KeyError("The configuration must include an 'environment' section.")
    if "eval" in config["environment"] and "training" not in config["environment"]:
        raise KeyError(
            "If the 'environment' section includes an 'eval' subsection, "
            "it must also include a 'training' subsection."
        )
    msg = (
        "If the 'dataprep' section is not included, "
        "the 'environment' section must include a 'dataset' key."
    )
    if "dataprep" not in config:
        if (
            "training" in config["environment"]
            and "dataset" not in config["environment"]["training"]
        ):
            raise KeyError(msg)
        if "eval" in config["environment"] and "dataset" not in config["environment"]["eval"]:
            raise KeyError(msg)
        if "training" not in config["environment"] and "eval" not in config["environment"]:
            if "dataset" not in config["environment"]:
                raise KeyError(msg)

    # agent section
    if "agent" not in config:
        raise KeyError("The configuration must include an 'agent' section.")
