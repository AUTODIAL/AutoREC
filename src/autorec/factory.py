"""
Factory helpers for building AutoREC environments and agents from configuration.

Configurations can be provided as YAML file paths or dictionaries. Environment configuration
is merged with package defaults, datasets are loaded into pandas DataFrames, and agent
configuration is used to construct ``DDQN_ECM`` instances with the requested training and
evaluation environments.
"""

from pathlib import Path
from typing import Union, Dict
import yaml
import pandas as pd

from autorec.environment import EIS_ECM_Env
from autorec.agent import DDQN_ECM

# Special treatment for example configurations: dataset_path is treated as relative to
# the configuration file instead of the current working directory.
_DEFAULT_CONFIGS_PATH = Path(__file__).resolve().parents[2] / "example" / "default_configs"
_DOCKER_DEFAULT_CONFIGS_PATH = Path("/app") / "default_configs"


def environment_builder(args: Union[str, Path, Dict]) -> EIS_ECM_Env:
    """Build an EIS_ECM_Env environment from configuration.

    The configurations provided will be treated as keyword arguments for instantiating
    EIS_ECM_Env. Thus, any required parameters for EIS_ECM_Env must be included in the
    configuration, and any optional parameters not included will be set to their default
    values as defined in EIS_ECM_Env.

    Note: The dataset_path parameter is required, and the factory will load the dataset from
    the specified path and include it in the environment configuration as a pd.DataFrame.
    The dataset file must be in .csv or .pkl format.

    Parameters
    ----------
    args : Union[str, Path, Dict]
        Configuration for the environment. Can be a YAML file path or a dictionary.

    Returns
    -------
    EIS_ECM_Env
        An instance of the EIS_ECM_Env environment.
    """
    if isinstance(args, (str, Path)):
        config = _config_reader(args)
    elif isinstance(args, dict):
        config = args.copy()
    else:
        raise TypeError("The 'args' parameter must be a str, Path, or dict.")

    # Deal with the dataset argument
    try:
        dataset_path = config.pop("dataset_path")
    except KeyError:
        raise KeyError(
            "The environment configuration must include a 'dataset_path' key (path to a .csv or .pkl dataset). "
            "For regular configs, this path is resolved relative to the current working directory; "
            "configs under example/default_configs (and /app/default_configs in Docker) are resolved relative to the config file."
        )
    dataset = _load_dataset(dataset_path)
    config["dataset"] = dataset

    return EIS_ECM_Env(**config)


# It doesn't make sense to have just agent_builder, since the agent needs an environment.
def environment_and_agent_builder(
    args: Union[str, Path, Dict],
) -> (EIS_ECM_Env, EIS_ECM_Env, DDQN_ECM):
    """Build both an EIS_ECM_Env environment and a DDQN_ECM agent from a single configuration.

    The configuration needs to have "environment" and "agent" sections.

    The configurations provided will be treated as keyword arguments for instantiating
    EIS_ECM_Env and DDQN_ECM. Thus, any required parameters for those classes must be included
    in the configuration, and any optional parameters not included will be set to their default
    values.

    Note: The dataset_path parameter is required for the environment configuration, and the
    factory will load the dataset from the specified path and include it in the environment
    configuration as a pd.DataFrame. The dataset file must be in .csv or .pkl format.

    Parameters
    ----------
    args : Union[str, Path, Dict]
        Configuration for the environment and agent. Can be a YAML file path or a dictionary.

    Returns
    -------
    (EIS_ECM_Env, EIS_ECM_Env, DDQN_ECM)
        A tuple containing two instances of the EIS_ECM_Env environments (for training and
        evaluation, in which the latter can be None) and the DDQN_ECM agent.
    """
    if isinstance(args, (str, Path)):
        config = _config_reader(args)
    elif isinstance(args, dict):
        config = args
    else:
        raise TypeError("The 'args' parameter must be a str, Path, or dict.")

    # Create the environment(s)
    if "environment" not in config:  # Environment section must be included
        raise KeyError("The configuration must include an 'environment' section.")
    print("Creating environment(s)...")
    env_config = config["environment"]
    if ("training" not in env_config) and ("eval" not in env_config):
        # Just a single environment configuration is provided, so we use it for training,
        # since that's the required one.
        training_env = environment_builder(env_config)
        eval_env = None
    elif ("training" in env_config) and ("eval" not in env_config):
        # Only training environment configuration is provided, and it is still ok.
        training_env = environment_builder(env_config["training"])
        eval_env = None
    elif ("training" in env_config) and ("eval" in env_config):
        # Both training and evaluation environment configurations are provided.
        training_env = environment_builder(env_config["training"])
        eval_env = environment_builder(env_config["eval"])
    else:
        raise KeyError(
            "The 'environment' section must include either 'training' or both "
            "'training' and 'eval' subsections."
        )

    # Create the agent
    if "agent" not in config:  # Agent section must be included
        raise KeyError("The configuration must include an 'agent' section.")
    print("Creating agent...")
    agent_config = config["agent"]
    if agent_config is None:
        print("Warning: The 'agent' section is empty. Using default agent configuration.")
        agent_config = {}
    # Insert the environment(s) into the agent configuration
    agent_config["training_env"] = training_env
    agent_config["eval_env"] = eval_env
    # Finally, create the agent
    agent = DDQN_ECM(**agent_config)

    return training_env, eval_env, agent


def _yaml_reader(file_path: Union[str, Path]) -> Dict:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the YAML file.

    Returns
    -------
    Dict
        Contents of the YAML file as a dictionary.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The configuration file {file_path} does not exist.")
    if Path(file_path).suffix not in [".yaml", ".yml"]:
        raise ValueError(
            "The configuration file must be a YAML file with .yaml or .yml extension."
        )
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def _config_reader(file_path: Union[str, Path]) -> Dict:
    """Read a config file and apply default_configs path conventions."""
    config = _yaml_reader(file_path)
    config_path = Path(file_path).resolve()
    default_config_paths = (
        _DEFAULT_CONFIGS_PATH.resolve(),
        _DOCKER_DEFAULT_CONFIGS_PATH,
    )
    if config_path.parent not in default_config_paths:
        return config

    config = config.copy()
    env_configs = []

    if "environment" not in config:
        env_configs = [config]
    elif isinstance(config["environment"], dict):
        env_config = config["environment"].copy()
        config["environment"] = env_config
        for section in ("training", "eval"):
            if section in env_config and isinstance(env_config[section], dict):
                env_config[section] = env_config[section].copy()
                env_configs.append(env_config[section])
        if not env_configs:
            env_configs = [env_config]

    for env_config in env_configs:
        if "dataset_path" in env_config:
            env_config["dataset_path"] = config_path.parent / env_config["dataset_path"]

    return config


def _load_dataset(path: Union[str, Path]) -> Dict:
    """Load dataset from a given path.

    This is for the environment builder, since EIS_ECM_Env may require a dataset in a
    pd.DataFrame format.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The dataset file {path} does not exist.")
    if path.suffix not in [".csv", ".pkl"]:
        raise ValueError("The dataset file must be in .csv or .pkl format.")

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".pkl":
        return pd.read_pickle(path)
