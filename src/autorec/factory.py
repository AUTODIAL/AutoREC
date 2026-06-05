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


# Load default values from a YAML file, in case of missing parameters, in which case we will
# just fill in using the defaults.
_DEFAULT_ENV_CONFIG_PATH = (
    Path(__file__).parent / "default_configs" / "environment_config.yaml"
)
_DEFAULT_AGENT_CONFIG_PATH = Path(__file__).parent / "default_configs" / "agent_config.yaml"
default_env_config = _yaml_reader(_DEFAULT_ENV_CONFIG_PATH)
default_agent_config = _yaml_reader(_DEFAULT_AGENT_CONFIG_PATH)
default_env_agent_config = {
    "environment": default_env_config,
    "agent": default_agent_config,
}


def environment_builder(args: Union[str, Path, Dict]) -> EIS_ECM_Env:
    """Build an EIS_ECM_Env environment from configuration.

    Note: If all configurations or parameters are not provided in the YAML,
    the factory will read the missing values from the default config files.
    The default environment config is stored in environment_config.yaml in the
    src/autorec/default_configs folder.

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
        config = _yaml_reader(args)
    elif isinstance(args, dict):
        config = args
    else:
        raise TypeError("The 'args' parameter must be a str, Path, or dict.")

    # Insert default values for missing parameters
    config = _insert_defaults(config, default_env_config)
    # Deal with the dataset argument
    dataset_path = config.pop("dataset_path")
    dataset = _load_dataset(dataset_path)
    config["dataset"] = dataset

    return EIS_ECM_Env(**config)


# It doesn't make sense to have just agent_builder, since the agent needs an environment.
def environment_and_agent_builder(
    args: Union[str, Path, Dict],
) -> (EIS_ECM_Env, EIS_ECM_Env, DDQN_ECM):
    """Build both an EIS_ECM_Env environment and a DDQN_ECM agent from a single configuration.

    The configuration needs to have "environment" and "agent" sections.

    Note: If all configurations or parameters are not provided in the YAML,
    the factory will read the missing values from the default config files.
    The default configs are stored in environment_config.yaml and agent_config.yaml
    in the src/autorec/default_configs folder.

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
        config = _yaml_reader(args)
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
    # Insert default values for missing parameters
    agent_config = _insert_defaults(agent_config, default_agent_config)
    # Insert the environment(s) into the agent configuration
    agent_config["training_env"] = training_env
    agent_config["eval_env"] = eval_env
    # Finally, create the agent
    agent = DDQN_ECM(**agent_config)

    return training_env, eval_env, agent


def _insert_defaults(config: Dict, default_config: Dict) -> Dict:
    """Insert default values into the configuration dictionary for any missing keys.

    Parameters
    ----------
    config : Dict
        The configuration dictionary to be filled with defaults.
    default_config : Dict
        The default configuration dictionary.

    Returns
    -------
    Dict
        The configuration dictionary with defaults inserted.
    """
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            config[key] = _insert_defaults(config.get(key, {}), value)
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
