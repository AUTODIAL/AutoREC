"""Collection of utility functions for factory and configuration management in AutoREC."""

from pathlib import Path
import os
from typing import Dict, Union
import yaml


# Define AUTOREC_ROOT environment variable, which points to the root directory of the AutoREC
# package. This environment variable will be used by the provided example configuration files.
AUTOREC_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("AUTOREC_ROOT", str(AUTOREC_ROOT))


def config_reader(file_path: Union[str, Path]) -> Dict:
    """Read a YAML configuration file and return its contents as a dictionary.

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
    # Expand environment variables in the config file
    config_text = os.path.expandvars(Path(file_path).read_text())

    config = yaml.safe_load(config_text)
    # Validate that the YAML file contains a dictionary
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a dictionary at the top level.")
    return config
