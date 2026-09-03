from pathlib import Path
from typing import Dict, Union
import pandas as pd

from autorec.environment import EIS_ECM_Env
from autorec.factory.utils import config_reader


def environment_builder(args: Union[str, Path, Dict]) -> EIS_ECM_Env:
    """Build an EIS_ECM_Env environment from configuration file or dictionary.

    The configurations provided will be treated as keyword arguments for instantiating the
    class. Thus, any required parameters must be included in the configuration, and any
    optional parameters not included will be set to their default values.

    Notes
    -----
    The "dataset" keyword here can be either a pd.DataFrame or a string indicating the path to
    the .pkl processed dataset file. If a path is provided, then the dataset will be loaded
    from the specified file.

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
        config = config_reader(args)
    elif isinstance(args, dict):
        config = args.copy()
    else:
        raise TypeError("The 'args' parameter must be a str, Path, or dict.")

    # Deal with the dataset argument
    if "dataset" not in config:
        raise KeyError("The environment configuration must include a 'dataset' key.")
    if isinstance(config["dataset"], (str, Path)):
        # In this case, we need to load the dataset
        dataset_path = config.pop("dataset")
        dataset = pd.read_pickle(dataset_path)
        config["dataset"] = dataset

    return EIS_ECM_Env(**config)
