from pathlib import Path
from typing import Dict, Union, Tuple
import pandas as pd

from autorec.data_preparation import EISDataPrep
from autorec.factory.utils import config_reader


def dataprep_builder(args: Union[str, Path, Dict]) -> Tuple[EISDataPrep, pd.DataFrame]:
    """Build an EISDataPrep data preparation class from configuration file or dictionary.

    The configurations provided will be treated as keyword arguments for instantiating the
    class. Thus, any required parameters must be included in the configuration, and any
    optional parameters not included will be set to their default values.

    Notes
    -----
    An additional keyword "output" may be provided. If provided, the processed dataset will be
    exported to the specified pickle file.

    Parameters
    ----------
    args : Union[str, Path, Dict]
        Configuration for the environment. Can be a YAML file path or a dictionary.

    Returns
    -------
    EISDataPrep
        An instance of the EISDataPrep environment.
    dataset
        Pandas data frame for the processed EIS dataset.
    """
    if isinstance(args, (str, Path)):
        config = config_reader(args)
    elif isinstance(args, dict):
        config = args.copy()
    else:
        raise TypeError("The 'args' parameter must be a str, Path, or dict.")

    # "output" is not part of the argument of EISDataPrep
    if "output" in config:
        output = config.pop("output")
    else:
        output = None

    # Main
    dataprep = EISDataPrep(**config)
    dataset = dataprep.load()
    # Export the dataset pickle file
    if output is not None:
        dataset.to_pickle(output)

    return dataprep, dataset
