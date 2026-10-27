from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from autorec.agent import DDQN_ECM


def agent_builder(args: Dict) -> DDQN_ECM:
    """Build a DDQN_ECM agent from configuration dictionary.

    The configurations provided will be treated as keyword arguments for instantiating the
    class. Thus, any required parameters must be included in the configuration, and any
    optional parameters not included will be set to their default values.

    Notes
    -----
    This builder only accepts a configuration dictionary, not a YAML file path. To instantiate
    an agent class, we need to have an environment instance, which cannot be passed in through
    a YAML file.

    Parameters
    ----------
    args : Dict
        Configuration for the environment

    Returns
    -------
    DDQN_ECM
        An instance of the DDQN_ECM agent.
    """
    from autorec.agent import DDQN_ECM

    config = args.copy()
    return DDQN_ECM(**config)
