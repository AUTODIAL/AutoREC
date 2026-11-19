Circuit-discovery environment
=============================

:py:class:`autorec.environment.EIS_ECM_Env` represents circuit construction as
a reinforcement-learning environment. It consumes a processed dataset, creates
the chromosome and action spaces, and evaluates circuit mutations against the
active EIS measurement.

Construct the environment after calling
:py:func:`autorec.runtime.configure_autorec_runtime`. Calling
:py:meth:`~autorec.environment.EIS_ECM_Env.step` can perform circuit fitting;
it is therefore substantially more expensive than ordinary state bookkeeping.
The optional history cache stores repeated circuit evaluations for the same EIS
input.

.. code-block:: python

   from autorec.environment import EIS_ECM_Env

   env = EIS_ECM_Env(dataset=dataset, seed=42)
   eis_index, observation = env.reset()

EIS_ECM_Env
-----------

.. autoapiclass:: autorec.environment.EIS_ECM_Env
   :members: all_actions_list, reset, step, get_cache_stats, print_cache_stats, clear_cache, metadata
   :member-order: bysource
