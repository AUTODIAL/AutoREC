DDQN agent
==========

:py:class:`autorec.agent.DDQN_ECM` trains and evaluates a Double Deep Q-Network
for circuit discovery. Construct it with a training environment and, when
needed, a separate evaluation environment. The :doc:`factory` can assemble
these objects from one configuration.

Training, fitting, plot generation, and GIF generation may be expensive and can
write artifacts beneath ``save_dir``. For evaluation without a separate
``eval_env``, pass ``use_eval_env=False`` so the training environment is used.
The current single-EIS implementation selects data by ``EIS_i``; its
``flatten_Z`` and ``ground_truth_circuit`` arguments are retained in the
signature but are replaced with values from the active environment. Pass
``max_actions`` explicitly unless ``action_cap`` was configured on the agent.

.. warning::

   :py:meth:`~autorec.agent.DDQN_ECM.save_model` exports the Keras model only.
   It is not a complete training checkpoint and does not preserve the replay
   buffer, optimizer state, schedules, counters, environments, or random-number
   generator state.

DDQN_ECM
--------

.. autoapiclass:: autorec.agent.DDQN_ECM
   :members: get_active_env, switch_to_training, switch_to_eval, set_training_env, set_eval_env, get_current_env_type, save_model, load_model, train, generate_ECM, eval_batch_eis, eval_all_eis, metadata
   :member-order: bysource
