Configuration factories
=======================

The factory interface creates data preparation, environment, agent, model, and
optimizer objects from dictionaries or YAML configuration files. Import the
documented functions from ``autorec.factory`` even though each object below is
indexed at its implementation module.

.. code-block:: python

   from autorec.factory import config_reader, pipeline_builder

   config = config_reader("configs_yaml/examples/demo_pipeline_config.yaml")
   dataprep, dataprep_eval, env, env_eval, agent = pipeline_builder(config)

``pipeline_builder`` copies the supplied top-level dictionary before wiring the
objects. It returns training and evaluation data-preparation objects and
environments followed by the agent. Without a data-preparation ``eval``
subsection, the evaluation data-preparation result refers to the training
preparation object. Without an environment ``eval`` subsection, the separate
evaluation environment is ``None``.

Configuration loading
---------------------

``config_reader`` expands environment variables in the YAML text and requires a
mapping at the document root.

``AUTOREC_ROOT`` is the repository root derived from the factory module's source
path. Importing the factory utilities publishes it as the ``AUTOREC_ROOT``
environment variable only when that variable is not already set, allowing the
example YAML files to use ``${AUTOREC_ROOT}`` paths.

.. autoapidata:: autorec.factory.utils.AUTOREC_ROOT

.. autoapifunction:: autorec.factory.utils.config_reader

Pipeline and component builders
-------------------------------

.. autoapifunction:: autorec.factory.pipeline.pipeline_builder

.. autoapifunction:: autorec.factory.dataprep.dataprep_builder

.. autoapifunction:: autorec.factory.environment.environment_builder

.. autoapifunction:: autorec.factory.agent.agent_builder

Model helpers
-------------

.. autoapidata:: autorec.factory.model.LayerConfig

.. autoapifunction:: autorec.factory.model.create_model

.. autoapifunction:: autorec.factory.model.get_model_config

Optimizer and schedule helpers
------------------------------

.. autoapidata:: autorec.factory.optimizer.OptimizerConfig

.. autoapifunction:: autorec.factory.optimizer.create_optimizer

.. autoapifunction:: autorec.factory.optimizer.get_optimizer_config

.. autoapifunction:: autorec.factory.optimizer.create_lr_schedule

.. autoapifunction:: autorec.factory.optimizer.get_lr_schedule_config
