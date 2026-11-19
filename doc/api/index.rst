API reference
=============

This reference covers the supported interfaces used by AutoREC's main data
preparation, training, and evaluation workflows. Sphinx derives signatures and
descriptions by parsing the Python source without importing AutoREC or its
scientific dependencies.

Import paths
------------

The main classes are imported from their defining modules:

.. code-block:: python

   from autorec.agent import DDQN_ECM
   from autorec.data_preparation import EISDataPrep
   from autorec.environment import EIS_ECM_Env

Configuration-driven builders and model helpers are re-exported from
``autorec.factory``:

.. code-block:: python

   from autorec.factory import config_reader, pipeline_builder

The individual factory functions are indexed at their implementation modules on
the :doc:`factory` page. This gives every object one canonical documentation
entry while retaining the shorter public import path shown above.

Reference pages
---------------

.. toctree::
   :maxdepth: 1

   data_preparation
   environment
   agent
   factory
   runtime
   utilities
