Data preparation
================

Use :py:class:`autorec.data_preparation.EISDataPrep` to process raw EIS CSV
files or validate an existing processed dataset. The constructor validates the
input path and mode; call :py:meth:`~autorec.data_preparation.EISDataPrep.load`
to populate ``dataset`` and perform validation.

``mode="process"`` expects a directory of raw CSV files. ``mode="load"``
expects a processed ``.pkl``, ``.pickle``, or ``.csv`` file. Set
``evaluation=True`` when ground-truth circuit labels are required. See the
:doc:`../tutorials/data_preparation` tutorial for the file layout and a complete
workflow.

.. code-block:: python

   from autorec.data_preparation import EISDataPrep

   prep = EISDataPrep("data/examples/training_dataset.pkl", mode="load")
   dataset = prep.load()

EISDataPrep
-----------

.. autoapiclass:: autorec.data_preparation.EISDataPrep
   :members: load, save, get_summary, get_validation_summary, metadata
   :member-order: bysource
