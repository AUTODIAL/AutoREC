Selected utilities
==================

These helpers support the standard scripts and evaluation workflow. Other
functions in ``autorec.utils`` currently serve internal fitting, plotting,
encoding, or experimental workflows and are not part of this curated reference.

Reproducibility
---------------

Call ``set_global_seed`` after runtime setup and before creating environments,
models, or agents. Deterministic TensorFlow operations improve reproducibility,
but the helper does not by itself guarantee identical results across different
hardware or dependency versions.

.. autoapifunction:: autorec.utils.set_global_seed

Evaluation results
------------------

``save_evaluation_results`` writes both CSV and pickle files. CSV is convenient
for inspection; pickle preserves nested Python objects and arrays in evaluation
columns.

.. autoapifunction:: autorec.utils.save_evaluation_results
