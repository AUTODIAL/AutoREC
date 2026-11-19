Runtime setup
=============

Call :py:func:`autorec.runtime.configure_autorec_runtime` before importing
AutoREC modules that initialize AutoEIS/JuliaCall, TensorFlow, or JAX. The helper
creates writable cache directories, applies thread limits, and optionally warms
up AutoEIS before the other scientific libraries are imported.

.. code-block:: python

   from autorec.runtime import configure_autorec_runtime

   configure_autorec_runtime(thread_count=1, warmup_autoeis=True)

   from autorec.factory import pipeline_builder

Passing ``warmup_autoeis=False`` skips Julia initialization, but callers are
then responsible for safe scientific-library initialization order.

.. autoapifunction:: autorec.runtime.configure_autorec_runtime
