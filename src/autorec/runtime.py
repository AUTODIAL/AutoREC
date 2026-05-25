"""Runtime setup helpers for AutoREC scripts.

This module intentionally avoids importing TensorFlow, JAX, NumPy, Matplotlib, or AutoEIS at
import time. Call ``configure_autorec_runtime`` before importing the rest of AutoREC in
standalone scripts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_autorec_runtime(
    thread_count: int | str = 1,
    warmup_autoeis: bool = True,
    suppress_tf_logs: bool = True,
) -> Path:
    """Configure native-library runtime settings for standalone AutoREC scripts.

    AutoREC uses AutoEIS/JuliaCall together with TensorFlow/JAX. In standalone terminal runs,
    initializing TensorFlow/JAX before Julia can crash the Python process at the native-library
    level. This helper sets writable runtime directories, limits BLAS/OpenMP thread pools, and
    optionally initializes AutoEIS/Julia before TensorFlow/JAX are imported.

    Args:
        thread_count: Thread count used for OpenMP, MKL, OpenBLAS, and NumExpr.
        warmup_autoeis: If True, initialize AutoEIS/Julia immediately.
        suppress_tf_logs: If True, suppress TensorFlow INFO/WARNING logs.

    Returns:
        The runtime cache directory used for JuliaCall and Matplotlib.
    """
    if sys.platform == "darwin":
        runtime_dir = Path.home() / "Library" / "Caches" / "autorec_eis"
    else:
        runtime_dir = Path.home() / ".cache" / "autorec_eis"

    julia_env_dir = runtime_dir / "julia_env"
    julia_depot_dir = runtime_dir / "julia_depot"
    matplotlib_dir = runtime_dir / "matplotlib"

    julia_env_dir.mkdir(parents=True, exist_ok=True)
    julia_depot_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(julia_env_dir))
    os.environ.setdefault("JULIA_DEPOT_PATH", str(julia_depot_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")

    thread_count = str(thread_count)
    os.environ.setdefault("OMP_NUM_THREADS", thread_count)
    os.environ.setdefault("MKL_NUM_THREADS", thread_count)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_count)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", thread_count)

    if suppress_tf_logs:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR) -
        # example: you are using different TensorFlow than the trained model you are loading

    if warmup_autoeis:
        import autoeis as ae

        ae.core.ec.karva_to_tree("+RR")

    return runtime_dir
