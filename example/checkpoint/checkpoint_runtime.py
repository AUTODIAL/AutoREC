"""Shared deterministic runtime setup for the checkpoint example scripts."""

from __future__ import annotations

import os
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = DIRECTORY.parents[1]
RUNTIME_CACHE = DIRECTORY / "outputs/runtime_cache"


def configure_cache_environment():
    """Set writable example-local caches before importing numerical libraries."""
    RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUTOREC_ROOT", str(REPOSITORY_ROOT))
    os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(RUNTIME_CACHE / "julia_env"))
    os.environ.setdefault("JULIA_DEPOT_PATH", str(RUNTIME_CACHE / "julia_depot"))


def configure_tensorflow_for_checkpoint(tf):
    """Match the deterministic TensorFlow settings recorded in checkpoints."""
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def deterministic_subprocess_environment():
    """Return an environment for a deterministic CPU checkpoint subprocess."""
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONHASHSEED="0",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    return environment
