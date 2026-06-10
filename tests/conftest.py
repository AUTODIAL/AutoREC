"""Shared pytest setup for tests that import native/scientific dependencies.

Several AutoREC modules import AutoEIS, ArviZ, Matplotlib, JuliaCall, or TensorFlow at
module import time. These libraries write cache/config files by default, so the test suite
redirects those paths to /tmp before any test modules import AutoREC internals.
"""

import os
from pathlib import Path


_TEST_CACHE_DIR = Path("/tmp") / "autorec_pytest_cache"
_TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(_TEST_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_TEST_CACHE_DIR / "matplotlib"))
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(_TEST_CACHE_DIR / "julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(_TEST_CACHE_DIR / "julia_depot"))
os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
