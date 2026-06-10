"""Tests for runtime setup helpers that prepare writable cache/config directories."""

import os
from pathlib import Path

from autorec.runtime import configure_autorec_runtime


def test_configure_autorec_runtime_sets_expected_environment(monkeypatch, tmp_path):
    """Runtime setup should create directories and set defaults when env vars are absent."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("PYTHON_JULIAPKG_PROJECT", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    runtime_dir = configure_autorec_runtime(
        thread_count=2, warmup_autoeis=False, suppress_tf_logs=True
    )

    assert runtime_dir == tmp_path / ".cache" / "autorec_eis"
    assert (runtime_dir / "julia_env").is_dir()
    assert (runtime_dir / "julia_depot").is_dir()
    assert (runtime_dir / "matplotlib").is_dir()
    assert Path(os.environ["PYTHON_JULIAPKG_PROJECT"]) == runtime_dir / "julia_env"
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "2"
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"


def test_configure_autorec_runtime_does_not_override_existing_values(monkeypatch, tmp_path):
    """Existing process-level settings should win over configure_autorec_runtime defaults."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "1")

    configure_autorec_runtime(thread_count=2, warmup_autoeis=False, suppress_tf_logs=True)

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "1"
