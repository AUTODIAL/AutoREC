"""Exact continuation across independent interpreters with a controlled circuit fitter."""

import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


def run_worker(directory, *, resume=None, horizon=12, count=12, cache="lru"):
    command = [
        sys.executable,
        str(Path(__file__).with_name("checkpoint_worker.py")),
        str(directory),
        "--horizon",
        str(horizon),
        "--count",
        str(count),
        "--cache",
        cache,
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
    )
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=240
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    with (directory / "result.pkl").open("rb") as stream:
        return pickle.load(stream)


@pytest.mark.parametrize("cache", ["lru", "clock"])
def test_fresh_process_continuation(tmp_path, cache):
    continuous = run_worker(tmp_path / "continuous", cache=cache)
    first = run_worker(tmp_path / "first", cache=cache, count=5)
    second = run_worker(tmp_path / "second", resume=tmp_path / "first" / "checkpoint", count=7)
    assert continuous["state"] == second["state"]
    assert continuous["trace"] == first["trace"] + second["trace"]
